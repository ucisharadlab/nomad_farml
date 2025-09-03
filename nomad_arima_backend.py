import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from scipy.stats import entropy
import random
import traceback
import os
import importlib.util
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Import all supported classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

# ARIMA and statistical dependencies
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    ARIMA_AVAILABLE = True
except ImportError:
    print("Warning: statsmodels not available. Falling back to simple adaptive priors.")
    ARIMA_AVAILABLE = False

# --- Constants ---
ALPHA_SMOOTHING = 1
RANDOM_SEED_FOR_SAMPLING = 42

# --- Model Instantiation ---
# Dictionary to map string names to classifier classes
CLASSIFIER_MAP = {
    "LogisticRegression": LogisticRegression,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "SVC": SVC,
    "GaussianNB": GaussianNB,
    "KNeighborsClassifier": KNeighborsClassifier,
    "DummyClassifier": DummyClassifier,
}

def get_classifier_from_config(model_config, custom_models_path):
    """
    Instantiates a classifier from its configuration dictionary.
    Handles both standard sklearn models and custom uploaded models.
    """
    model_type = model_config.get("type")
    params = model_config.get("params", {})
    
    if model_config.get("is_custom"):
        module_name = model_config.get("module_name")
        class_name = model_config.get("class_name")
        if not module_name or not class_name:
            raise ValueError(f"Custom model '{model_config['name']}' is missing module or class name.")
        
        filepath = os.path.join(custom_models_path, f"{module_name}.py")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Custom model file not found for '{model_config['name']}' at {filepath}")
        
        try:
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            custom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_module)
            ClassifierClass = getattr(custom_module, class_name)
        except Exception as e:
            raise ImportError(f"Failed to load custom model '{class_name}' from '{filepath}': {e}")
    else:
        if model_type not in CLASSIFIER_MAP:
            raise ValueError(f"Unknown model type: '{model_type}'")
        ClassifierClass = CLASSIFIER_MAP[model_type]
        
    # Filter params to only include those accepted by the constructor
    # This avoids errors if extra params like 'random_state' are passed to models that don't use it
    valid_params = {k: v for k, v in params.items() if k in ClassifierClass().get_params()}

    return ClassifierClass(**valid_params)


# --- Page-Hinkley Test Implementation ---
class PageHinkleyTest:
    """
    Implementation of the Page-Hinkley test for drift detection.
    """
    def __init__(self, threshold=50.0, delta=0.005):
        """
        Initialize Page-Hinkley test.
        
        Args:
            threshold (float): Detection threshold λ
            delta (float): Tolerance parameter for drift detection
        """
        self.threshold = threshold
        self.delta = delta
        self.reset()
    
    def reset(self):
        """Reset the test to initial state."""
        self.m_t = 0.0  # Cumulative sum
        self.M_t = 0.0  # Minimum value reached
        self.sum_x = 0.0
        self.n = 0
        
    def update(self, x):
        """
        Update the test with a new value and check for drift.
        
        Args:
            x (float): New residual value
            
        Returns:
            bool: True if drift is detected, False otherwise
        """
        self.n += 1
        self.sum_x += x
        
        # Calculate running average
        avg = self.sum_x / self.n if self.n > 0 else 0
        
        # Update cumulative sum (adjusted by average and delta)
        self.m_t += (x - avg - self.delta)
        
        # Update minimum
        self.M_t = min(self.M_t, self.m_t)
        
        # Check for drift
        drift_detected = (self.m_t - self.M_t) > self.threshold
        
        return drift_detected


# --- ARIMA-based Adaptive Priors Manager ---
class AdaptivePriorsManager:
    """
    Manages adaptive class priors using ARIMA models and Page-Hinkley test.
    """
    def __init__(self, class_names, initial_priors, 
                 ph_threshold=50.0, ph_delta=0.005, 
                 buffer_size=100, min_buffer_for_arima=20,
                 arima_order=(1, 0, 1), enable_arima=True):
        """
        Initialize adaptive priors manager.
        
        Args:
            class_names (list): List of class names
            initial_priors (dict): Initial class probabilities
            ph_threshold (float): Page-Hinkley threshold
            ph_delta (float): Page-Hinkley delta parameter
            buffer_size (int): Size of event buffer for retraining
            min_buffer_for_arima (int): Minimum buffer size before using ARIMA
            arima_order (tuple): ARIMA order (p, d, q)
            enable_arima (bool): Whether to use ARIMA (falls back to simple if False)
        """
        self.class_names = class_names
        self.current_priors = initial_priors.copy()
        self.initial_priors = initial_priors.copy()
        
        # ARIMA configuration
        self.enable_arima = enable_arima and ARIMA_AVAILABLE
        self.arima_order = arima_order
        self.min_buffer_for_arima = min_buffer_for_arima
        
        # Page-Hinkley test
        self.ph_test = PageHinkleyTest(threshold=ph_threshold, delta=ph_delta)
        
        # Event buffer for retraining
        self.buffer_size = buffer_size
        self.event_buffer = deque(maxlen=buffer_size)
        
        # ARIMA models (one per class)
        self.arima_models = {}
        self.last_forecasts = {name: initial_priors.get(name, 1.0/len(class_names)) 
                              for name in class_names}
        
        # Statistics
        self.drift_count = 0
        self.total_events = 0
        
        if not self.enable_arima:
            print("WARNING: ARIMA not available or disabled. Using simple adaptive priors.")
    
    def _initialize_arima_models(self):
        """Initialize ARIMA models from buffer data."""
        if len(self.event_buffer) < self.min_buffer_for_arima:
            return False
            
        try:
            # Convert event buffer to class counts per time step
            class_counts = []
            for event_class in self.event_buffer:
                counts = {name: 0 for name in self.class_names}
                counts[event_class] = 1
                class_counts.append(counts)
            
            # Train ARIMA model for each class
            for class_name in self.class_names:
                series = [counts[class_name] for counts in class_counts]
                
                # Only train if we have some variation
                if len(set(series)) > 1 and len(series) >= self.min_buffer_for_arima:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model = ARIMA(series, order=self.arima_order)
                            fitted = model.fit(method='hannan_rissanen', low_memory=True)
                            self.arima_models[class_name] = fitted
                    except Exception as e:
                        print(f"Warning: Failed to initialize ARIMA for class {class_name}: {e}")
                        continue
            
            return len(self.arima_models) > 0
            
        except Exception as e:
            print(f"Warning: ARIMA initialization failed: {e}")
            return False
    
    def _predict_next_distribution(self):
        """Predict class distribution for next event using ARIMA models."""
        if not self.enable_arima or not self.arima_models:
            return self.current_priors.copy()
            
        predictions = {}
        
        for class_name in self.class_names:
            if class_name in self.arima_models:
                try:
                    # Forecast one step ahead
                    forecast = self.arima_models[class_name].forecast(steps=1)
                    predictions[class_name] = max(0.001, float(forecast[0]))  # Ensure positive
                except Exception as e:
                    print(f"Warning: ARIMA prediction failed for {class_name}: {e}")
                    predictions[class_name] = self.current_priors.get(class_name, 1.0/len(self.class_names))
            else:
                predictions[class_name] = self.current_priors.get(class_name, 1.0/len(self.class_names))
        
        # Normalize to probabilities
        total = sum(predictions.values())
        if total > 0:
            predictions = {k: v/total for k, v in predictions.items()}
        else:
            predictions = {name: 1.0/len(self.class_names) for name in self.class_names}
            
        return predictions
    
    def _incremental_update_arima(self, observed_class):
        """Incrementally update ARIMA models with new observation."""
        if not self.enable_arima or not self.arima_models:
            return
            
        # For simplicity, we'll do a lightweight update by adjusting recent forecasts
        # In a full implementation, you'd want to use more sophisticated online ARIMA updating
        for class_name in self.class_names:
            if class_name in self.arima_models:
                try:
                    # Simple approach: slightly adjust the last forecast based on observation
                    observed = 1.0 if class_name == observed_class else 0.0
                    current_forecast = self.last_forecasts.get(class_name, 0.5)
                    # Exponential smoothing-style update
                    alpha = 0.1  # Learning rate
                    self.last_forecasts[class_name] = alpha * observed + (1 - alpha) * current_forecast
                except Exception as e:
                    print(f"Warning: Incremental ARIMA update failed for {class_name}: {e}")
    
    def _retrain_arima_models(self):
        """Retrain ARIMA models using current buffer."""
        self.arima_models.clear()
        return self._initialize_arima_models()
    
    def _simple_adaptive_update(self, observed_class, beta=0.3):
        """Simple adaptive update as fallback when ARIMA is not available."""
        # Count recent observations in buffer
        if not self.event_buffer:
            return
            
        recent_counts = {name: 0 for name in self.class_names}
        for event_class in self.event_buffer:
            if event_class in recent_counts:
                recent_counts[event_class] += 1
        
        total_recent = sum(recent_counts.values())
        if total_recent == 0:
            return
            
        # Update priors using exponential smoothing
        for class_name in self.class_names:
            observed_prob = recent_counts[class_name] / total_recent
            old_prior = self.current_priors.get(class_name, 1.0/len(self.class_names))
            self.current_priors[class_name] = beta * observed_prob + (1 - beta) * old_prior
        
        # Normalize
        total_prob = sum(self.current_priors.values())
        if total_prob > 0:
            self.current_priors = {k: v/total_prob for k, v in self.current_priors.items()}
    
    def update(self, observed_class):
        """
        Update adaptive priors with new observed class.
        
        Args:
            observed_class (str): The observed class for the current event
            
        Returns:
            dict: Updated class priors
            bool: True if drift was detected and models were retrained
        """
        self.total_events += 1
        drift_detected = False
        
        # Add to buffer
        self.event_buffer.append(observed_class)
        
        if self.enable_arima:
            # Get prediction for this event (before observing the true class)
            predicted_probs = self._predict_next_distribution()
            
            # Calculate surprise (negative log-likelihood of observed class)
            prob_observed = predicted_probs.get(observed_class, 1e-10)
            residual = -np.log(max(prob_observed, 1e-10))
            
            # Update Page-Hinkley test
            drift_detected = self.ph_test.update(residual)
            
            if drift_detected:
                print(f"Drift detected at event {self.total_events}. Retraining ARIMA models.")
                self.drift_count += 1
                
                # Retrain ARIMA models
                retrain_success = self._retrain_arima_models()
                
                # Reset Page-Hinkley test
                self.ph_test.reset()
                
                if not retrain_success:
                    print("ARIMA retraining failed, falling back to simple update.")
                    self._simple_adaptive_update(observed_class)
            else:
                # Incremental update
                self._incremental_update_arima(observed_class)
                
                # Update current priors based on latest forecasts
                self.current_priors = self._predict_next_distribution()
        else:
            # Use simple adaptive update
            self._simple_adaptive_update(observed_class)
        
        # Initialize ARIMA models if we have enough data and haven't yet
        if (self.enable_arima and not self.arima_models and 
            len(self.event_buffer) >= self.min_buffer_for_arima):
            print(f"Initializing ARIMA models with {len(self.event_buffer)} events.")
            self._initialize_arima_models()
        
        return self.current_priors.copy(), drift_detected
    
    def get_current_priors(self):
        """Get current class priors."""
        return self.current_priors.copy()
    
    def get_statistics(self):
        """Get adaptation statistics."""
        return {
            "total_events": self.total_events,
            "drift_count": self.drift_count,
            "arima_enabled": self.enable_arima,
            "arima_models_count": len(self.arima_models),
            "buffer_size": len(self.event_buffer)
        }


# --- Helper Functions (largely unchanged) ---
def get_quality_metrics(y_true, y_pred, labels, class_names_map=None):
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.array([]), 0.0, {}, {"precision":0.0, "recall":0.0, "f1-score":0.0, "support":0}

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)
    p_r_f1_s = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    
    metrics_per_class = {}
    if class_names_map:
        for i, label_numeric in enumerate(labels):
            class_name_str = class_names_map.get(int(label_numeric), str(label_numeric))
            metrics_per_class[class_name_str] = {
                "precision": float(p_r_f1_s[0][i]), "recall": float(p_r_f1_s[1][i]),
                "f1-score": float(p_r_f1_s[2][i]), "support": int(p_r_f1_s[3][i])
            }
            
    weighted_avg = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
    avg_metrics = {"precision": float(weighted_avg[0]), "recall": float(weighted_avg[1]), "f1-score": float(weighted_avg[2]), "support": int(np.sum(p_r_f1_s[3]))}
    return cm, float(acc), metrics_per_class, avg_metrics


# --- Model Class (largely unchanged) ---
class Model:
    def __init__(self, name, classifier_instance, cost, preprocessor):
        self.name = name
        self.cost = float(cost)
        self.classifier = classifier_instance
        self.preprocessor = preprocessor
        self.trained_pipeline_ = None
        self.cm = None
        self.accuracy = 0.0
        self.metrics_per_class = {}
        self.avg_metrics = {}
        self.exit_classes = set()
        self.label_encoder_classes_ = None
        self.class_names_map = None

    def train(self, X_train, y_train):
        pipeline_steps = [('preprocessor', self.preprocessor if self.preprocessor else StandardScaler()), ('classifier', self.classifier)]
        self.trained_pipeline_ = Pipeline(steps=pipeline_steps)
        self.trained_pipeline_.fit(X_train, y_train)

    def predict(self, X_test):
        if self.trained_pipeline_ is None: raise RuntimeError("Model not trained")
        return self.trained_pipeline_.predict(X_test)

    def predict_proba(self, X_test):
        if self.trained_pipeline_ is None: raise RuntimeError("Model not trained")
        if not hasattr(self.trained_pipeline_.named_steps['classifier'], 'predict_proba'):
            predictions = self.predict(X_test)
            n_classes = len(self.label_encoder_classes_)
            probas = np.zeros((X_test.shape[0], n_classes))
            class_to_idx_map = {label: i for i, label in enumerate(self.label_encoder_classes_)}
            for i, p_encoded in enumerate(predictions):
                if p_encoded in class_to_idx_map:
                    probas[i, class_to_idx_map[p_encoded]] = 1.0
            return probas
        return self.trained_pipeline_.predict_proba(X_test)

    def evaluate(self, X_test, y_test, labels, class_names_map):
        self.label_encoder_classes_ = labels
        self.class_names_map = class_names_map
        if X_test.shape[0] == 0:
            print(f"Warning: Model '{self.name}' evaluation skipped, X_test is empty.")
            return
        y_pred = self.predict(X_test)
        self.cm, self.accuracy, self.metrics_per_class, self.avg_metrics = get_quality_metrics(
            y_test, y_pred, labels=labels, class_names_map=class_names_map
        )

    def get_quality(self, class_name_str, metric_type="f1-score"):
        return float(self.metrics_per_class.get(class_name_str, {}).get(metric_type, 0.0))


# --- Enhanced NOMAD Engine Class ---
class NOMADEngine:
    def __init__(self, models_dict, role_model_name, epsilon,
                 initial_class_priors_str_keys, unique_classes_ordered_str, class_map_numeric_to_str,
                 quality_metric_for_ec="f1-score", safety_check_type="conservative",
                 adaptive_config=None):
        self.models_dict = models_dict
        if role_model_name not in self.models_dict:
            raise ValueError(f"Role model '{role_model_name}' not found in provided models.")
        self.role_model = self.models_dict[role_model_name]
        self.epsilon = float(epsilon)
        self.initial_class_priors_str_keys = initial_class_priors_str_keys
        self.unique_classes_ordered_str = unique_classes_ordered_str
        self.class_map_numeric_to_str = class_map_numeric_to_str
        self.class_map_str_to_numeric = {v: k for k, v in self.class_map_numeric_to_str.items()}

        self.quality_metric_for_ec = quality_metric_for_ec
        self.safety_check_type = safety_check_type

        # Initialize adaptive priors manager
        if adaptive_config is None:
            adaptive_config = {
                'ph_threshold': 50.0,
                'ph_delta': 0.005,
                'buffer_size': 100,
                'min_buffer_for_arima': 20,
                'arima_order': (1, 0, 1),
                'enable_arima': True
            }
        
        self.adaptive_manager = AdaptivePriorsManager(
            class_names=unique_classes_ordered_str,
            initial_priors=initial_class_priors_str_keys,
            **adaptive_config
        )
        
        print(f"DEBUG: NOMADEngine initialized. Role Model: {self.role_model.name}. Epsilon: {self.epsilon}")
        print(f"DEBUG: Initial Priors (str_keys): {self.initial_class_priors_str_keys}")
        print(f"DEBUG: Unique Classes Ordered (str): {self.unique_classes_ordered_str}")
        print(f"DEBUG: Adaptive Manager Config: {adaptive_config}")

        self._determine_all_exit_classes()

    def _determine_all_exit_classes(self):
        print("DEBUG: Determining exit classes for all models...")
        for model_name, model_obj in self.models_dict.items():
            model_obj.exit_classes = set()
            for class_name_str in self.unique_classes_ordered_str:
                quality_model = model_obj.get_quality(class_name_str, self.quality_metric_for_ec)
                quality_role_model = self.role_model.get_quality(class_name_str, self.quality_metric_for_ec)
                if quality_role_model == 0:
                    if quality_model == 0: model_obj.exit_classes.add(class_name_str)
                elif quality_model >= quality_role_model * (1 - self.epsilon):
                    model_obj.exit_classes.add(class_name_str)
            if model_obj == self.role_model:
                 model_obj.exit_classes = set(self.unique_classes_ordered_str)
            print(f"DEBUG: Model '{model_name}' exit classes: {model_obj.exit_classes}")

    def _calculate_utility(self, model, current_class_priors_for_event_str_keys):
        if model.cost == 0:
            sum_prob = sum(current_class_priors_for_event_str_keys.get(cn_str, 0.0) for cn_str in model.exit_classes)
            return float('inf') if sum_prob > 0 else 0.0
        sum_prob_exit_classes = sum(current_class_priors_for_event_str_keys.get(cn_str, 0.0) for cn_str in model.exit_classes)
        utility = float(sum_prob_exit_classes / model.cost) if model.cost > 0 else float('inf')
        return utility

    def _update_probabilities(self, p_in_str_keys, softmax_output):
        p_temp_str_keys = {}
        current_sum = 0.0
        model_proba_class_labels_numeric = self.role_model.label_encoder_classes_

        if model_proba_class_labels_numeric is None or len(model_proba_class_labels_numeric) != len(softmax_output):
            print(f"CRITICAL WARNING: _update_probabilities class order mismatch or missing. Softmax len: {len(softmax_output)}, Expected labels len: {len(model_proba_class_labels_numeric or [])}. Trying fallback with unique_classes_ordered_str.")
            if len(self.unique_classes_ordered_str) == len(softmax_output):
                for i, class_name_str_ordered in enumerate(self.unique_classes_ordered_str):
                    val = p_in_str_keys.get(class_name_str_ordered, 0.0) * softmax_output[i]
                    p_temp_str_keys[class_name_str_ordered] = val
                    current_sum += val
            else:
                print(f"CRITICAL ERROR: Fallback for _update_probabilities also failed. Lengths unique_classes_ordered_str {len(self.unique_classes_ordered_str)} vs softmax {len(softmax_output)}. Probabilities will become uniform.")
        else:
            for i, numeric_label in enumerate(model_proba_class_labels_numeric):
                class_name_str = self.class_map_numeric_to_str.get(int(numeric_label))
                if class_name_str and class_name_str in p_in_str_keys:
                    val = p_in_str_keys[class_name_str] * softmax_output[i]
                    p_temp_str_keys[class_name_str] = val
                    current_sum += val
        
        p_out_str_keys = {}
        if current_sum > 1e-9:
            for class_name_str_key in self.unique_classes_ordered_str:
                p_out_str_keys[class_name_str_key] = float(p_temp_str_keys.get(class_name_str_key, 0.0) / current_sum)
        else:
            print(f"DEBUG _update_probabilities: current_sum of probabilities is ~0 ({current_sum}). Resetting to uniform.")
            num_classes = len(self.unique_classes_ordered_str)
            uniform_prob = 1.0 / num_classes if num_classes > 0 else 0.0
            for class_name_str_key in self.unique_classes_ordered_str:
                p_out_str_keys[class_name_str_key] = uniform_prob
        return p_out_str_keys

    def _check_chain_safety_conservative(self, M_new_name, S_current_names):
        S_potential_names = S_current_names + [M_new_name]
        for class_name_target_str in self.unique_classes_ordered_str:
            M_exit_for_class_j_name = None; M_exit_for_class_j_idx = -1
            S_final_for_this_class_check = S_potential_names
            for idx, model_name_in_chain in enumerate(S_potential_names):
                model_in_chain = self.models_dict[model_name_in_chain]
                if class_name_target_str in model_in_chain.exit_classes:
                    M_exit_for_class_j_name = model_name_in_chain; M_exit_for_class_j_idx = idx; break
            if M_exit_for_class_j_name is None:
                S_final_for_this_class_check = S_potential_names + ([self.role_model.name] if self.role_model.name not in S_potential_names else [])
                M_exit_for_class_j_name = self.role_model.name
                if M_exit_for_class_j_name not in S_final_for_this_class_check:
                    print(f"DEBUG SAFETY ERROR: Role model {self.role_model.name} not in final check chain for class {class_name_target_str}. Chain: {S_final_for_this_class_check}")
                    return False
                M_exit_for_class_j_idx = S_final_for_this_class_check.index(M_exit_for_class_j_name)
            chain_recall_j = 1.0
            for k_idx in range(M_exit_for_class_j_idx + 1):
                model_k_obj = self.models_dict[S_final_for_this_class_check[k_idx]]
                chain_recall_j *= model_k_obj.get_quality(class_name_target_str, "recall")
            recall_role_model_cj = self.role_model.get_quality(class_name_target_str, "recall")
            quality_bound = (1.0 - self.epsilon) * recall_role_model_cj
            if chain_recall_j < quality_bound:
                return False
        return True

    def _get_smoothed_misclassification_prob(self, model_k, class_j_true_str, class_i_pred_str):
        true_label_numeric = self.class_map_str_to_numeric.get(class_j_true_str)
        pred_label_numeric = self.class_map_str_to_numeric.get(class_i_pred_str)

        if true_label_numeric is None or model_k.label_encoder_classes_ is None or \
           true_label_numeric not in model_k.label_encoder_classes_: return 0.0
        try: cm_row_idx = list(model_k.label_encoder_classes_).index(true_label_numeric)
        except ValueError: return 0.0
        
        n_ji = 0
        if pred_label_numeric is not None and pred_label_numeric in model_k.label_encoder_classes_:
            try: 
                cm_col_idx = list(model_k.label_encoder_classes_).index(pred_label_numeric)
                if model_k.cm is not None and cm_row_idx < model_k.cm.shape[0] and cm_col_idx < model_k.cm.shape[1]:
                    n_ji = int(model_k.cm[cm_row_idx, cm_col_idx])
            except ValueError: pass
        
        sum_n_jl = int(np.sum(model_k.cm[cm_row_idx, :])) if model_k.cm is not None and cm_row_idx < model_k.cm.shape[0] else 0
        S_total_pseudo_counts = ALPHA_SMOOTHING * len(self.unique_classes_ordered_str)
        if (sum_n_jl + S_total_pseudo_counts) == 0: return 0.0
        return float(n_ji + ALPHA_SMOOTHING) / (sum_n_jl + S_total_pseudo_counts)

    def _check_chain_safety_relaxed(self, M_new_name, S_current_names):
        S_potential_names = S_current_names + [M_new_name]
        for class_j_true_str in self.unique_classes_ordered_str:
            M_exit_for_class_j_name = None; M_exit_for_class_j_idx = -1
            S_final_for_this_class_check = S_potential_names
            for idx, model_name_in_chain in enumerate(S_potential_names):
                model_in_chain = self.models_dict[model_name_in_chain]
                if class_j_true_str in model_in_chain.exit_classes:
                    M_exit_for_class_j_name = model_name_in_chain; M_exit_for_class_j_idx = idx; break
            if M_exit_for_class_j_name is None:
                S_final_for_this_class_check = S_potential_names + ([self.role_model.name] if self.role_model.name not in S_potential_names else [])
                M_exit_for_class_j_name = self.role_model.name
                if M_exit_for_class_j_name not in S_final_for_this_class_check:
                    print(f"DEBUG SAFETY ERROR (Relaxed): Role model {self.role_model.name} not in final check chain for class {class_j_true_str}. Chain: {S_final_for_this_class_check}")
                    return False
                M_exit_for_class_j_idx = S_final_for_this_class_check.index(M_exit_for_class_j_name)
            
            M_exit_model_obj = self.models_dict[M_exit_for_class_j_name]
            prod_pass_through_refined = 1.0
            for k_idx in range(M_exit_for_class_j_idx):
                model_k_obj = self.models_dict[S_final_for_this_class_check[k_idx]]
                prob_k_misclassify_to_its_exit_class = sum(
                    self._get_smoothed_misclassification_prob(model_k_obj, class_j_true_str, c_i_pred) for c_i_pred in model_k_obj.exit_classes
                )
                prod_pass_through_refined *= (1.0 - prob_k_misclassify_to_its_exit_class)
            
            chain_quality_j_refined = prod_pass_through_refined * M_exit_model_obj.get_quality(class_j_true_str, "recall")
            recall_role_model_cj = self.role_model.get_quality(class_j_true_str, "recall")
            quality_bound = (1.0 - self.epsilon) * recall_role_model_cj
            if chain_quality_j_refined < quality_bound:
                return False
        return True
        
    def select_and_classify_event(self, event_features_df):
        print(f"DEBUG select_and_classify_event: Called. Features shape: {event_features_df.shape}")
        current_class_priors_for_this_event_str_keys = self.adaptive_manager.get_current_priors()
        models_available_dict = self.models_dict.copy()
        realized_chain_names = []
        current_event_cost = 0.0
        final_prediction_name_str = None

        while models_available_dict:
            utility_candidates = []
            if not models_available_dict:
                print("DEBUG select_and_classify_event: No models available in dict.")
                break
            for model_name, model_obj in models_available_dict.items():
                utility_candidates.append(
                    (self._calculate_utility(model_obj, current_class_priors_for_this_event_str_keys),
                     -model_obj.cost, model_name)
                )
            
            if not utility_candidates:
                print("DEBUG select_and_classify_event: No utility candidates generated.")
                break
            
            utility_candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
            selected_model_name_by_utility = utility_candidates[0][2]
            M_star = self.models_dict[selected_model_name_by_utility]
            print(f"DEBUG select_and_classify_event: M_star selected: {M_star.name}, Cost: {M_star.cost}, Utility: {utility_candidates[0][0]:.4f}")
            
            is_safe = False
            if self.safety_check_type == "conservative":
                is_safe = self._check_chain_safety_conservative(M_star.name, realized_chain_names)
            elif self.safety_check_type == "relaxed":
                 is_safe = self._check_chain_safety_relaxed(M_star.name, realized_chain_names)
            print(f"DEBUG select_and_classify_event: M_star '{M_star.name}' safety check: {is_safe}")

            if not is_safe:
                del models_available_dict[M_star.name]
                if M_star == self.role_model:
                    print("DEBUG select_and_classify_event: Role model deemed unsafe, breaking loop.")
                    break 
                continue

            pred_current_encoded = M_star.predict(event_features_df)[0]
            pred_current_name_str = self.class_map_numeric_to_str.get(int(pred_current_encoded), "UnknownPrediction")
            print(f"DEBUG select_and_classify_event: M_star '{M_star.name}' predicted (encoded): {pred_current_encoded}, (name): {pred_current_name_str}")
            
            softmax_vector = M_star.predict_proba(event_features_df)[0]
            current_event_cost += M_star.cost
            realized_chain_names.append(M_star.name)
            final_prediction_name_str = pred_current_name_str

            if pred_current_name_str in M_star.exit_classes:
                print(f"DEBUG select_and_classify_event: M_star '{M_star.name}' EXITED for class '{pred_current_name_str}'.")
                print(f"DEBUG select_and_classify_event: Returning. Prediction: {final_prediction_name_str}, Cost: {current_event_cost}, Chain: {realized_chain_names}")
                return final_prediction_name_str, current_event_cost, realized_chain_names

            del models_available_dict[M_star.name]
            current_class_priors_for_this_event_str_keys = self._update_probabilities(
                current_class_priors_for_this_event_str_keys, softmax_vector
            )
            print(f"DEBUG select_and_classify_event: M_star '{M_star.name}' did not exit. Updated priors for next iteration in chain: {current_class_priors_for_this_event_str_keys}")
            if M_star == self.role_model:
                print("DEBUG select_and_classify_event: Role model processed but didn't exit (should be an exit class). This implies full coverage of role model's exit classes.")
                break

        print(f"DEBUG select_and_classify_event: Loop finished. Realized chain: {realized_chain_names}. Current final_prediction: {final_prediction_name_str}")
        if self.role_model.name not in realized_chain_names:
            print(f"DEBUG select_and_classify_event: Role model not in chain, adding and predicting with it now.")
            current_event_cost += self.role_model.cost
            realized_chain_names.append(self.role_model.name)
        
        pred_final_encoded = self.role_model.predict(event_features_df)[0]
        final_prediction_name_str = self.class_map_numeric_to_str.get(int(pred_final_encoded), "UnknownRolePrediction")
        print(f"DEBUG select_and_classify_event: Fallback/Final prediction by Role Model '{self.role_model.name}': {final_prediction_name_str} (encoded: {pred_final_encoded})")
        
        print(f"DEBUG select_and_classify_event: Returning (after loop/fallback). Prediction: {final_prediction_name_str}, Cost: {current_event_cost}, Chain: {realized_chain_names}")
        return final_prediction_name_str, current_event_cost, realized_chain_names

    def stream_evaluate_strategy(self,
                                 X_test_overall, y_test_encoded_overall,
                                 candidate_model_names_for_counts, X_test_column_names,
                                 batch_size=10, workload_phases=None,
                                 X_by_class_str_keys=None, y_by_class_str_keys=None,**kwargs):

        nomad_predictions_final_encoded_list = []
        true_labels_final_list = []
        total_cost = 0.0
        model_run_counts_live = {name: 0 for name in candidate_model_names_for_counts}
        events_actually_processed_count = 0

        if workload_phases and X_by_class_str_keys and y_by_class_str_keys and len(workload_phases) > 0:
            total_events_defined_by_plan = sum(phase.get('duration', 0) for phase in workload_phases)
            print(f"DEBUG stream_evaluate_strategy: Processing {total_events_defined_by_plan} events based on workload phases.")
        else:
            total_events_defined_by_plan = X_test_overall.shape[0] if X_test_overall is not None else 0
            print(f"DEBUG stream_evaluate_strategy: Processing {total_events_defined_by_plan} events sequentially from test data.")
            if total_events_defined_by_plan == 0 :
                print("DEBUG stream_evaluate_strategy: X_test_overall is empty or None for sequential processing! No events will be processed this way.")

        yield {
            "type": "setup", "total_events": total_events_defined_by_plan,
            "role_model_static_accuracy": float(self.role_model.accuracy),
            "role_model_cost_per_event": float(self.role_model.cost),
            "exit_classes_info": {name: list(model.exit_classes) for name, model in self.models_dict.items()},
            "class_names_ordered": self.unique_classes_ordered_str,
            "initial_priors": {k_str: float(v) for k_str, v in self.adaptive_manager.get_current_priors().items()},
            "adaptive_stats": self.adaptive_manager.get_statistics()
        }

        previous_phase_target_dist_probs = np.array([
            self.initial_class_priors_str_keys.get(c, 0.0) for c in self.unique_classes_ordered_str
        ])
        previous_phase_target_dist_probs = np.maximum(previous_phase_target_dist_probs, 1e-9)
        if np.sum(previous_phase_target_dist_probs) > 0:
            previous_phase_target_dist_probs /= np.sum(previous_phase_target_dist_probs)
        else:
            print("Warning: Sum of initial_class_priors is zero. Setting previous_phase_target_dist_probs to uniform for KL calculation.")
            num_cls = len(self.unique_classes_ordered_str)
            previous_phase_target_dist_probs = np.full(num_cls, 1.0/num_cls if num_cls > 0 else 0.0)

        current_event_index_in_plan = 0

        if workload_phases and X_by_class_str_keys and y_by_class_str_keys and len(workload_phases) > 0:
            random.seed(RANDOM_SEED_FOR_SAMPLING); np.random.seed(RANDOM_SEED_FOR_SAMPLING)
            print("DEBUG stream_evaluate_strategy: Starting Phased Workload Execution.")
            for phase_idx, phase_def in enumerate(workload_phases):
                duration = phase_def.get('duration', 0)
                target_dist_str_keys = phase_def.get('target_distribution', {})
                print(f"DEBUG stream_evaluate_strategy: Phase {phase_idx}, Duration: {duration}, TargetDist: {target_dist_str_keys}")

                current_phase_target_dist_probs = np.array([target_dist_str_keys.get(c_str, 0.0) for c_str in self.unique_classes_ordered_str])
                if not np.isclose(np.sum(current_phase_target_dist_probs), 1.0) or np.any(current_phase_target_dist_probs < 0):
                    print(f"Warning: Invalid target distribution for phase {phase_idx}. Sum: {np.sum(current_phase_target_dist_probs)}. Normalizing or using uniform.")
                    if np.sum(current_phase_target_dist_probs) > 1e-9 : current_phase_target_dist_probs /= np.sum(current_phase_target_dist_probs)
                    else: current_phase_target_dist_probs = np.ones(len(self.unique_classes_ordered_str)) / max(1, len(self.unique_classes_ordered_str))
                    target_dist_str_keys = {self.unique_classes_ordered_str[i]: current_phase_target_dist_probs[i] for i in range(len(self.unique_classes_ordered_str))}
                
                q_probs_for_kl = np.maximum(current_phase_target_dist_probs, 1e-9); q_probs_for_kl /= np.sum(q_probs_for_kl)
                kl_divergence = entropy(pk=q_probs_for_kl, qk=previous_phase_target_dist_probs) if len(q_probs_for_kl) == len(previous_phase_target_dist_probs) else -1.0
                
                yield {"type": "phase_shift", "phase_index": phase_idx, "duration": duration,
                       "target_distribution": target_dist_str_keys,
                       "kl_divergence_from_previous": float(kl_divergence if np.isfinite(kl_divergence) else -1.0),
                       "previous_target_distribution": {self.unique_classes_ordered_str[i]: previous_phase_target_dist_probs[i] for i in range(len(previous_phase_target_dist_probs))}}
                previous_phase_target_dist_probs = current_phase_target_dist_probs.copy()

                phase_X_list, phase_y_list = [], []
                num_samples_per_class_in_phase = { c_str: int(round(duration * target_dist_str_keys.get(c_str, 0.0))) for c_str in self.unique_classes_ordered_str }
                current_total_samples_calc = sum(num_samples_per_class_in_phase.values())
                if current_total_samples_calc != duration and duration > 0:
                    diff = duration - current_total_samples_calc
                    if diff != 0:
                        adjust_class_candidates = [k for k,v in target_dist_str_keys.items() if v > 0]
                        if not adjust_class_candidates and self.unique_classes_ordered_str: adjust_class_candidates = [self.unique_classes_ordered_str[0]]
                        if adjust_class_candidates:
                            adjust_class = random.choice(adjust_class_candidates)
                            num_samples_per_class_in_phase[adjust_class] += diff
                            num_samples_per_class_in_phase[adjust_class] = max(0, num_samples_per_class_in_phase[adjust_class])
                print(f"DEBUG stream_evaluate_strategy: Phase {phase_idx} num_samples_per_class: {num_samples_per_class_in_phase}")

                for class_name_str, num_samples in num_samples_per_class_in_phase.items():
                    if num_samples == 0: continue
                    available_X_df = X_by_class_str_keys.get(class_name_str)
                    available_y_arr = y_by_class_str_keys.get(class_name_str)
                    if available_X_df is None or available_X_df.empty or available_y_arr is None or len(available_y_arr) == 0 :
                        print(f"Warning: No samples available or empty for class '{class_name_str}' for phase {phase_idx}.")
                        continue
                    indices_to_sample = random.choices(range(available_X_df.shape[0]), k=num_samples)
                    phase_X_list.extend([available_X_df.iloc[k] for k in indices_to_sample])
                    phase_y_list.extend([available_y_arr[k] for k in indices_to_sample])

                if not phase_X_list:
                    print(f"Warning: No data could be sampled for phase {phase_idx}. Duration {duration}.")
                    current_event_index_in_plan += duration
                    continue

                combined_phase_data = list(zip(phase_X_list, phase_y_list)); random.shuffle(combined_phase_data)
                phase_X_sampled_df = pd.DataFrame([item[0] for item in combined_phase_data], columns=X_test_column_names)
                phase_y_sampled_encoded = np.array([item[1] for item in combined_phase_data])
                print(f"DEBUG stream_evaluate_strategy: Phase {phase_idx} sampled {phase_X_sampled_df.shape[0]} events.")
                
                for i_phase in range(phase_X_sampled_df.shape[0]):
                    current_event_index_in_plan += 1
                    event_features_df = phase_X_sampled_df.iloc[i_phase].to_frame().T
                    true_label_encoded_this_event = phase_y_sampled_encoded[i_phase]
                    
                    pred_name_str, cost_for_event, chain_for_event = self.select_and_classify_event(event_features_df)
                    print(f"DEBUG stream_evaluate_strategy (Phased Event {current_event_index_in_plan}): Pred='{pred_name_str}', Cost={cost_for_event}, TrueLabel={true_label_encoded_this_event}")

                    pred_encoded = self.class_map_str_to_numeric.get(pred_name_str, -1)
                    if pred_encoded == -1 and self.role_model.label_encoder_classes_ is not None and len(self.role_model.label_encoder_classes_) > 0:
                        pred_encoded = int(self.role_model.label_encoder_classes_[0])

                    nomad_predictions_final_encoded_list.append(pred_encoded)
                    true_labels_final_list.append(true_label_encoded_this_event)
                    events_actually_processed_count += 1
                    total_cost += cost_for_event
                    for model_name_in_chain in chain_for_event:
                        if model_name_in_chain in model_run_counts_live: model_run_counts_live[model_name_in_chain] += 1
                    
                    # Update adaptive priors with true class
                    true_class_str = self.class_map_numeric_to_str.get(int(true_label_encoded_this_event))
                    if true_class_str:
                        updated_priors, drift_detected = self.adaptive_manager.update(true_class_str)
                        if drift_detected:
                            yield {
                                "type": "drift_detected",
                                "event_number": current_event_index_in_plan,
                                "updated_priors": {k: float(v) for k, v in updated_priors.items()},
                                "adaptive_stats": self.adaptive_manager.get_statistics()
                            }

                    if (current_event_index_in_plan % batch_size == 0) or \
                       (current_event_index_in_plan == total_events_defined_by_plan):
                        live_nomad_accuracy = accuracy_score(true_labels_final_list, nomad_predictions_final_encoded_list) if true_labels_final_list else 0.0
                        yield { "type": "batch_update", "last_event_in_batch": current_event_index_in_plan,
                                "cumulative_cost": float(total_cost), "live_nomad_accuracy": live_nomad_accuracy,
                                "model_run_counts_snapshot": {k_str: int(v) for k_str,v in model_run_counts_live.items()},
                                "current_priors": {k: float(v) for k, v in self.adaptive_manager.get_current_priors().items()},
                                "adaptive_stats": self.adaptive_manager.get_statistics()}
        else:
            print(f"DEBUG stream_evaluate_strategy: Starting Sequential Workload Execution. X_test_overall shape: {X_test_overall.shape if X_test_overall is not None else 'None'}")
            if X_test_overall is None or X_test_overall.empty:
                print("DEBUG stream_evaluate_strategy: X_test_overall is None or empty in sequential mode branch. No events will be processed.")
            else:
                for i in range(X_test_overall.shape[0]):
                    current_event_index_in_plan += 1
                    print(f"DEBUG stream_evaluate_strategy (Sequential Event {current_event_index_in_plan}, index {i})")
                    
                    if isinstance(X_test_overall, pd.DataFrame): event_features_df = X_test_overall.iloc[[i]]
                    else: event_features_df = pd.DataFrame([X_test_overall[i,:]], columns=X_test_column_names)
                    true_label_encoded_this_event = y_test_encoded_overall[i]

                    pred_name_str, cost_for_event, chain_for_event = self.select_and_classify_event(event_features_df)
                    print(f"DEBUG stream_evaluate_strategy (Sequential Event {current_event_index_in_plan}): Pred='{pred_name_str}', Cost={cost_for_event}, TrueLabel={true_label_encoded_this_event}")
                    
                    pred_encoded = self.class_map_str_to_numeric.get(pred_name_str, -1)
                    if pred_encoded == -1 and self.role_model.label_encoder_classes_ is not None and len(self.role_model.label_encoder_classes_) > 0:
                        pred_encoded = int(self.role_model.label_encoder_classes_[0])

                    nomad_predictions_final_encoded_list.append(pred_encoded)
                    true_labels_final_list.append(true_label_encoded_this_event)
                    events_actually_processed_count += 1
                    total_cost += cost_for_event
                    for model_name_in_chain in chain_for_event:
                        if model_name_in_chain in model_run_counts_live: model_run_counts_live[model_name_in_chain] += 1
                    
                    # Update adaptive priors with true class
                    true_class_str = self.class_map_numeric_to_str.get(int(true_label_encoded_this_event))
                    if true_class_str:
                        updated_priors, drift_detected = self.adaptive_manager.update(true_class_str)
                        if drift_detected:
                            yield {
                                "type": "drift_detected",
                                "event_number": current_event_index_in_plan,
                                "updated_priors": {k: float(v) for k, v in updated_priors.items()},
                                "adaptive_stats": self.adaptive_manager.get_statistics()
                            }

                    live_nomad_accuracy = accuracy_score(true_labels_final_list, nomad_predictions_final_encoded_list) if true_labels_final_list else 0.0
                    print(f"DEBUG BATCH YIELD (Event: {current_event_index_in_plan}): total_cost={total_cost}, live_nomad_accuracy={live_nomad_accuracy}, len(true_labels)={len(true_labels_final_list)}, len(nomad_preds)={len(nomad_predictions_final_encoded_list)}")
                    yield { "type": "batch_update", "last_event_in_batch": current_event_index_in_plan,
                            "cumulative_cost": float(total_cost), "live_nomad_accuracy": live_nomad_accuracy,
                            "model_run_counts_snapshot": {k_str: int(v) for k_str,v in model_run_counts_live.items()},
                            "current_priors": {k: float(v) for k, v in self.adaptive_manager.get_current_priors().items()},
                            "adaptive_stats": self.adaptive_manager.get_statistics()}
        
        print(f"DEBUG stream_evaluate_strategy: Finished processing loops. Total events actually processed: {events_actually_processed_count}")
        print(f"DEBUG stream_evaluate_strategy: Final nomad_predictions length: {len(nomad_predictions_final_encoded_list)}, true_labels length: {len(true_labels_final_list)}")
        print(f"DEBUG stream_evaluate_strategy: Final total_cost: {total_cost}")

        avg_cost = float(total_cost / events_actually_processed_count) if events_actually_processed_count > 0 else 0.0
        
        final_cm_nomad, final_acc_nomad, final_metrics_per_class_nomad, final_avg_metrics_nomad = np.array([]), 0.0, {}, {"precision":0.0, "recall":0.0, "f1-score":0.0, "support":0}

        if events_actually_processed_count == 0:
            print("DEBUG stream_evaluate_strategy: No events were actually processed. Reporting zero metrics for NOMAD.")
        else:
            eval_labels_numeric = self.role_model.label_encoder_classes_
            if eval_labels_numeric is None:
                print("CRITICAL WARNING: Role model label_encoder_classes_ is None. Attempting to use unique_classes_ordered for metrics.")
                eval_labels_numeric = [self.class_map_str_to_numeric.get(s) for s in self.unique_classes_ordered_str]
                eval_labels_numeric = [l for l in eval_labels_numeric if l is not None]
                if not eval_labels_numeric:
                    print("CRITICAL ERROR: Cannot determine numeric labels for final evaluation. Metrics might be incorrect or fail.")
                    eval_labels_numeric = np.unique(true_labels_final_list + nomad_predictions_final_encoded_list).tolist()

            print(f"DEBUG stream_evaluate_strategy: Inputs to get_quality_metrics for NOMAD:")
            print(f"  y_true length: {len(true_labels_final_list)}, sample: {true_labels_final_list[:5]}")
            print(f"  y_pred length: {len(nomad_predictions_final_encoded_list)}, sample: {nomad_predictions_final_encoded_list[:5]}")
            print(f"  labels (numeric): {eval_labels_numeric}")
            print(f"  class_names_map (num->str): {self.class_map_numeric_to_str}")

            try:
                final_cm_nomad, final_acc_nomad, final_metrics_per_class_nomad, final_avg_metrics_nomad = get_quality_metrics(
                    true_labels_final_list, nomad_predictions_final_encoded_list,
                    labels=list(eval_labels_numeric),
                    class_names_map=self.class_map_numeric_to_str
                )
                print(f"DEBUG stream_evaluate_strategy: NOMAD final accuracy from get_quality_metrics: {final_acc_nomad}")
            except Exception as e_metrics:
                print(f"ERROR stream_evaluate_strategy: Failed during final get_quality_metrics for NOMAD: {e_metrics}")
                print(traceback.format_exc())

        yield {
            "type": "summary",
            "nomad_overall_metrics": { "accuracy": float(final_acc_nomad), "avg_metrics": final_avg_metrics_nomad,
                                       "average_cost": avg_cost, "cm": final_cm_nomad.tolist() if final_cm_nomad.size > 0 else [], },
            "nomad_per_class_metrics": final_metrics_per_class_nomad,
            "role_model_performance": { "name": self.role_model.name, "accuracy": float(self.role_model.accuracy),
                                        "avg_metrics": self.role_model.avg_metrics, "cost": float(self.role_model.cost),
                                        "cm": self.role_model.cm.tolist() if self.role_model.cm is not None and self.role_model.cm.size > 0 else [],
                                        "per_class_metrics": self.role_model.metrics_per_class },
            "final_model_run_counts": {k_str: int(v) for k_str, v in model_run_counts_live.items()},
            "final_adaptive_stats": self.adaptive_manager.get_statistics(),
            "final_priors": {k: float(v) for k, v in self.adaptive_manager.get_current_priors().items()}
        }

# --- load_and_preprocess_data_for_flask (unchanged) ---
def load_and_preprocess_data_for_flask(csv_file_path):
    print(f"DEBUG load_and_preprocess: Loading data from {csv_file_path}")
    try: data = pd.read_csv(csv_file_path)
    except Exception as e: return None, None, None, None, None, None, None, None, None, None, None, f"Error loading CSV: {str(e)}"

    if data.empty: return None, None, None, None, None, None, None, None, None, None, None, "CSV file is empty."
    if data.shape[1] < 2: return None, None, None, None, None, None, None, None, None, None, None, "CSV must have at least two columns."

    X_df = data.iloc[:, :-2]
    y_raw_series = data.iloc[:, -2]
    print(f"DEBUG load_and_preprocess: X_df shape {X_df.shape}, y_raw_series length {len(y_raw_series)}")

    if X_df.empty and data.shape[1] >= 1:
        print("DEBUG load_and_preprocess: X_df is empty (likely single column CSV). Creating empty DataFrame for X.")
        X_df = pd.DataFrame(index=data.index)

    label_encoder = LabelEncoder()
    y_encoded_array = label_encoder.fit_transform(y_raw_series)
    unique_labels_encoded = np.sort(np.unique(y_encoded_array))
    
    class_map_num_to_str = {int(enc): str(name) for enc, name in zip(unique_labels_encoded, label_encoder.inverse_transform(unique_labels_encoded))}
    unique_class_names_ordered_str = [class_map_num_to_str[int(el)] for el in unique_labels_encoded]
    print(f"DEBUG load_and_preprocess: Detected classes (str): {unique_class_names_ordered_str}")
    print(f"DEBUG load_and_preprocess: Class map num->str: {class_map_num_to_str}")

    X_by_class_str_keys = {}
    y_by_class_str_keys = {}
    for class_name_str in unique_class_names_ordered_str:
        numeric_label = -1
        for k_num, v_str in class_map_num_to_str.items():
            if v_str == class_name_str: numeric_label = k_num; break
        
        mask = (y_encoded_array == numeric_label)
        X_by_class_str_keys[class_name_str] = X_df[mask].copy()
        y_by_class_str_keys[class_name_str] = y_encoded_array[mask].copy()
        print(f"DEBUG load_and_preprocess: Segregated data for class '{class_name_str}': X shape {X_by_class_str_keys[class_name_str].shape}, y length {len(y_by_class_str_keys[class_name_str])}")

    numerical_cols = X_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"DEBUG load_and_preprocess: Numerical cols: {numerical_cols}, Categorical cols: {categorical_cols}")
    
    transformers_list = []
    if numerical_cols: transformers_list.append(('num', StandardScaler(), numerical_cols))
    if categorical_cols: transformers_list.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols))
    
    if not transformers_list and not X_df.empty: 
        print("Warning load_and_preprocess: No numerical or categorical columns identified for dedicated preprocessing.")

    preprocessor = ColumnTransformer(transformers=transformers_list, remainder='passthrough')
    
    return data, X_df, y_encoded_array, label_encoder, unique_labels_encoded, \
           class_map_num_to_str, unique_class_names_ordered_str, preprocessor, X_df.columns.tolist(), \
           X_by_class_str_keys, y_by_class_str_keys, None


def train_evaluate_individual_models_for_flask(X_full, y_encoded_full, preprocessor, class_map_numeric_to_str, unique_enc_labels, candidate_configs_dict, custom_models_path):
    """
    Trains and evaluates models based on a dynamic configuration dictionary.
    """
    if len(y_encoded_full) == 0:
        raise ValueError("Cannot train models with an empty target variable.")
    
    # Train/Test split
    can_stratify = len(np.unique(y_encoded_full)) > 1 and np.min(np.unique(y_encoded_full, return_counts=True)[1]) >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_encoded_full, test_size=0.3, random_state=42,
        stratify=y_encoded_full if can_stratify else None
    )

    trained_models_dict = {}
    model_summaries = []

    for name, config in candidate_configs_dict.items():
        print(f"Processing model '{name}'...")
        try:
            # Instantiate classifier using the new factory function
            clf_instance = get_classifier_from_config(config, custom_models_path)
            model = Model(name, clf_instance, float(config['cost']), preprocessor=preprocessor)
            
            if X_train.shape[0] > 0:
                model.train(X_train, y_train)
            
            if X_test.shape[0] > 0:
                model.evaluate(X_test, y_test, labels=unique_enc_labels, class_names_map=class_map_numeric_to_str)
            
            trained_models_dict[name] = model
            model_summaries.append({
                "name": name, "accuracy": model.accuracy,
                "f1_score_weighted": model.avg_metrics.get('f1-score', 0.0),
                "cost": model.cost
            })
        except Exception as e:
            print(f"ERROR: Failed to process model '{name}': {e}")
            print(traceback.format_exc())
            model_summaries.append({ "name": f"{name} (FAILED)", "accuracy": 0, "f1_score_weighted": 0, "cost": config['cost'] })

    return trained_models_dict, model_summaries, X_test, y_test, y_train


def run_nomad_simulation_for_flask_streamed(trained_models_dict, role_model_name, epsilon,
                                           initial_class_priors_str_keys, unique_class_names_ordered_str,
                                           class_map_numeric_to_str, quality_metric_for_ec, safety_check_type,
                                           X_test_data_overall, y_test_encoded_data_overall,
                                           candidate_configs, X_test_column_names, **kwargs):
    """
    Initializes and runs the NOMAD engine, yielding simulation updates.
    Supports both independent and dependent models.
    """
    # Extract adaptive configuration from kwargs
    adaptive_config = kwargs.get('adaptive_config', {
        'ph_threshold': 50.0,
        'ph_delta': 0.005,
        'buffer_size': 100,
        'min_buffer_for_arima': 20,
        'arima_order': (1, 0, 1),
        'enable_arima': True
    })
    
    # Check if any model has dependencies to enable dependent model mode
    enable_dependent_models = kwargs.get('enable_dependent_models', False)
    if not enable_dependent_models:
        # Auto-detect if any models have dependencies
        for model_obj in trained_models_dict.values():
            if hasattr(model_obj, 'prerequisites') and model_obj.prerequisites:
                enable_dependent_models = True
                print(f"DEBUG: Auto-detected dependent models. Model '{model_obj.name}' has prerequisites: {model_obj.prerequisites}")
                break
    
    nomad_engine = NOMADEngine(
        models_dict=trained_models_dict, 
        role_model_name=role_model_name, 
        epsilon=epsilon,
        initial_class_priors_str_keys=initial_class_priors_str_keys,
        unique_classes_ordered_str=unique_class_names_ordered_str,
        class_map_numeric_to_str=class_map_numeric_to_str,
        quality_metric_for_ec=quality_metric_for_ec,
        safety_check_type=safety_check_type,
        adaptive_config=adaptive_config,
        enable_dependent_models=enable_dependent_models
    )
    
    candidate_model_names = list(candidate_configs.keys())

    # Pass all keyword arguments to the strategy evaluator
    yield from nomad_engine.stream_evaluate_strategy(
        X_test_overall=X_test_data_overall,
        y_test_encoded_overall=y_test_encoded_data_overall,
        candidate_model_names_for_counts=candidate_model_names,
        X_test_column_names=X_test_column_names,
        **kwargs
    )
