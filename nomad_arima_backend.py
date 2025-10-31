import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, recall_score
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
ALPHA_SMOOTHING = 1  # Laplace smoothing parameter
RANDOM_SEED = 42
EPSILON_THRESHOLD = 1e-9  # For numerical stability

# --- Model Instantiation ---
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
    """Instantiates a classifier from its configuration dictionary."""
    model_type = model_config.get("type")
    params = model_config.get("params", {})
    
    if model_config.get("is_custom"):
        module_name = model_config.get("module_name")
        class_name = model_config.get("class_name")
        if not module_name or not class_name:
            raise ValueError(f"Custom model '{model_config['name']}' is missing module or class name.")
        
        filepath = os.path.join(custom_models_path, f"{module_name}.py")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Custom model file not found at {filepath}")
        
        try:
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            custom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_module)
            ClassifierClass = getattr(custom_module, class_name)
        except Exception as e:
            raise ImportError(f"Failed to load custom model: {e}")
    else:
        if model_type not in CLASSIFIER_MAP:
            raise ValueError(f"Unknown model type: '{model_type}'")
        ClassifierClass = CLASSIFIER_MAP[model_type]
    
    valid_params = {k: v for k, v in params.items() if k in ClassifierClass().get_params()}
    return ClassifierClass(**valid_params)


# --- Page-Hinkley Test Implementation (Section 6 and Appendix D.2) ---
class PageHinkleyTest:
    """
    Implementation of the Page-Hinkley test for drift detection.
    Mathematical formulation from Appendix D.2 of the paper.
    """
    def __init__(self, threshold_lambda=50.0, delta=0.005):
        """
        Initialize Page-Hinkley test.
        
        Args:
            threshold_lambda (float): Detection threshold λ
            delta (float): Tolerance parameter δ for drift detection
        """
        self.threshold_lambda = threshold_lambda
        self.delta = delta
        self.reset()
    
    def reset(self):
        """Reset the test to initial state."""
        self.m_t = 0.0  # Cumulative sum m_T
        self.M_t = 0.0  # Minimum value M_T
        self.sum_x = 0.0
        self.n = 0
        
    def update(self, x):
        """
        Update the test with a new residual value and check for drift.
        Implements the PH test from Appendix D.2.
        
        Args:
            x (float): New residual value (negative log-likelihood)
            
        Returns:
            bool: True if drift is detected (m_T - M_T > λ)
        """
        self.n += 1
        self.sum_x += x
        
        # Calculate running average x̄_T
        avg = self.sum_x / self.n if self.n > 0 else 0
        
        # Update cumulative sum: m_T = Σ(x_t - x̄_t - δ)
        self.m_t += (x - avg - self.delta)
        
        # Update minimum: M_T = min(m_t, t = 1...T)
        self.M_t = min(self.M_t, self.m_t)
        
        # Check for drift: m_T - M_T > λ
        drift_detected = (self.m_t - self.M_t) > self.threshold_lambda
        
        return drift_detected


# --- Adaptive Priors Manager (Algorithm 5) ---
class AdaptivePriorsManager:
    """
    Implements Algorithm 5: Event-by-Event Adaptive Priors with ARIMA and Page-Hinkley Test
    """
    def __init__(self, class_names, initial_priors, 
                 ph_threshold=50.0, ph_delta=0.005, 
                 buffer_size=1000, min_buffer_for_arima=20,
                 arima_order=(1, 0, 1), enable_arima=True):
        """
        Initialize adaptive priors manager as per Algorithm 5.
        """
        self.class_names = class_names
        self.current_priors = initial_priors.copy()
        self.initial_priors = initial_priors.copy()
        
        # ARIMA configuration
        self.enable_arima = enable_arima and ARIMA_AVAILABLE
        self.arima_order = arima_order
        self.min_buffer_for_arima = min_buffer_for_arima
        
        # Page-Hinkley test (Line 3 in Algorithm 5)
        self.ph_test = PageHinkleyTest(threshold_lambda=ph_threshold, delta=ph_delta)
        
        # Event buffer (Line 4 in Algorithm 5)
        self.buffer_size = buffer_size
        self.event_buffer = deque(maxlen=buffer_size)
        
        # ARIMA models (Line 2 in Algorithm 5)
        self.arima_models = {}
        self.last_forecasts = {name: initial_priors.get(name, 1.0/len(class_names)) 
                              for name in class_names}
        
        # Statistics
        self.drift_count = 0
        self.total_events = 0
        
        if not self.enable_arima:
            print("WARNING: ARIMA not available. Using simple adaptive priors.")
    
    def _initialize_arima_models(self):
        """Initialize ARIMA models with historical data (Line 2 in Algorithm 5)."""
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
                
                if len(set(series)) > 1 and len(series) >= self.min_buffer_for_arima:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model = ARIMA(series, order=self.arima_order)
                            fitted = model.fit(method='hannan_rissanen', low_memory=True)
                            self.arima_models[class_name] = fitted
                    except Exception as e:
                        continue
            
            return len(self.arima_models) > 0
            
        except Exception as e:
            return False
    
    def _predict_next_distribution(self):
        """Predict distribution for next event using ARIMA (Line 6 in Algorithm 5)."""
        if not self.enable_arima or not self.arima_models:
            return self.current_priors.copy()
            
        predictions = {}
        
        for class_name in self.class_names:
            if class_name in self.arima_models:
                try:
                    # Forecast one step ahead
                    forecast = self.arima_models[class_name].forecast(steps=1)
                    predictions[class_name] = max(0.001, float(forecast[0]))
                except Exception:
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
    
    def _retrain_arima_models(self):
        """Retrain ARIMA models using events in buffer (Line 12 in Algorithm 5)."""
        self.arima_models.clear()
        return self._initialize_arima_models()
    
    def _incremental_update(self, observed_class):
        """Incrementally update ARIMA with observed class (Line 14 in Algorithm 5)."""
        if not self.enable_arima or not self.arima_models:
            return
            
        for class_name in self.class_names:
            if class_name in self.arima_models:
                try:
                    observed = 1.0 if class_name == observed_class else 0.0
                    current_forecast = self.last_forecasts.get(class_name, 0.5)
                    # Simple exponential smoothing update
                    alpha = 0.1
                    self.last_forecasts[class_name] = alpha * observed + (1 - alpha) * current_forecast
                except Exception:
                    pass
    
    def update(self, observed_class):
        """
        Update adaptive priors with new observed class (Algorithm 5, Lines 5-15).
        
        Args:
            observed_class (str): The observed class for the current event
            
        Returns:
            dict: Updated class priors
            bool: True if drift was detected
        """
        self.total_events += 1
        drift_detected = False
        
        # Add to buffer (Line 8 in Algorithm 5)
        self.event_buffer.append(observed_class)
        
        if self.enable_arima:
            # Get prediction for this event (Line 6)
            predicted_probs = self._predict_next_distribution()
            
            # Calculate residual (negative log-likelihood) (Line 9)
            prob_observed = predicted_probs.get(observed_class, 1e-10)
            residual = -np.log(max(prob_observed, 1e-10))
            
            # Update Page-Hinkley test (Line 10)
            drift_detected = self.ph_test.update(residual)
            
            if drift_detected:
                # Lines 11-13 in Algorithm 5
                print(f"Drift detected at event {self.total_events}. Retraining ARIMA models.")
                self.drift_count += 1
                
                # Retrain ARIMA models (Line 12)
                self._retrain_arima_models()
                
                # Reset PH detector (Line 13)
                self.ph_test.reset()
            
            # Incremental update (Line 14)
            self._incremental_update(observed_class)
            
            # Get forecast from updated models (Line 15)
            self.current_priors = self._predict_next_distribution()
        else:
            # Simple fallback if ARIMA not available
            self._simple_adaptive_update(observed_class)
        
        return self.current_priors.copy(), drift_detected
    
    def _simple_adaptive_update(self, observed_class, beta=0.3):
        """Simple adaptive update as fallback when ARIMA is not available."""
        if not self.event_buffer:
            return
            
        recent_counts = {name: 0 for name in self.class_names}
        for event_class in self.event_buffer:
            if event_class in recent_counts:
                recent_counts[event_class] += 1
        
        total_recent = sum(recent_counts.values())
        if total_recent == 0:
            return
            
        for class_name in self.class_names:
            observed_prob = recent_counts[class_name] / total_recent
            old_prior = self.current_priors.get(class_name, 1.0/len(self.class_names))
            self.current_priors[class_name] = beta * observed_prob + (1 - beta) * old_prior
        
        # Normalize
        total_prob = sum(self.current_priors.values())
        if total_prob > 0:
            self.current_priors = {k: v/total_prob for k, v in self.current_priors.items()}
    
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


# --- Model Class ---
class Model:
    def __init__(self, name, classifier_instance, cost, preprocessor):
        self.name = name
        self.cost = float(cost)
        self.classifier = classifier_instance
        self.preprocessor = preprocessor
        self.trained_pipeline_ = None
        self.cm = None  # Confusion matrix
        self.accuracy = 0.0
        self.metrics_per_class = {}
        self.avg_metrics = {}
        self.exit_classes = set()
        self.label_encoder_classes_ = None
        self.class_names_map = None

    def train(self, X_train, y_train):
        """Train the model pipeline."""
        pipeline_steps = [
            ('preprocessor', self.preprocessor if self.preprocessor else StandardScaler()),
            ('classifier', self.classifier)
        ]
        self.trained_pipeline_ = Pipeline(steps=pipeline_steps)
        self.trained_pipeline_.fit(X_train, y_train)

    def predict(self, X_test):
        """Predict classes for test data."""
        if self.trained_pipeline_ is None:
            raise RuntimeError("Model not trained")
        return self.trained_pipeline_.predict(X_test)

    def predict_proba(self, X_test):
        """Predict class probabilities (softmax output)."""
        if self.trained_pipeline_ is None:
            raise RuntimeError("Model not trained")
        if not hasattr(self.trained_pipeline_.named_steps['classifier'], 'predict_proba'):
            # Fallback for classifiers without predict_proba
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
        """Evaluate model and compute confusion matrix and quality metrics."""
        self.label_encoder_classes_ = labels
        self.class_names_map = class_names_map
        if X_test.shape[0] == 0:
            return
        y_pred = self.predict(X_test)
        self.cm, self.accuracy, self.metrics_per_class, self.avg_metrics = get_quality_metrics(
            y_test, y_pred, labels=labels, class_names_map=class_names_map
        )

    def get_quality(self, class_name_str, metric_type="recall"):
        """Get quality Q(M, C_j) for a specific class (Definition 2.1)."""
        return float(self.metrics_per_class.get(class_name_str, {}).get(metric_type, 0.0))


def get_quality_metrics(y_true, y_pred, labels, class_names_map=None):
    """Compute confusion matrix and quality metrics."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.array([]), 0.0, {}, {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0}

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)
    p_r_f1_s = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    
    metrics_per_class = {}
    if class_names_map:
        for i, label_numeric in enumerate(labels):
            class_name_str = class_names_map.get(int(label_numeric), str(label_numeric))
            metrics_per_class[class_name_str] = {
                "precision": float(p_r_f1_s[0][i]),
                "recall": float(p_r_f1_s[1][i]),
                "f1-score": float(p_r_f1_s[2][i]),
                "support": int(p_r_f1_s[3][i])
            }
            
    weighted_avg = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
    avg_metrics = {
        "precision": float(weighted_avg[0]),
        "recall": float(weighted_avg[1]),
        "f1-score": float(weighted_avg[2]),
        "support": int(np.sum(p_r_f1_s[3]))
    }
    return cm, float(acc), metrics_per_class, avg_metrics


# --- NOMAD Engine Class (Implements Algorithms 1-4) ---
class NOMADEngine:
    def __init__(self, models_dict, role_model_name, epsilon,
                 initial_class_priors_str_keys, unique_classes_ordered_str, class_map_numeric_to_str,
                 quality_metric="recall", safety_mode="global", passthrough_mode="relaxed",
                 adaptive_config=None):
        """
        Initialize NOMAD Engine.
        
        Args:
            models_dict: Dictionary of trained models
            role_model_name: Name of the role model M_r (Definition 2.1)
            epsilon: Quality tolerance factor ε (Definition 2.2)
            initial_class_priors_str_keys: Initial class probabilities Prob(C_j)
            unique_classes_ordered_str: Ordered list of class names
            class_map_numeric_to_str: Mapping from numeric to string class labels
            quality_metric: Quality metric to use (default: "recall")
            safety_mode: "global" or "class-based" (Section 4 and Appendix B)
            passthrough_mode: "relaxed" or "conservative" (Section 4.1 and Appendix A)
            adaptive_config: Configuration for adaptive priors manager
        """
        self.models_dict = models_dict
        if role_model_name not in self.models_dict:
            raise ValueError(f"Role model '{role_model_name}' not found in provided models.")
        self.role_model = self.models_dict[role_model_name]
        self.epsilon = float(epsilon)
        self.initial_class_priors_str_keys = initial_class_priors_str_keys
        self.unique_classes_ordered_str = unique_classes_ordered_str
        self.class_map_numeric_to_str = class_map_numeric_to_str
        self.class_map_str_to_numeric = {v: k for k, v in self.class_map_numeric_to_str.items()}

        self.quality_metric = quality_metric
        self.safety_mode = safety_mode
        self.passthrough_mode = passthrough_mode

        # Initialize adaptive priors manager
        if adaptive_config is None:
            adaptive_config = {
                'ph_threshold': 50.0,
                'ph_delta': 0.005,
                'buffer_size': 1000,
                'min_buffer_for_arima': 20,
                'arima_order': (1, 0, 1),
                'enable_arima': True
            }
        
        self.adaptive_manager = AdaptivePriorsManager(
            class_names=unique_classes_ordered_str,
            initial_priors=initial_class_priors_str_keys,
            **adaptive_config
        )
        
        # Determine exit classes for all models (Definition 2.5)
        self._determine_all_exit_classes()

    def _determine_all_exit_classes(self):
        """
        Determine exit classes EC(M_i) for all models (Definition 2.5).
        EC(M_i) = {C_j ∈ C | Q(M_i, C_j) ≥ Q(M_r, C_j) × (1 - ε)}
        """
        for model_name, model_obj in self.models_dict.items():
            model_obj.exit_classes = set()
            for class_name_str in self.unique_classes_ordered_str:
                quality_model = model_obj.get_quality(class_name_str, self.quality_metric)
                quality_role_model = self.role_model.get_quality(class_name_str, self.quality_metric)
                
                # Check if M_i provides ε-comparable quality for this class
                if quality_role_model == 0:
                    if quality_model == 0:
                        model_obj.exit_classes.add(class_name_str)
                elif quality_model >= quality_role_model * (1 - self.epsilon):
                    model_obj.exit_classes.add(class_name_str)
            
            # Role model exits for all classes
            if model_obj == self.role_model:
                model_obj.exit_classes = set(self.unique_classes_ordered_str)
                
            print(f"Model '{model_name}' exit classes: {model_obj.exit_classes}")

    def _calculate_utility(self, model, current_class_priors):
        """
        Calculate utility U(M_i) for model selection (Section 3.1).
        U(M_i) = Σ_{C_j ∈ EC(M_i)} Prob_current(C_j) / cost(M_i)
        """
        if model.cost == 0:
            sum_prob = sum(current_class_priors.get(cn_str, 0.0) for cn_str in model.exit_classes)
            return float('inf') if sum_prob > 0 else 0.0
        
        sum_prob_exit_classes = sum(current_class_priors.get(cn_str, 0.0) for cn_str in model.exit_classes)
        utility = float(sum_prob_exit_classes / model.cost) if model.cost > 0 else float('inf')
        return utility

    def _update_beliefs(self, p_in_str_keys, softmax_output):
        """
        Update beliefs using Bayesian update (Algorithm 3).
        Uses Hadamard product and renormalization.
        """
        p_temp_str_keys = {}
        current_sum = 0.0
        model_proba_class_labels_numeric = self.role_model.label_encoder_classes_

        if model_proba_class_labels_numeric is None or len(model_proba_class_labels_numeric) != len(softmax_output):
            # Fallback for mismatch
            if len(self.unique_classes_ordered_str) == len(softmax_output):
                for i, class_name_str_ordered in enumerate(self.unique_classes_ordered_str):
                    val = p_in_str_keys.get(class_name_str_ordered, 0.0) * softmax_output[i]
                    p_temp_str_keys[class_name_str_ordered] = val
                    current_sum += val
        else:
            # Standard update with Hadamard product (Line 5 in Algorithm 3)
            for i, numeric_label in enumerate(model_proba_class_labels_numeric):
                class_name_str = self.class_map_numeric_to_str.get(int(numeric_label))
                if class_name_str and class_name_str in p_in_str_keys:
                    val = p_in_str_keys[class_name_str] * softmax_output[i]
                    p_temp_str_keys[class_name_str] = val
                    current_sum += val
        
        # Normalization (Lines 6-11 in Algorithm 3)
        p_out_str_keys = {}
        if current_sum > EPSILON_THRESHOLD:
            for class_name_str_key in self.unique_classes_ordered_str:
                p_out_str_keys[class_name_str_key] = float(p_temp_str_keys.get(class_name_str_key, 0.0) / current_sum)
        else:
            # Fallback to uniform distribution
            num_classes = len(self.unique_classes_ordered_str)
            uniform_prob = 1.0 / num_classes if num_classes > 0 else 0.0
            for class_name_str_key in self.unique_classes_ordered_str:
                p_out_str_keys[class_name_str_key] = uniform_prob
        
        return p_out_str_keys

    def _get_passthrough_probability(self, model_k, class_j_str):
        """
        Compute passthrough probability Prob_pt(M, C_j) (Section 4.1).
        Uses either conservative (Appendix A) or relaxed estimation.
        """
        if self.passthrough_mode == "conservative":
            # Conservative estimation: uses only recall (Appendix A)
            return model_k.get_quality(class_j_str, "recall")
        else:
            # Relaxed estimation with Laplace smoothing (Section 4.1)
            true_label_numeric = self.class_map_str_to_numeric.get(class_j_str)
            
            if (true_label_numeric is None or model_k.label_encoder_classes_ is None or
                true_label_numeric not in model_k.label_encoder_classes_ or model_k.cm is None):
                return 0.0
                
            try:
                # Get row index for true class
                cm_row_idx = list(model_k.label_encoder_classes_).index(true_label_numeric)
                if cm_row_idx >= model_k.cm.shape[0]:
                    return 0.0
                    
                # Calculate misclassification probability into exit classes
                misclass_prob = 0.0
                N_j = np.sum(model_k.cm[cm_row_idx, :])  # Total samples of class j
                
                for exit_class_str in model_k.exit_classes:
                    exit_class_numeric = self.class_map_str_to_numeric.get(exit_class_str)
                    if exit_class_numeric is not None and exit_class_numeric in model_k.label_encoder_classes_:
                        cm_col_idx = list(model_k.label_encoder_classes_).index(exit_class_numeric)
                        if cm_col_idx < model_k.cm.shape[1]:
                            # Apply Laplace smoothing
                            n_misclass = model_k.cm[cm_row_idx, cm_col_idx]
                            prob_misclass = (n_misclass + ALPHA_SMOOTHING) / (N_j + ALPHA_SMOOTHING * len(self.unique_classes_ordered_str))
                            misclass_prob += prob_misclass
                
                # Passthrough probability = 1 - misclassification probability
                return max(0.0, 1.0 - misclass_prob)
                
            except (ValueError, IndexError):
                return 0.0

    def _check_chain_safety(self, M_new_name, S_current_names):
        """
        Check chain safety based on configured mode.
        Implements Algorithm 4 (global) or Algorithm 6 (class-based).
        """
        if self.safety_mode == "global":
            return self._check_global_chain_safety(M_new_name, S_current_names)
        else:
            return self._check_class_based_chain_safety(M_new_name, S_current_names)

    def _check_global_chain_safety(self, M_new_name, S_current_names):
        """
        Algorithm 4: Check Global Chain Safety.
        Ensures Σ_j Prob(C_j) * Q_proj(S, C_j) ≥ (1-ε) * Σ_j Prob(C_j) * Q(M_r, C_j)
        """
        S_potential = S_current_names + [M_new_name]
        Q_global_proj = 0.0
        
        for class_j_str in self.unique_classes_ordered_str:
            # Find exit model for this class
            M_exit_name = None
            prob_pass_cum = 1.0
            
            for model_name in S_potential:
                model_obj = self.models_dict[model_name]
                if class_j_str in model_obj.exit_classes:
                    M_exit_name = model_name
                    break
                else:
                    # Must pass through this model
                    prob_pass_cum *= self._get_passthrough_probability(model_obj, class_j_str)
            
            # If no exit model found, fallback to role model
            if M_exit_name is None:
                M_exit_name = self.role_model.name
                # Need to pass through all models in chain
                for model_name in S_potential:
                    if model_name != self.role_model.name:
                        model_obj = self.models_dict[model_name]
                        prob_pass_cum *= self._get_passthrough_probability(model_obj, class_j_str)
            
            # Calculate projected quality
            M_exit_obj = self.models_dict[M_exit_name]
            Q_proj_j = prob_pass_cum * M_exit_obj.get_quality(class_j_str, self.quality_metric)
            
            # Weight by class probability
            class_prob = self.initial_class_priors_str_keys.get(class_j_str, 0.0)
            Q_global_proj += class_prob * Q_proj_j
        
        # Calculate threshold
        Q_threshold = 0.0
        for class_j_str in self.unique_classes_ordered_str:
            class_prob = self.initial_class_priors_str_keys.get(class_j_str, 0.0)
            Q_threshold += class_prob * self.role_model.get_quality(class_j_str, self.quality_metric)
        Q_threshold *= (1 - self.epsilon)
        
        # Chain is safe if projected quality meets threshold
        return Q_global_proj >= Q_threshold

    def _check_class_based_chain_safety(self, M_new_name, S_current_names):
        """
        Algorithm 6: Check Class-Based Chain Safety (Appendix B).
        Ensures Q_proj(S, C_j) ≥ (1-ε) * Q(M_r, C_j) for all C_j
        """
        S_potential = S_current_names + [M_new_name]
        
        for class_j_str in self.unique_classes_ordered_str:
            # Find exit model for this class
            M_exit_name = None
            prob_pass_cum = 1.0
            
            for model_name in S_potential:
                model_obj = self.models_dict[model_name]
                if class_j_str in model_obj.exit_classes:
                    M_exit_name = model_name
                    break
                else:
                    # Must pass through this model
                    prob_pass_cum *= self._get_passthrough_probability(model_obj, class_j_str)
            
            # If no exit model found, fallback to role model
            if M_exit_name is None:
                M_exit_name = self.role_model.name
                for model_name in S_potential:
                    if model_name != self.role_model.name:
                        model_obj = self.models_dict[model_name]
                        prob_pass_cum *= self._get_passthrough_probability(model_obj, class_j_str)
            
            # Calculate projected quality for this class
            M_exit_obj = self.models_dict[M_exit_name]
            Q_proj = prob_pass_cum * M_exit_obj.get_quality(class_j_str, self.quality_metric)
            Q_thresh = (1 - self.epsilon) * self.role_model.get_quality(class_j_str, self.quality_metric)
            
            # Chain is unsafe if any class fails the threshold
            if Q_proj < Q_thresh:
                return False
        
        return True

    def process_event(self, event_features_df):
        """
        Algorithm 1: NOMAD Event Processing.
        Process a single event through the NOMAD framework.
        """
        # Line 2: Initialize
        M_available = self.models_dict.copy()
        Prob_current = self.adaptive_manager.get_current_priors()
        S_current = []  # Current model chain
        current_event_cost = 0.0
        final_prediction_str = None
        
        # Line 3: Main loop
        while M_available:
            # Line 4: Select next model (Algorithm 2)
            M_selected_name = self._select_next_model(Prob_current, M_available)
            
            # Line 5: Check if selection was successful
            if M_selected_name is None:
                break
            
            # Line 6: Check chain safety (Algorithm 4 or 6)
            if not self._check_chain_safety(M_selected_name, S_current):
                # Remove unsafe model and continue
                del M_available[M_selected_name]
                continue
            
            # Line 7: Execute selected model
            M_selected = self.models_dict[M_selected_name]
            pred_encoded = M_selected.predict(event_features_df)[0]
            softmax_vector = M_selected.predict_proba(event_features_df)[0]
            
            # Line 8: Append to chain
            S_current.append(M_selected_name)
            current_event_cost += M_selected.cost
            
            # Convert prediction to string class name
            pred_str = self.class_map_numeric_to_str.get(int(pred_encoded), "Unknown")
            final_prediction_str = pred_str
            
            # Line 9: Check exit condition
            if pred_str in M_selected.exit_classes:
                return final_prediction_str, current_event_cost, S_current
            
            # Line 10: Update beliefs (Algorithm 3)
            Prob_current = self._update_beliefs(Prob_current, softmax_vector)
            
            # Line 11: Remove selected model from available
            del M_available[M_selected_name]
        
        # Line 12-13: Fallback to role model if needed
        if self.role_model.name not in S_current:
            pred_encoded = self.role_model.predict(event_features_df)[0]
            final_prediction_str = self.class_map_numeric_to_str.get(int(pred_encoded), "Unknown")
            current_event_cost += self.role_model.cost
            S_current.append(self.role_model.name)
        
        return final_prediction_str, current_event_cost, S_current

    def _select_next_model(self, Prob_current, M_available):
        """
        Algorithm 2: Select Next Model.
        Select model with highest utility.
        """
        M_best = None
        U_max = -1
        
        for model_name, model_obj in M_available.items():
            # Calculate utility
            utility = self._calculate_utility(model_obj, Prob_current)
            
            if utility > U_max:
                U_max = utility
                M_best = model_name
        
        return M_best

    def stream_evaluate(self, X_test, y_test, batch_size=10, **kwargs):
        """
        Evaluate NOMAD on a stream of events.
        """
        predictions = []
        true_labels = []
        total_cost = 0.0
        model_run_counts = {name: 0 for name in self.models_dict.keys()}
        
        n_events = X_test.shape[0] if X_test is not None else 0
        
        for i in range(n_events):
            # Get event features
            if isinstance(X_test, pd.DataFrame):
                event_features_df = X_test.iloc[[i]]
            else:
                event_features_df = pd.DataFrame([X_test[i, :]])
            
            true_label = y_test[i]
            
            # Process event through NOMAD
            pred_str, cost, chain = self.process_event(event_features_df)
            
            # Convert prediction back to numeric
            pred_encoded = self.class_map_str_to_numeric.get(pred_str, -1)
            
            predictions.append(pred_encoded)
            true_labels.append(true_label)
            total_cost += cost
            
            # Update model run counts
            for model_name in chain:
                if model_name in model_run_counts:
                    model_run_counts[model_name] += 1
            
            # Update adaptive priors with true class
            true_class_str = self.class_map_numeric_to_str.get(int(true_label))
            if true_class_str:
                self.adaptive_manager.update(true_class_str)
            
            # Yield batch updates
            if (i + 1) % batch_size == 0 or i == n_events - 1:
                if true_labels:
                    current_accuracy = accuracy_score(true_labels, predictions)
                    yield {
                        "type": "batch_update",
                        "event_number": i + 1,
                        "cumulative_cost": float(total_cost),
                        "current_accuracy": current_accuracy,
                        "model_run_counts": model_run_counts.copy()
                    }
        
        # Final summary
        if true_labels:
            final_accuracy = accuracy_score(true_labels, predictions)
            avg_cost = total_cost / len(true_labels) if true_labels else 0.0
            
            yield {
                "type": "summary",
                "final_accuracy": float(final_accuracy),
                "average_cost": avg_cost,
                "total_events": len(true_labels),
                "model_run_counts": model_run_counts
            }
