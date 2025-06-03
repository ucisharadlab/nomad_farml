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
import pmdarima # For auto_arima

# Import classifiers (remains the same)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

# --- Configuration ---
CANDIDATE_MODELS_CONFIG = [ # (Ensure your models are diverse enough)
    ("Dummy (Uniform)", DummyClassifier(strategy="uniform"), 0.1),
    ("Decision Tree (Shallow)", DecisionTreeClassifier(max_depth=3, random_state=42), 1.0),
    ("Gaussian Naive Bayes", GaussianNB(), 1.2),
    ("K-Nearest Neighbors (K=5)", KNeighborsClassifier(n_neighbors=5), 2.5),
    ("Decision Tree (Deeper)", DecisionTreeClassifier(max_depth=10, random_state=42), 12.0),
    ("Random Forest (Small)", RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42), 25.0),
    ("Random Forest (Large)", RandomForestClassifier(n_estimators=100, random_state=42), 35.0),
]
ALPHA_SMOOTHING = 1
RANDOM_SEED_FOR_SAMPLING = 42
ARIMA_HISTORY_LENGTH = 100  # Number of past predictions to use for ARIMA fitting
MIN_ARIMA_HISTORY_FOR_FIT = 30 # Minimum data points needed before attempting to fit ARIMA

# --- Helper Functions --- (get_quality_metrics remains the same)
def get_quality_metrics(y_true, y_pred, labels, class_names_map=None):
    if not isinstance(y_true, (list, np.ndarray)) or not isinstance(y_pred, (list, np.ndarray)):
        return np.array([]), 0.0, {}, {"precision":0.0, "recall":0.0, "f1-score":0.0, "support":0}
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.array([]), 0.0, {}, {"precision":0.0, "recall":0.0, "f1-score":0.0, "support":0}
    if len(y_true) != len(y_pred):
        min_len = min(len(y_true), len(y_pred))
        y_true = np.array(y_true)[:min_len]; y_pred = np.array(y_pred)[:min_len]
        if min_len == 0: return np.array([]), 0.0, {}, {"precision":0.0, "recall":0.0, "f1-score":0.0, "support":0}
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)
    p_r_f1_s = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    metrics_per_class = {}
    if class_names_map:
        for i, label_numeric in enumerate(labels):
            class_name_str = class_names_map.get(int(label_numeric), str(label_numeric))
            metrics_per_class[class_name_str] = {
                "precision": float(p_r_f1_s[0][i]), "recall": float(p_r_f1_s[1][i]),
                "f1-score": float(p_r_f1_s[2][i]), "support": int(p_r_f1_s[3][i])}
    else:
        for i, label_numeric in enumerate(labels):
             metrics_per_class[str(label_numeric)] = {
                "precision": float(p_r_f1_s[0][i]), "recall": float(p_r_f1_s[1][i]),
                "f1-score": float(p_r_f1_s[2][i]), "support": int(p_r_f1_s[3][i])}
    weighted_avg = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
    avg_metrics = {"precision": float(weighted_avg[0]), "recall": float(weighted_avg[1]),
                   "f1-score": float(weighted_avg[2]), "support": int(np.sum(p_r_f1_s[3]))}
    return cm, float(acc), metrics_per_class, avg_metrics

# --- Model Class --- (No changes, assuming it's correct from before)
class Model:
    def __init__(self, name, classifier_instance, cost, preprocessor):
        self.name = name; self.cost = float(cost); self.classifier = classifier_instance
        self.cm = None; self.accuracy = 0.0; self.metrics_per_class = {}; self.avg_metrics = {}
        self.exit_classes = set(); self.label_encoder_classes_ = None 
        self.class_names_map = None; self.preprocessor = preprocessor
        self.trained_pipeline_ = None
    def train(self, X_train, y_train): # Simplified for brevity
        steps = [('preprocessor', self.preprocessor if self.preprocessor else StandardScaler()), ('classifier', self.classifier)]
        self.trained_pipeline_ = Pipeline(steps=steps)
        try: self.trained_pipeline_.fit(X_train, y_train)
        except Exception as e: print(f"ERROR training {self.name}: {e}")
    def predict(self, X_test): return self.trained_pipeline_.predict(X_test) if self.trained_pipeline_ else np.array([])
    def predict_proba(self, X_test):
        if not self.trained_pipeline_: return np.array([])
        if hasattr(self.trained_pipeline_.named_steps['classifier'], 'predict_proba'):
            return self.trained_pipeline_.predict_proba(X_test)
        preds = self.predict(X_test)
        if self.label_encoder_classes_ is None or len(self.label_encoder_classes_)==0: return np.zeros((X_test.shape[0],0))
        n_classes = len(self.label_encoder_classes_)
        probas = np.zeros((X_test.shape[0], n_classes))
        class_to_idx = {lbl: i for i, lbl in enumerate(self.label_encoder_classes_)}
        for i, p_enc in enumerate(preds):
            if p_enc in class_to_idx: probas[i, class_to_idx[p_enc]] = 1.0
        return probas
    def evaluate(self, X_test, y_test, labels, class_names_map):
        self.label_encoder_classes_ = labels; self.class_names_map = class_names_map
        if X_test.shape[0] == 0: self.accuracy = 0.0; self.metrics_per_class = {}; self.avg_metrics = {}; self.cm = np.array([]); return
        y_pred = self.predict(X_test)
        self.cm, self.accuracy, self.metrics_per_class, self.avg_metrics = get_quality_metrics(y_test, y_pred, labels=labels, class_names_map=class_names_map)
    def get_quality(self, class_name_str, metric_type="f1-score"):
        return float(self.metrics_per_class.get(class_name_str, {}).get(metric_type, 0.0))
    def __repr__(self): return f"Model(name='{self.name}', cost={self.cost}, accuracy={self.accuracy:.4f})"

# --- NOMAD Engine Class ---
class NOMADEngine:
    def __init__(self, models_dict, role_model_name, epsilon,
                 initial_class_priors_str_keys, unique_classes_ordered_str, class_map_numeric_to_str,
                 quality_metric_for_ec="f1-score", safety_check_type="conservative",
                 adaptive_update_window=50, adaptive_beta=0.3): # adaptive_update_window for ARIMA refit, beta for smoothing
        
        self.models_dict = models_dict
        if role_model_name not in self.models_dict:
            raise ValueError(f"Role model '{role_model_name}' not found.")
        self.role_model = self.models_dict[role_model_name]
        self.epsilon = float(epsilon)
        
        self.initial_class_priors_str_keys = initial_class_priors_str_keys.copy() # For "Static Priors NOMAD"
        self.unique_classes_ordered_str = unique_classes_ordered_str
        self.class_map_numeric_to_str = class_map_numeric_to_str
        self.class_map_str_to_numeric = {v: k for k, v in self.class_map_numeric_to_str.items()}

        self.quality_metric_for_ec = quality_metric_for_ec
        self.safety_check_type = safety_check_type

        # ARIMA specific attributes
        self.adaptive_update_window = int(adaptive_update_window) # How often to refit ARIMA
        self.adaptive_beta = float(adaptive_beta) # Smoothing factor for ARIMA priors
        self.current_dynamic_class_priors_arima = self.initial_class_priors_str_keys.copy() # Start ARIMA priors from initial
        
        # History for ARIMA: stores 0/1 for each class based on ARIMA path's prediction
        self.arima_prediction_history_per_class = {
            cls_name: [] for cls_name in self.unique_classes_ordered_str
        }
        self.arima_models_per_class = {
            cls_name: None for cls_name in self.unique_classes_ordered_str
        }
        
        print(f"DEBUG NOMADEngine initialized. Role: {self.role_model.name}. Adaptive Window (ARIMA refit): {self.adaptive_update_window}, Beta (ARIMA smooth): {self.adaptive_beta}")
        self._determine_all_exit_classes()

    def _determine_all_exit_classes(self): # (Same as before)
        for model_name, model_obj in self.models_dict.items():
            model_obj.exit_classes = set()
            for class_name_str in self.unique_classes_ordered_str:
                quality_model = model_obj.get_quality(class_name_str, self.quality_metric_for_ec)
                quality_role_model = self.role_model.get_quality(class_name_str, self.quality_metric_for_ec)
                if quality_role_model == 0:
                    if quality_model == 0: model_obj.exit_classes.add(class_name_str)
                elif quality_model >= quality_role_model * (1 - self.epsilon):
                    model_obj.exit_classes.add(class_name_str)
            if model_obj == self.role_model: model_obj.exit_classes = set(self.unique_classes_ordered_str)
    
    def _calculate_utility(self, model, current_priors_for_path_str_keys): # (Same as before)
        if model.cost == 0:
            sum_prob = sum(current_priors_for_path_str_keys.get(cn_str, 0.0) for cn_str in model.exit_classes)
            return float('inf') if sum_prob > 0 else 0.0
        sum_prob_exit_classes = sum(current_priors_for_path_str_keys.get(cn_str, 0.0) for cn_str in model.exit_classes)
        return float(sum_prob_exit_classes / model.cost) if model.cost > 0 else float('inf')

    def _update_probabilities(self, p_in_str_keys, softmax_output): # (Same as before)
        p_temp_str_keys = {}; current_sum = 0.0
        model_proba_class_labels_numeric = self.role_model.label_encoder_classes_
        if model_proba_class_labels_numeric is None or len(model_proba_class_labels_numeric) != len(softmax_output):
            print(f"CRITICAL WARNING: _update_probabilities class order mismatch. Softmax len: {len(softmax_output)}, Expected labels len: {len(model_proba_class_labels_numeric or [])}.")
            if len(self.unique_classes_ordered_str) == len(softmax_output):
                for i, cn_ord_str in enumerate(self.unique_classes_ordered_str):
                    val = p_in_str_keys.get(cn_ord_str, 0.0) * softmax_output[i]; p_temp_str_keys[cn_ord_str] = val; current_sum += val
            else: print(f"CRITICAL ERROR: Fallback for _update_probabilities also failed.")
        else:
            for i, num_label in enumerate(model_proba_class_labels_numeric):
                cn_str = self.class_map_numeric_to_str.get(int(num_label))
                if cn_str and cn_str in p_in_str_keys:
                    val = p_in_str_keys[cn_str] * softmax_output[i]; p_temp_str_keys[cn_str] = val; current_sum += val
        p_out_str_keys = {}
        if current_sum > 1e-9:
            for cn_key_str in self.unique_classes_ordered_str: p_out_str_keys[cn_key_str] = float(p_temp_str_keys.get(cn_key_str, 0.0) / current_sum)
        else:
            num_cls = len(self.unique_classes_ordered_str)
            unif_prob = 1.0 / num_cls if num_cls > 0 else 0.0
            for cn_key_str in self.unique_classes_ordered_str: p_out_str_keys[cn_key_str] = unif_prob
        return p_out_str_keys

    def _check_chain_safety_conservative(self, M_new_name, S_current_names): # (Same as before)
        S_potential_names = S_current_names + [M_new_name]
        for class_name_target_str in self.unique_classes_ordered_str:
            M_exit_for_class_j_name = None; M_exit_for_class_j_idx = -1
            S_final_for_this_class_check = S_potential_names
            for idx, model_name_in_chain in enumerate(S_potential_names):
                model_in_chain = self.models_dict[model_name_in_chain]
                if class_name_target_str in model_in_chain.exit_classes: M_exit_for_class_j_name = model_name_in_chain; M_exit_for_class_j_idx = idx; break
            if M_exit_for_class_j_name is None:
                S_final_for_this_class_check = S_potential_names + ([self.role_model.name] if self.role_model.name not in S_potential_names else [])
                M_exit_for_class_j_name = self.role_model.name
                if M_exit_for_class_j_name not in S_final_for_this_class_check: return False
                M_exit_for_class_j_idx = S_final_for_this_class_check.index(M_exit_for_class_j_name)
            chain_recall_j = 1.0
            for k_idx in range(M_exit_for_class_j_idx + 1): chain_recall_j *= self.models_dict[S_final_for_this_class_check[k_idx]].get_quality(class_name_target_str, "recall")
            recall_role_model_cj = self.role_model.get_quality(class_name_target_str, "recall")
            if chain_recall_j < (1.0 - self.epsilon) * recall_role_model_cj: return False
        return True

    def _get_smoothed_misclassification_prob(self, model_k, class_j_true_str, class_i_pred_str): # (Same as before)
        true_label_numeric = self.class_map_str_to_numeric.get(class_j_true_str); pred_label_numeric = self.class_map_str_to_numeric.get(class_i_pred_str)
        if true_label_numeric is None or model_k.label_encoder_classes_ is None or true_label_numeric not in model_k.label_encoder_classes_: return 0.0
        try: cm_row_idx = list(model_k.label_encoder_classes_).index(true_label_numeric)
        except ValueError: return 0.0
        n_ji = 0
        if pred_label_numeric is not None and pred_label_numeric in model_k.label_encoder_classes_:
            try: 
                cm_col_idx = list(model_k.label_encoder_classes_).index(pred_label_numeric)
                if model_k.cm is not None and cm_row_idx < model_k.cm.shape[0] and cm_col_idx < model_k.cm.shape[1]: n_ji = int(model_k.cm[cm_row_idx, cm_col_idx])
            except ValueError: pass
        sum_n_jl = int(np.sum(model_k.cm[cm_row_idx, :])) if model_k.cm is not None and cm_row_idx < model_k.cm.shape[0] else 0
        S_total_pseudo_counts = ALPHA_SMOOTHING * len(self.unique_classes_ordered_str)
        if (sum_n_jl + S_total_pseudo_counts) == 0: return 0.0
        return float(n_ji + ALPHA_SMOOTHING) / (sum_n_jl + S_total_pseudo_counts)

    def _check_chain_safety_relaxed(self, M_new_name, S_current_names): # (Same as before)
        S_potential_names = S_current_names + [M_new_name]
        for class_j_true_str in self.unique_classes_ordered_str:
            M_exit_for_class_j_name = None; M_exit_for_class_j_idx = -1; S_final_for_this_class_check = S_potential_names
            for idx, model_name_in_chain in enumerate(S_potential_names):
                model_in_chain = self.models_dict[model_name_in_chain]
                if class_j_true_str in model_in_chain.exit_classes: M_exit_for_class_j_name = model_name_in_chain; M_exit_for_class_j_idx = idx; break
            if M_exit_for_class_j_name is None:
                S_final_for_this_class_check = S_potential_names + ([self.role_model.name] if self.role_model.name not in S_potential_names else [])
                M_exit_for_class_j_name = self.role_model.name
                if M_exit_for_class_j_name not in S_final_for_this_class_check: return False
                M_exit_for_class_j_idx = S_final_for_this_class_check.index(M_exit_for_class_j_name)
            M_exit_model_obj = self.models_dict[M_exit_for_class_j_name]; prod_pass_through_refined = 1.0
            for k_idx in range(M_exit_for_class_j_idx):
                model_k_obj = self.models_dict[S_final_for_this_class_check[k_idx]]
                prob_k_misclassify = sum(self._get_smoothed_misclassification_prob(model_k_obj, class_j_true_str, c_i_pred) for c_i_pred in model_k_obj.exit_classes)
                prod_pass_through_refined *= (1.0 - prob_k_misclassify)
            chain_quality_j_refined = prod_pass_through_refined * M_exit_model_obj.get_quality(class_j_true_str, "recall")
            recall_role_model_cj = self.role_model.get_quality(class_j_true_str, "recall")
            if chain_quality_j_refined < (1.0 - self.epsilon) * recall_role_model_cj: return False
        return True

    def _resolve_event_with_priors(self, event_features_df, current_priors_for_path_str_keys):
        # This method contains the logic from the original select_and_classify_event's while loop
        # It uses the passed current_priors_for_path_str_keys for its decision making.
        # print(f"DEBUG _resolve_event: Using priors: {current_priors_for_path_str_keys}")
        
        current_priors_for_this_event_str_keys = current_priors_for_path_str_keys.copy()
        models_available_dict = self.models_dict.copy()
        realized_chain_names = []
        current_event_cost = 0.0
        final_prediction_name_str = None

        while models_available_dict:
            utility_candidates = []
            if not models_available_dict: break
            for model_name, model_obj in models_available_dict.items():
                utility_candidates.append(
                    (self._calculate_utility(model_obj, current_priors_for_this_event_str_keys),
                     -model_obj.cost, model_name)
                )
            if not utility_candidates: break
            utility_candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
            
            selected_model_name_by_utility = utility_candidates[0][2]
            M_star = self.models_dict[selected_model_name_by_utility]
            
            is_safe = False # Safety check logic remains the same
            if self.safety_check_type == "conservative": is_safe = self._check_chain_safety_conservative(M_star.name, realized_chain_names)
            elif self.safety_check_type == "relaxed": is_safe = self._check_chain_safety_relaxed(M_star.name, realized_chain_names)

            if not is_safe:
                del models_available_dict[M_star.name]
                if M_star == self.role_model: break 
                continue

            pred_current_encoded = M_star.predict(event_features_df)[0]
            pred_current_name_str = self.class_map_numeric_to_str.get(int(pred_current_encoded), "UnknownPrediction")
            softmax_vector = M_star.predict_proba(event_features_df)[0]
            current_event_cost += M_star.cost
            realized_chain_names.append(M_star.name)
            final_prediction_name_str = pred_current_name_str

            if pred_current_name_str in M_star.exit_classes:
                return final_prediction_name_str, current_event_cost, realized_chain_names

            del models_available_dict[M_star.name]
            current_priors_for_this_event_str_keys = self._update_probabilities(
                current_priors_for_this_event_str_keys, softmax_vector
            )
            if M_star == self.role_model: break
        
        # Fallback to Role Model if loop finishes
        if self.role_model.name not in realized_chain_names:
            current_event_cost += self.role_model.cost
            realized_chain_names.append(self.role_model.name)
        pred_final_encoded = self.role_model.predict(event_features_df)[0]
        final_prediction_name_str = self.class_map_numeric_to_str.get(int(pred_final_encoded), "UnknownRolePrediction")
        return final_prediction_name_str, current_event_cost, realized_chain_names

    def _update_arima_priors(self, current_event_count):
        if self.adaptive_update_window <= 0: # ARIMA adaptation disabled
            return

        if current_event_count > 0 and current_event_count % self.adaptive_update_window == 0:
            print(f"DEBUG _update_arima_priors: Attempting to refit ARIMA models at event {current_event_count}")
            new_arima_forecast_dist_q = {} # string_name -> forecasted_prob

            for class_name_str in self.unique_classes_ordered_str:
                history = self.arima_prediction_history_per_class.get(class_name_str, [])
                
                if len(history) < MIN_ARIMA_HISTORY_FOR_FIT: # Not enough data points
                    print(f"DEBUG _update_arima_priors: Insufficient history for class '{class_name_str}' (len {len(history)}). Using previous prior.")
                    new_arima_forecast_dist_q[class_name_str] = self.current_dynamic_class_priors_arima.get(class_name_str, 
                                                                1.0/max(1,len(self.unique_classes_ordered_str))) # Fallback
                    continue
                
                try:
                    # Ensure history is a 1D array or list for pmdarima
                    series = pd.Series(history) 
                    # Check for constant series, which can cause issues with ARIMA
                    if series.nunique() <= 1:
                        print(f"DEBUG _update_arima_priors: History for class '{class_name_str}' is constant. Using mean as forecast.")
                        forecast = series.mean()
                    else:
                        arima_model = pmdarima.auto_arima(
                            series,
                            start_p=1, start_q=1, max_p=3, max_q=3, m=1, # Non-seasonal ARIMA
                            start_P=0, seasonal=False, d=None, D=0, trace=False,
                            error_action='ignore', suppress_warnings=True, stepwise=True
                        )
                        self.arima_models_per_class[class_name_str] = arima_model
                        forecast = arima_model.predict(n_periods=1)[0]
                    
                    # Forecast might be outside [0,1], clip it
                    new_arima_forecast_dist_q[class_name_str] = np.clip(forecast, 0.0, 1.0)
                    print(f"DEBUG _update_arima_priors: Class '{class_name_str}', History len: {len(history)}, Forecast: {forecast:.4f}, Clipped: {new_arima_forecast_dist_q[class_name_str]:.4f}")

                except Exception as e_arima:
                    print(f"ERROR _update_arima_priors: ARIMA fitting/prediction failed for class '{class_name_str}': {e_arima}")
                    print(traceback.format_exc())
                    # Fallback: use the previous dynamic prior for this class if ARIMA fails
                    new_arima_forecast_dist_q[class_name_str] = self.current_dynamic_class_priors_arima.get(class_name_str, 
                                                                1.0/max(1,len(self.unique_classes_ordered_str)))


            # Normalize Q_ARIMA
            sum_q_arima = sum(new_arima_forecast_dist_q.values())
            if sum_q_arima > 1e-9:
                q_arima_normalized = {k: v / sum_q_arima for k, v in new_arima_forecast_dist_q.items()}
            else: # Fallback to uniform if sum is zero (e.g., all forecasts were zero)
                print("Warning _update_arima_priors: Sum of ARIMA forecasts is zero. Using uniform distribution.")
                num_cls = len(self.unique_classes_ordered_str)
                unif_p = 1.0 / num_cls if num_cls > 0 else 0.0
                q_arima_normalized = {cls_name: unif_p for cls_name in self.unique_classes_ordered_str}
            
            print(f"DEBUG _update_arima_priors: Normalized Q_ARIMA: {q_arima_normalized}")

            # Smooth update of current_dynamic_class_priors_arima
            updated_priors = {}
            for class_name_str in self.unique_classes_ordered_str:
                old_prior = self.current_dynamic_class_priors_arima.get(class_name_str, 0.0)
                new_forecasted_prior = q_arima_normalized.get(class_name_str, 0.0)
                updated_priors[class_name_str] = (self.adaptive_beta * new_forecasted_prior) + \
                                                 ((1.0 - self.adaptive_beta) * old_prior)
            
            # Final normalization for the smoothed priors
            sum_updated_priors = sum(updated_priors.values())
            if sum_updated_priors > 1e-9:
                self.current_dynamic_class_priors_arima = {k: v / sum_updated_priors for k, v in updated_priors.items()}
            else: # Should be rare if previous normalizations worked
                self.current_dynamic_class_priors_arima = q_arima_normalized # Fallback to pre-smoothing normalized

            print(f"DEBUG _update_arima_priors: Smoothed & Updated ARIMA dynamic priors: {self.current_dynamic_class_priors_arima}")


    def stream_evaluate_strategy(self,
                                 X_test_overall, y_test_encoded_overall,
                                 candidate_model_names_for_counts, X_test_column_names,
                                 batch_size=10, workload_phases=None,
                                 X_by_class_str_keys=None, y_by_class_str_keys=None):

        # Lists for Static Priors Path
        nomad_static_preds_list = []
        total_cost_static = 0.0
        # Lists for ARIMA Adaptive Priors Path
        nomad_arima_preds_list = []
        total_cost_arima = 0.0
        
        true_labels_final_list = [] # Common for both paths
        
        model_run_counts_static = {name: 0 for name in candidate_model_names_for_counts}
        model_run_counts_arima = {name: 0 for name in candidate_model_names_for_counts}
        
        events_actually_processed_count = 0

        # Reset ARIMA history at the start of a new stream evaluation
        self.arima_prediction_history_per_class = { cls_name: [] for cls_name in self.unique_classes_ordered_str }
        self.current_dynamic_class_priors_arima = self.initial_class_priors_str_keys.copy() # Reset ARIMA priors
        self.arima_models_per_class = { cls_name: None for cls_name in self.unique_classes_ordered_str }


        if workload_phases and X_by_class_str_keys and y_by_class_str_keys and len(workload_phases) > 0:
            total_events_defined_by_plan = sum(phase.get('duration', 0) for phase in workload_phases)
        else:
            total_events_defined_by_plan = X_test_overall.shape[0] if X_test_overall is not None else 0
        print(f"DEBUG stream_evaluate_strategy: Total events planned: {total_events_defined_by_plan}")


        yield { "type": "setup", "total_events": total_events_defined_by_plan,
                "role_model_static_accuracy": float(self.role_model.accuracy),
                "role_model_cost_per_event": float(self.role_model.cost),
                "exit_classes_info": {name: list(model.exit_classes) for name, model in self.models_dict.items()},
                "class_names_ordered": self.unique_classes_ordered_str,
                "initial_priors": {k_str: float(v) for k_str, v in self.initial_class_priors_str_keys.items()} }

        previous_phase_target_dist_probs_for_kl = np.array([ self.initial_class_priors_str_keys.get(c, 0.0) for c in self.unique_classes_ordered_str ])
        # ... (KL divergence previous_phase_target_dist_probs normalization logic from before) ...
        previous_phase_target_dist_probs_for_kl = np.maximum(previous_phase_target_dist_probs_for_kl, 1e-9)
        if np.sum(previous_phase_target_dist_probs_for_kl) > 1e-9: previous_phase_target_dist_probs_for_kl /= np.sum(previous_phase_target_dist_probs_for_kl)
        else: previous_phase_target_dist_probs_for_kl = np.full(len(self.unique_classes_ordered_str), 1.0/max(1,len(self.unique_classes_ordered_str)))


        current_event_index_in_plan = 0

        if workload_phases and X_by_class_str_keys and y_by_class_str_keys and len(workload_phases) > 0:
            # --- Phased Workload Execution ---
            # ... (phase setup, KL divergence yield, sampling logic for phase_X_sampled_df, phase_y_sampled_encoded from previous response) ...
            # This part generates event_features_df and true_label_encoded_this_event for each iteration
            random.seed(RANDOM_SEED_FOR_SAMPLING); np.random.seed(RANDOM_SEED_FOR_SAMPLING)
            print("DEBUG stream_evaluate_strategy: Starting Phased Workload Execution.")
            for phase_idx, phase_def in enumerate(workload_phases):
                duration = phase_def.get('duration', 0)
                target_dist_str_keys = phase_def.get('target_distribution', {})
                # ... (KL divergence yield for phase shift - from previous response) ...
                current_phase_target_dist_probs = np.array([target_dist_str_keys.get(c_str, 0.0) for c_str in self.unique_classes_ordered_str])
                # ... (Normalize current_phase_target_dist_probs) ...
                if not np.isclose(np.sum(current_phase_target_dist_probs), 1.0) or np.any(current_phase_target_dist_probs < 0):
                    if np.sum(current_phase_target_dist_probs) > 1e-9 : current_phase_target_dist_probs /= np.sum(current_phase_target_dist_probs)
                    else: current_phase_target_dist_probs = np.ones(len(self.unique_classes_ordered_str)) / max(1, len(self.unique_classes_ordered_str))
                target_dist_str_keys = {self.unique_classes_ordered_str[i]: current_phase_target_dist_probs[i] for i in range(len(self.unique_classes_ordered_str))}
                q_probs_for_kl = np.maximum(current_phase_target_dist_probs, 1e-9); q_probs_for_kl /= np.sum(q_probs_for_kl)
                kl_div = entropy(pk=q_probs_for_kl, qk=previous_phase_target_dist_probs_for_kl) if len(q_probs_for_kl) == len(previous_phase_target_dist_probs_for_kl) else -1.0
                yield {"type": "phase_shift", "phase_index": phase_idx, "duration": duration, "target_distribution": target_dist_str_keys,
                       "kl_divergence_from_previous": float(kl_div if np.isfinite(kl_div) else -1.0),
                       "previous_target_distribution": {self.unique_classes_ordered_str[i]: previous_phase_target_dist_probs_for_kl[i] for i in range(len(previous_phase_target_dist_probs_for_kl))}}
                previous_phase_target_dist_probs_for_kl = current_phase_target_dist_probs.copy()

                phase_X_list, phase_y_list = [], [] # ... (sampling logic as before) ...
                num_samples_per_class_in_phase = { c_str: int(round(duration * target_dist_str_keys.get(c_str, 0.0))) for c_str in self.unique_classes_ordered_str }
                # ... (adjust rounding for num_samples_per_class_in_phase) ...
                current_total_samples_calc = sum(num_samples_per_class_in_phase.values())
                if current_total_samples_calc != duration and duration > 0:
                    diff = duration - current_total_samples_calc
                    if diff != 0:
                        adjust_class_candidates = [k for k,v in target_dist_str_keys.items() if v > 0] or [self.unique_classes_ordered_str[0]] if self.unique_classes_ordered_str else []
                        if adjust_class_candidates:
                            adjust_class = random.choice(adjust_class_candidates)
                            num_samples_per_class_in_phase[adjust_class] = max(0, num_samples_per_class_in_phase.get(adjust_class,0) + diff)

                for class_name_str, num_samples in num_samples_per_class_in_phase.items():
                    if num_samples == 0: continue
                    available_X_df = X_by_class_str_keys.get(class_name_str); available_y_arr = y_by_class_str_keys.get(class_name_str)
                    if available_X_df is None or available_X_df.empty: continue
                    indices_to_sample = random.choices(range(available_X_df.shape[0]), k=num_samples)
                    phase_X_list.extend([available_X_df.iloc[k] for k in indices_to_sample])
                    phase_y_list.extend([available_y_arr[k] for k in indices_to_sample])
                
                if not phase_X_list: current_event_index_in_plan += duration; continue
                combined_phase_data = list(zip(phase_X_list, phase_y_list)); random.shuffle(combined_phase_data)
                phase_X_sampled_df = pd.DataFrame([item[0] for item in combined_phase_data], columns=X_test_column_names)
                phase_y_sampled_encoded = np.array([item[1] for item in combined_phase_data])

                for i_phase in range(phase_X_sampled_df.shape[0]):
                    current_event_index_in_plan += 1
                    event_features_df = phase_X_sampled_df.iloc[[i_phase]] # Ensure it's a DataFrame
                    true_label_encoded_this_event = phase_y_sampled_encoded[i_phase]
                    true_labels_final_list.append(true_label_encoded_this_event)
                    events_actually_processed_count += 1

                    # --- Static Priors Path ---
                    pred_static_str, cost_static, chain_static = self._resolve_event_with_priors(
                        event_features_df, self.initial_class_priors_str_keys
                    )
                    pred_static_encoded = self.class_map_str_to_numeric.get(pred_static_str, -1)
                    nomad_static_preds_list.append(pred_static_encoded)
                    total_cost_static += cost_static
                    for model_name in chain_static: model_run_counts_static[model_name] +=1

                    # --- ARIMA Adaptive Priors Path ---
                    pred_arima_str, cost_arima, chain_arima = self._resolve_event_with_priors(
                        event_features_df, self.current_dynamic_class_priors_arima
                    )
                    pred_arima_encoded = self.class_map_str_to_numeric.get(pred_arima_str, -1)
                    nomad_arima_preds_list.append(pred_arima_encoded)
                    total_cost_arima += cost_arima
                    for model_name in chain_arima: model_run_counts_arima[model_name] +=1
                    
                    # Update ARIMA prediction history (based on ARIMA path's prediction)
                    for cls_name_hist in self.unique_classes_ordered_str:
                        self.arima_prediction_history_per_class[cls_name_hist].append(1 if cls_name_hist == pred_arima_str else 0)
                        if len(self.arima_prediction_history_per_class[cls_name_hist]) > ARIMA_HISTORY_LENGTH:
                            self.arima_prediction_history_per_class[cls_name_hist].pop(0) # Keep fixed history length
                    
                    # Trigger ARIMA prior update if interval met
                    self._update_arima_priors(events_actually_processed_count) # Pass actual processed count

                    # Batch update
                    if (current_event_index_in_plan % batch_size == 0) or \
                       (current_event_index_in_plan == total_events_defined_by_plan):
                        live_acc_static = accuracy_score(true_labels_final_list, nomad_static_preds_list) if nomad_static_preds_list else 0.0
                        live_acc_arima = accuracy_score(true_labels_final_list, nomad_arima_preds_list) if nomad_arima_preds_list else 0.0
                        print(f"DEBUG BATCH YIELD (Event Plan: {current_event_index_in_plan}, Actual Proc: {events_actually_processed_count}): "
                              f"CostStatic={total_cost_static}, AccStatic={live_acc_static:.3f}, "
                              f"CostARIMA={total_cost_arima}, AccARIMA={live_acc_arima:.3f}")
                        yield { "type": "batch_update",
                                "last_event_in_batch": current_event_index_in_plan,
                                "nomad_static_cumulative_cost": float(total_cost_static),
                                "nomad_static_live_accuracy": live_acc_static,
                                "nomad_arima_cumulative_cost": float(total_cost_arima),
                                "nomad_arima_live_accuracy": live_acc_arima,
                                # For simplicity, model_run_counts can be from one path e.g. ARIMA, or averaged/summed
                                "model_run_counts_snapshot": {k:v for k,v in model_run_counts_arima.items()}
                              }
        else: # --- Sequential Workload Execution ---
            if X_test_overall is not None and not X_test_overall.empty:
                for i in range(X_test_overall.shape[0]):
                    current_event_index_in_plan += 1
                    event_features_df = X_test_overall.iloc[[i]]
                    true_label_encoded_this_event = y_test_encoded_overall[i]
                    true_labels_final_list.append(true_label_encoded_this_event)
                    events_actually_processed_count += 1

                    # Static Priors Path
                    pred_static_str, cost_static, chain_static = self._resolve_event_with_priors(event_features_df, self.initial_class_priors_str_keys)
                    pred_static_encoded = self.class_map_str_to_numeric.get(pred_static_str, -1); nomad_static_preds_list.append(pred_static_encoded)
                    total_cost_static += cost_static; 
                    for mn in chain_static: model_run_counts_static[mn] +=1


                    # ARIMA Adaptive Priors Path
                    pred_arima_str, cost_arima, chain_arima = self._resolve_event_with_priors(event_features_df, self.current_dynamic_class_priors_arima)
                    pred_arima_encoded = self.class_map_str_to_numeric.get(pred_arima_str, -1); nomad_arima_preds_list.append(pred_arima_encoded)
                    total_cost_arima += cost_arima
                    for mn in chain_arima: model_run_counts_arima[mn] +=1

                    for cls_name_hist in self.unique_classes_ordered_str:
                        self.arima_prediction_history_per_class[cls_name_hist].append(1 if cls_name_hist == pred_arima_str else 0)
                        if len(self.arima_prediction_history_per_class[cls_name_hist]) > ARIMA_HISTORY_LENGTH: self.arima_prediction_history_per_class[cls_name_hist].pop(0)
                    self._update_arima_priors(events_actually_processed_count)

                    if (current_event_index_in_plan % batch_size == 0) or (current_event_index_in_plan == total_events_defined_by_plan):
                        live_acc_static = accuracy_score(true_labels_final_list, nomad_static_preds_list) if nomad_static_preds_list else 0.0
                        live_acc_arima = accuracy_score(true_labels_final_list, nomad_arima_preds_list) if nomad_arima_preds_list else 0.0
                        print(f"DEBUG BATCH YIELD (Event Plan: {current_event_index_in_plan}, Actual Proc: {events_actually_processed_count}): "
                              f"CostStatic={total_cost_static}, AccStatic={live_acc_static:.3f}, "
                              f"CostARIMA={total_cost_arima}, AccARIMA={live_acc_arima:.3f}")
                        yield { "type": "batch_update", "last_event_in_batch": current_event_index_in_plan,
                                "nomad_static_cumulative_cost": float(total_cost_static), "nomad_static_live_accuracy": live_acc_static,
                                "nomad_arima_cumulative_cost": float(total_cost_arima), "nomad_arima_live_accuracy": live_acc_arima,
                                "model_run_counts_snapshot": {k:v for k,v in model_run_counts_arima.items()} } # Or static, or combined
            else: print("DEBUG stream_evaluate_strategy: X_test_overall is None or empty, skipping sequential processing loop.")


        # --- Final Summary ---
        print(f"DEBUG stream_evaluate_strategy: Finished. Actually processed: {events_actually_processed_count} events.")
        avg_cost_static = float(total_cost_static / events_actually_processed_count) if events_actually_processed_count > 0 else 0.0
        avg_cost_arima = float(total_cost_arima / events_actually_processed_count) if events_actually_processed_count > 0 else 0.0
        
        summary_payload = {"type": "summary"}
        eval_labels_numeric = self.role_model.label_encoder_classes_ # Use role model's for consistency
        if eval_labels_numeric is None: # Fallback
            eval_labels_numeric = [self.class_map_str_to_numeric.get(s) for s in self.unique_classes_ordered_str if self.class_map_str_to_numeric.get(s) is not None]
            if not eval_labels_numeric and true_labels_final_list : eval_labels_numeric = np.unique(true_labels_final_list).tolist()


        if events_actually_processed_count > 0 and eval_labels_numeric:
            cm_static, acc_static, metrics_pc_static, avg_m_static = get_quality_metrics(true_labels_final_list, nomad_static_preds_list, labels=list(eval_labels_numeric), class_names_map=self.class_map_numeric_to_str)
            cm_arima, acc_arima, metrics_pc_arima, avg_m_arima = get_quality_metrics(true_labels_final_list, nomad_arima_preds_list, labels=list(eval_labels_numeric), class_names_map=self.class_map_numeric_to_str)
            summary_payload["nomad_static_overall_metrics"] = {"accuracy": float(acc_static), "avg_metrics": avg_m_static, "average_cost": avg_cost_static, "cm": cm_static.tolist() if cm_static.size > 0 else []}
            summary_payload["nomad_static_per_class_metrics"] = metrics_pc_static
            summary_payload["nomad_arima_overall_metrics"] = {"accuracy": float(acc_arima), "avg_metrics": avg_m_arima, "average_cost": avg_cost_arima, "cm": cm_arima.tolist() if cm_arima.size > 0 else []}
            summary_payload["nomad_arima_per_class_metrics"] = metrics_pc_arima
        else: # Default empty metrics
            empty_overall = {"accuracy": 0.0, "avg_metrics": {"precision":0.0, "recall":0.0, "f1-score":0.0, "support":0}, "average_cost": 0.0, "cm": []}
            empty_per_class = {c: {"precision":0.0, "recall":0.0, "f1-score":0.0, "support":0} for c in self.unique_classes_ordered_str}
            summary_payload["nomad_static_overall_metrics"] = empty_overall
            summary_payload["nomad_static_per_class_metrics"] = empty_per_class
            summary_payload["nomad_arima_overall_metrics"] = empty_overall
            summary_payload["nomad_arima_per_class_metrics"] = empty_per_class

        summary_payload["role_model_performance"] = { "name": self.role_model.name, "accuracy": float(self.role_model.accuracy),
                                        "avg_metrics": self.role_model.avg_metrics, "cost": float(self.role_model.cost),
                                        "cm": self.role_model.cm.tolist() if self.role_model.cm is not None else [],
                                        "per_class_metrics": self.role_model.metrics_per_class }
        # For run counts, maybe show combined or from one strategy
        summary_payload["final_model_run_counts_static"] = {k:v for k,v in model_run_counts_static.items()}
        summary_payload["final_model_run_counts_arima"] = {k:v for k,v in model_run_counts_arima.items()}
        
        yield summary_payload


# --- load_and_preprocess_data_for_flask & train_evaluate_individual_models_for_flask & run_nomad_simulation_for_flask_streamed
# (These remain the same as the versions from my response to "Give me both files completely with logging"
# as they correctly set up and pass data including segregated data for phases)
# For brevity, I will not paste them again here but assume they are the last correct versions.
# Make sure `run_nomad_simulation_for_flask_streamed` correctly passes all params to `NOMADEngine`.
# And `load_and_preprocess_data_for_flask` returns X_by_class_str_keys and y_by_class_str_keys.
# The functions `train_evaluate_individual_models_for_flask` and `run_nomad_simulation_for_flask_streamed`
# are assumed to be the last versions I provided that were working with the previous logging.
# The critical change is inside NOMADEngine.

# --- load_and_preprocess_data_for_flask (mostly same, ensure segregated data is DataFrame/array) ---
def load_and_preprocess_data_for_flask(csv_file_path):
    print(f"DEBUG load_and_preprocess: Loading data from {csv_file_path}")
    try: data = pd.read_csv(csv_file_path)
    except Exception as e: return None, None, None, None, None, None, None, None, None, None, None, f"Error loading CSV: {str(e)}"

    if data.empty: return None, None, None, None, None, None, None, None, None, None, None, "CSV file is empty."
    if data.shape[1] < 2: return None, None, None, None, None, None, None, None, None, None, None, "CSV must have at least two columns."

    X_df = data.iloc[:, :-2]
    y_raw_series = data.iloc[:, -2]
    print(f"DEBUG load_and_preprocess: X_df shape {X_df.shape}, y_raw_series length {len(y_raw_series)}")


    if X_df.empty and data.shape[1] >= 1: # If only one col, it became y_raw, X_df is empty
        print("DEBUG load_and_preprocess: X_df is empty (likely single column CSV). Creating empty DataFrame for X.")
        X_df = pd.DataFrame(index=data.index) # Keep index for potential alignment

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
        for k_num, v_str in class_map_num_to_str.items(): # Find numeric label for this string name
            if v_str == class_name_str: numeric_label = k_num; break
        
        mask = (y_encoded_array == numeric_label)
        X_by_class_str_keys[class_name_str] = X_df[mask].copy() # Store as DataFrame copy
        y_by_class_str_keys[class_name_str] = y_encoded_array[mask].copy() # Store as NumPy array copy
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


# --- train_evaluate_individual_models_for_flask (with more logging) ---
def train_evaluate_individual_models_for_flask(X_full, y_encoded_full, preprocessor, class_map_numeric_to_str, unique_enc_labels, candidate_configs):
    print(f"DEBUG train_evaluate_models: X_full shape: {X_full.shape}, y_encoded_full length: {len(y_encoded_full)}")
    if X_full.empty and not any(cfg[1].__class__ == DummyClassifier for cfg in candidate_configs):
        print("Warning train_evaluate_models: X_full is empty. Only DummyClassifiers might train successfully.")
    if len(y_encoded_full) == 0:
        print("ERROR train_evaluate_models: y_encoded_full is empty. Cannot train.")
        raise ValueError("Cannot train models with an empty target variable (y_encoded_full).")
    
    min_samples_per_class = np.min(np.unique(y_encoded_full, return_counts=True)[1]) if len(y_encoded_full) > 0 else 0
    test_size_param = 0.3
    X_train, X_test, y_train, y_test = None, None, None, None # Initialize

    # Robust train/test split logic
    num_samples = X_full.shape[0]
    can_stratify = min_samples_per_class >= 2 and len(np.unique(y_encoded_full)) > 1
    
    if num_samples <= 1 or (num_samples * test_size_param < 1) or (num_samples * (1-test_size_param) < 1):
        print(f"DEBUG train_evaluate_models: Dataset too small for split (samples: {num_samples}). Using full data for train/test.")
        X_train, X_test, y_train, y_test = X_full, X_full, y_encoded_full, y_encoded_full
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_full, y_encoded_full, test_size=test_size_param, random_state=42,
                stratify=y_encoded_full if can_stratify else None
            )
            print(f"DEBUG train_evaluate_models: Split successful. X_train: {X_train.shape}, X_test: {X_test.shape}")
        except ValueError as e:
            print(f"DEBUG train_evaluate_models: Stratified split failed ({e}). Trying non-stratified.")
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_full, y_encoded_full, test_size=test_size_param, random_state=42, stratify=None
                )
                print(f"DEBUG train_evaluate_models: Non-stratified split successful. X_train: {X_train.shape}, X_test: {X_test.shape}")
            except Exception as e_split_again: # Should be rare if num_samples > 1
                print(f"ERROR train_evaluate_models: Even non-stratified split failed ({e_split_again}). Using full data for train/test.")
                X_train, X_test, y_train, y_test = X_full, X_full, y_encoded_full, y_encoded_full
    
    if X_train is None or y_train is None or X_test is None or y_test is None : # Should not happen if logic above is complete
        print("CRITICAL ERROR train_evaluate_models: Train/Test split resulted in None values. This should not happen.")
        # Fallback to prevent crash, though indicates deeper issue
        X_train, X_test, y_train, y_test = X_full, X_full, y_encoded_full, y_encoded_full


    trained_models_dict = {}
    model_summaries = []

    for name, clf_instance, cost_val in candidate_configs:
        print(f"DEBUG train_evaluate_models: Processing model '{name}'")
        model = Model(name, clf_instance, float(cost_val), preprocessor=preprocessor)
        try:
            if X_train.empty and not isinstance(clf_instance, DummyClassifier):
                print(f"Warning train_evaluate_models: Skipping training for '{name}', X_train is empty and not Dummy.")
                model.accuracy = 0.0 # Default
            else:
                model.train(X_train, y_train)
            
            if X_test.shape[0] > 0 and len(y_test) > 0:
                model.evaluate(X_test, y_test, labels=unique_enc_labels, class_names_map=class_map_numeric_to_str)
            else:
                print(f"Warning train_evaluate_models: Test set for model '{name}' is empty. Evaluation metrics will be default/zero.")
                # Ensure metrics are defaulted if evaluate isn't called or if it handles empty y_test
                model.accuracy = 0.0
                model.metrics_per_class = {cn_map.get(int(l), str(l)): {"precision":0.0, "recall":0.0, "f1-score":0.0, "support":0} for l, cn_map in zip(unique_enc_labels, [class_map_numeric_to_str]*len(unique_enc_labels))}
                model.avg_metrics = {"precision":0.0, "recall":0.0, "f1-score":0.0, "support":0}

            trained_models_dict[name] = model
            model_summaries.append({ "name": name, "accuracy": model.accuracy,
                                     "f1_score_weighted": model.avg_metrics.get('f1-score', 0.0),
                                     "cost": model.cost })
        except Exception as e_model_train:
            print(f"ERROR train_evaluate_models: Failed for model '{name}': {e_model_train}")
            print(traceback.format_exc())
    print(f"DEBUG train_evaluate_models: Finished. Returning X_test shape: {X_test.shape if X_test is not None else 'None'}, y_test length: {len(y_test) if y_test is not None else 'None'}")
    return trained_models_dict, model_summaries, X_test, y_test, y_train


# --- run_nomad_simulation_for_flask_streamed (passes data to NOMADEngine) ---
# (No new logging here, NOMADEngine.stream_evaluate_strategy has the detailed logs)
def run_nomad_simulation_for_flask_streamed(trained_models_dict, role_model_name, epsilon,
                                           initial_class_priors_str_keys, unique_class_names_ordered_str,
                                           class_map_numeric_to_str, quality_metric_for_ec, safety_check_type,
                                           X_test_data_overall, y_test_encoded_data_overall,
                                           candidate_configs, X_test_column_names,
                                           batch_size=10, adaptive_update_window=0, adaptive_beta=0.3,
                                           workload_phases=None, 
                                           X_by_class_str_keys=None, y_by_class_str_keys=None):
    print(f"DEBUG run_nomad_simulation: Initializing NOMADEngine. Role: {role_model_name}, Epsilon: {epsilon}")
    if workload_phases and len(workload_phases) > 0:
        print(f"DEBUG run_nomad_simulation: Workload phases provided: {len(workload_phases)} phases.")
    else:
        print(f"DEBUG run_nomad_simulation: No workload phases, using sequential X_test_data_overall (shape: {X_test_data_overall.shape if X_test_data_overall is not None else 'None'}).")

    nomad_engine = NOMADEngine(
        models_dict=trained_models_dict,
        role_model_name=role_model_name,
        epsilon=epsilon,
        initial_class_priors_str_keys=initial_class_priors_str_keys, # Corrected
        unique_classes_ordered_str=unique_class_names_ordered_str,   # Corrected
        class_map_numeric_to_str=class_map_numeric_to_str,           # Corrected
        quality_metric_for_ec=quality_metric_for_ec,
        safety_check_type=safety_check_type,
        adaptive_update_window=adaptive_update_window,
        adaptive_beta=adaptive_beta
    )
    candidate_model_names = [name for name, _, _ in candidate_configs]

    for update_package in nomad_engine.stream_evaluate_strategy(
        X_test_overall=X_test_data_overall,
        y_test_encoded_overall=y_test_encoded_data_overall,
        candidate_model_names_for_counts=candidate_model_names,
        X_test_column_names=X_test_column_names,
        batch_size=batch_size,
        workload_phases=workload_phases,
        X_by_class_str_keys=X_by_class_str_keys,
        y_by_class_str_keys=y_by_class_str_keys
    ):
        yield update_package