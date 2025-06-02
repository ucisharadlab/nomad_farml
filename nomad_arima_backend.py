import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

# --- Configuration ---
CANDIDATE_MODELS_CONFIG = [
    ("Dummy (Uniform)", DummyClassifier(strategy="uniform"), 0.1),
    ("Decision Tree (Shallow)", DecisionTreeClassifier(max_depth=3, random_state=42), 1.0),
    ("Gaussian Naive Bayes", GaussianNB(), 1.2),
    ("K-Nearest Neighbors (K=5)", KNeighborsClassifier(n_neighbors=5), 2.5),
    ("Decision Tree (Deeper)", DecisionTreeClassifier(max_depth=10, random_state=42), 12.0),
    ("Random Forest (Small)", RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42), 25.0),
    ("Random Forest (Large)", RandomForestClassifier(n_estimators=100, random_state=42), 35.0),
]

ALPHA_SMOOTHING = 1

# --- Helper Functions ---
def get_quality_metrics(y_true, y_pred, labels, class_names_map=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)
    p_r_f1_s = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    metrics_per_class = {}
    for i, label in enumerate(labels):
        class_name = class_names_map[int(label)] if class_names_map and int(label) in class_names_map else str(label)
        metrics_per_class[class_name] = {
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

# --- Model Class ---
class Model:
    def __init__(self, name, classifier_instance, cost, preprocessor):
        self.name = name
        self.cost = float(cost)
        self.classifier = classifier_instance
        self.cm = None
        self.accuracy = 0.0
        self.metrics_per_class = {}
        self.avg_metrics = {}
        self.exit_classes = set()
        self.label_encoder_classes_ = None
        self.class_names_map = None
        self.preprocessor = preprocessor
        self.trained_pipeline_ = None

    def train(self, X_train, y_train):
        pipeline_steps = []
        if self.preprocessor:
            pipeline_steps.append(('preprocessor', self.preprocessor))
        else:
            pipeline_steps.append(('scaler', StandardScaler()))
        pipeline_steps.append(('classifier', self.classifier))
        self.trained_pipeline_ = Pipeline(steps=pipeline_steps)
        self.trained_pipeline_.fit(X_train, y_train)

    def predict(self, X_test):
        return self.trained_pipeline_.predict(X_test)

    def predict_proba(self, X_test):
        if hasattr(self.trained_pipeline_.named_steps['classifier'], 'predict_proba'):
            return self.trained_pipeline_.predict_proba(X_test)
        else:
            predictions = self.predict(X_test)
            n_classes = len(self.label_encoder_classes_ if self.label_encoder_classes_ is not None else [])
            if n_classes == 0:
                 return np.array([])
            probas = np.zeros((X_test.shape[0], n_classes))
            class_indices = {label: i for i, label in enumerate(self.label_encoder_classes_)}
            for i, p_encoded in enumerate(predictions):
                if p_encoded in class_indices:
                    class_idx = class_indices[p_encoded]
                    probas[i, class_idx] = 1.0
            return probas

    def evaluate(self, X_test, y_test, labels, class_names_map):
        self.label_encoder_classes_ = labels
        self.class_names_map = class_names_map
        y_pred = self.predict(X_test)
        self.cm, self.accuracy, self.metrics_per_class, self.avg_metrics = get_quality_metrics(y_test, y_pred, labels, class_names_map)

    def get_quality(self, class_name, metric_type="f1-score"):
        if class_name not in self.metrics_per_class:
            return 0.0
        return float(self.metrics_per_class[class_name].get(metric_type, 0.0))

    def __repr__(self):
        return f"Model(name='{self.name}', cost={self.cost}, accuracy={self.accuracy:.4f})"

# --- NOMAD Engine Class ---
class NOMADEngine:
    def __init__(self, models_dict, role_model_name, epsilon,
                 initial_class_priors, unique_classes_ordered, class_names_map,
                 quality_metric_for_ec="f1-score", safety_check_type="conservative",
                 adaptive_update_window=0, adaptive_beta=0.3):
        self.models_dict = models_dict
        if role_model_name not in self.models_dict:
            raise ValueError(f"Role model '{role_model_name}' not found in provided models.")
        self.role_model = self.models_dict[role_model_name]
        self.epsilon = float(epsilon)
        self.initial_class_priors = initial_class_priors
        self.unique_classes_ordered = unique_classes_ordered
        self.class_names_map = class_names_map
        self.quality_metric_for_ec = quality_metric_for_ec
        self.safety_check_type = safety_check_type

        self.adaptive_update_window = int(adaptive_update_window)
        self.adaptive_beta = float(adaptive_beta)
        self.current_dynamic_class_priors = self.initial_class_priors.copy()
        self._event_count_since_last_prior_update = 0
        self._recent_prediction_counts_by_class_name = {cn: 0 for cn in self.unique_classes_ordered}

        self._determine_all_exit_classes()


    def _determine_all_exit_classes(self):
        for model_name, model_obj in self.models_dict.items():
            model_obj.exit_classes = set()
            for class_name_str in self.unique_classes_ordered:
                quality_model = model_obj.get_quality(class_name_str, self.quality_metric_for_ec)
                quality_role_model = self.role_model.get_quality(class_name_str, self.quality_metric_for_ec)
                if quality_role_model == 0:
                    if quality_model == 0:
                         model_obj.exit_classes.add(class_name_str)
                elif quality_model >= quality_role_model * (1 - self.epsilon):
                    model_obj.exit_classes.add(class_name_str)

            if model_obj == self.role_model:
                 model_obj.exit_classes = set(self.unique_classes_ordered)

    def _calculate_utility(self, model, current_class_priors_for_event):
        if model.cost == 0:
            return float('inf') if sum(current_class_priors_for_event[cn_str] for cn_str in model.exit_classes if cn_str in current_class_priors_for_event) > 0 else 0.0

        sum_prob_exit_classes = sum(current_class_priors_for_event[cn_str] for cn_str in model.exit_classes if cn_str in current_class_priors_for_event)
        return float(sum_prob_exit_classes / model.cost)

    def _update_probabilities(self, p_in, softmax_output):
        p_temp = {}
        current_sum = 0.0
        
        model_proba_class_labels_numeric = self.role_model.label_encoder_classes_

        if model_proba_class_labels_numeric is None or len(model_proba_class_labels_numeric) != len(softmax_output):
            print(f"Warning: Class order for softmax update is ambiguous. Using unique_classes_ordered as a fallback for mapping. Lengths: {len(model_proba_class_labels_numeric if model_proba_class_labels_numeric is not None else [])} vs {len(softmax_output)}")
            
            if self.unique_classes_ordered and len(self.unique_classes_ordered) == len(softmax_output):
                for i, class_name_str_ordered in enumerate(self.unique_classes_ordered):
                    p_temp[class_name_str_ordered] = p_in.get(class_name_str_ordered, 0.0) * softmax_output[i]
                    current_sum += p_temp[class_name_str_ordered]
            else:
                 pass 

        else:
            for i, numeric_label in enumerate(model_proba_class_labels_numeric):
                class_name_str = self.class_names_map.get(int(numeric_label))
                if class_name_str and class_name_str in p_in:
                    val = p_in[class_name_str] * softmax_output[i]
                    p_temp[class_name_str] = val
                    current_sum += val
        
        p_out = {}
        if current_sum > 1e-9:
            for class_name_str_key in self.unique_classes_ordered:
                p_out[class_name_str_key] = float(p_temp.get(class_name_str_key, 0.0) / current_sum)
        else:
            num_classes = len(self.unique_classes_ordered)
            uniform_prob = 1.0 / num_classes if num_classes > 0 else 0.0
            for class_name_str_key in self.unique_classes_ordered:
                p_out[class_name_str_key] = uniform_prob
        return p_out


    def _check_chain_safety_conservative(self, M_new_name, S_current_names):
        S_potential_names = S_current_names + [M_new_name]
        for class_name_target_str in self.unique_classes_ordered:
            M_exit_for_class_j_name = None
            M_exit_for_class_j_idx = -1
            S_final_for_this_class_check = S_potential_names

            for idx, model_name_in_chain in enumerate(S_potential_names):
                model_in_chain = self.models_dict[model_name_in_chain]
                if class_name_target_str in model_in_chain.exit_classes:
                    M_exit_for_class_j_name = model_name_in_chain
                    M_exit_for_class_j_idx = idx
                    break
            
            if M_exit_for_class_j_name is None:
                S_final_for_this_class_check = S_potential_names + ([self.role_model.name] if self.role_model.name not in S_potential_names else [])
                
                M_exit_for_class_j_name = self.role_model.name
                if M_exit_for_class_j_name not in S_final_for_this_class_check:
                    return False
                M_exit_for_class_j_idx = S_final_for_this_class_check.index(M_exit_for_class_j_name)


            chain_recall_j = 1.0
            for k_idx in range(M_exit_for_class_j_idx + 1):
                model_k_name = S_final_for_this_class_check[k_idx]
                model_k_obj = self.models_dict[model_k_name]
                recall_mk_cj = model_k_obj.get_quality(class_name_target_str, "recall")
                chain_recall_j *= recall_mk_cj
            
            recall_role_model_cj = self.role_model.get_quality(class_name_target_str, "recall")
            quality_bound = (1.0 - self.epsilon) * recall_role_model_cj
            if chain_recall_j < quality_bound:
                return False
        return True

    def _get_smoothed_misclassification_prob(self, model_k, class_j_true_str, class_i_pred_str):
        true_label_numeric = -1
        for num_label, name_str in model_k.class_names_map.items():
            if name_str == class_j_true_str:
                true_label_numeric = int(num_label)
                break
        
        if true_label_numeric == -1 or \
           model_k.label_encoder_classes_ is None or \
           true_label_numeric not in model_k.label_encoder_classes_:
            return 0.0

        try:
            cm_row_idx = list(model_k.label_encoder_classes_).index(true_label_numeric)
        except ValueError:
             return 0.0

        pred_label_numeric = -1
        for num_label, name_str in model_k.class_names_map.items():
            if name_str == class_i_pred_str:
                pred_label_numeric = int(num_label)
                break
        
        n_ji = 0
        if pred_label_numeric != -1 and \
           model_k.label_encoder_classes_ is not None and \
           pred_label_numeric in model_k.label_encoder_classes_:
            try:
                cm_col_idx = list(model_k.label_encoder_classes_).index(pred_label_numeric)
                if model_k.cm is not None and cm_row_idx < model_k.cm.shape[0] and cm_col_idx < model_k.cm.shape[1]:
                     n_ji = int(model_k.cm[cm_row_idx, cm_col_idx])
            except ValueError:
                 pass
        
        sum_n_jl = 0
        if model_k.cm is not None and cm_row_idx < model_k.cm.shape[0]:
            sum_n_jl = int(np.sum(model_k.cm[cm_row_idx, :]))

        alpha_i_val = ALPHA_SMOOTHING
        S_total_pseudo_counts = ALPHA_SMOOTHING * len(self.unique_classes_ordered)
        
        if (sum_n_jl + S_total_pseudo_counts) == 0: return 0.0
        
        return float(n_ji + alpha_i_val) / (sum_n_jl + S_total_pseudo_counts)

    def _check_chain_safety_relaxed(self, M_new_name, S_current_names):
        S_potential_names = S_current_names + [M_new_name]
        for class_j_true_str in self.unique_classes_ordered:
            M_exit_for_class_j_name = None
            M_exit_for_class_j_idx = -1
            S_final_for_this_class_check = S_potential_names

            for idx, model_name_in_chain in enumerate(S_potential_names):
                model_in_chain = self.models_dict[model_name_in_chain]
                if class_j_true_str in model_in_chain.exit_classes:
                    M_exit_for_class_j_name = model_name_in_chain
                    M_exit_for_class_j_idx = idx
                    break
            
            if M_exit_for_class_j_name is None:
                S_final_for_this_class_check = S_potential_names + ([self.role_model.name] if self.role_model.name not in S_potential_names else [])
                M_exit_for_class_j_name = self.role_model.name
                if M_exit_for_class_j_name not in S_final_for_this_class_check: return False
                M_exit_for_class_j_idx = S_final_for_this_class_check.index(M_exit_for_class_j_name)

            M_exit_model_obj = self.models_dict[M_exit_for_class_j_name]
            
            prod_pass_through_refined = 1.0
            for k_idx in range(M_exit_for_class_j_idx):
                model_k_name = S_final_for_this_class_check[k_idx]
                model_k_obj = self.models_dict[model_k_name]
                
                prob_k_misclassify_to_its_exit_class = 0.0
                for class_i_pred_str in model_k_obj.exit_classes:
                    prob_k_misclassify_to_its_exit_class += self._get_smoothed_misclassification_prob(
                        model_k_obj, class_j_true_str, class_i_pred_str
                    )
                
                pass_through_k_refined = 1.0 - prob_k_misclassify_to_its_exit_class
                prod_pass_through_refined *= pass_through_k_refined
            
            chain_quality_j_refined = prod_pass_through_refined * M_exit_model_obj.get_quality(class_j_true_str, "recall")
            
            recall_role_model_cj = self.role_model.get_quality(class_j_true_str, "recall")
            quality_bound = (1.0 - self.epsilon) * recall_role_model_cj
            if chain_quality_j_refined < quality_bound:
                return False
        return True

    def select_and_classify_event(self, event_features_df):
        current_class_priors_for_this_event = self.current_dynamic_class_priors.copy()
        models_available_dict = self.models_dict.copy()
        realized_chain_names = []
        current_event_cost = 0.0
        final_prediction_name = None

        while models_available_dict:
            utility_candidates = []
            for model_name, model_obj in models_available_dict.items():
                utility_candidates.append(
                    (self._calculate_utility(model_obj, current_class_priors_for_this_event),
                     -model_obj.cost,
                     model_name)
                )
            if not utility_candidates: break
            utility_candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
            
            selected_model_name_by_utility = utility_candidates[0][2]
            M_star = self.models_dict[selected_model_name_by_utility]
            
            is_safe = False
            if self.safety_check_type == "conservative":
                is_safe = self._check_chain_safety_conservative(M_star.name, realized_chain_names)
            elif self.safety_check_type == "relaxed":
                 is_safe = self._check_chain_safety_relaxed(M_star.name, realized_chain_names)

            if not is_safe:
                del models_available_dict[M_star.name]
                if M_star == self.role_model:
                    break 
                continue

            pred_current_encoded = M_star.predict(event_features_df)[0]
            pred_current_name = self.class_names_map.get(int(pred_current_encoded), "UnknownPrediction")
            
            softmax_vector = M_star.predict_proba(event_features_df)[0]
            current_event_cost += M_star.cost
            realized_chain_names.append(M_star.name)
            final_prediction_name = pred_current_name

            if pred_current_name in M_star.exit_classes:
                return final_prediction_name, current_event_cost, realized_chain_names

            del models_available_dict[M_star.name]
            current_class_priors_for_this_event = self._update_probabilities(
                current_class_priors_for_this_event, softmax_vector
            )
            
            if M_star == self.role_model:
                break

        if self.role_model.name not in realized_chain_names:
            current_event_cost += self.role_model.cost
            realized_chain_names.append(self.role_model.name)
            pred_final_encoded = self.role_model.predict(event_features_df)[0]
            final_prediction_name = self.class_names_map.get(int(pred_final_encoded), "UnknownRolePrediction")
        
        if final_prediction_name is None:
            if self.role_model.name not in realized_chain_names:
                 current_event_cost += self.role_model.cost
                 realized_chain_names.append(self.role_model.name)
            pred_final_encoded = self.role_model.predict(event_features_df)[0]
            final_prediction_name = self.class_names_map.get(int(pred_final_encoded), "DefaultRolePredictionOnError")


        return final_prediction_name, current_event_cost, realized_chain_names

    def stream_evaluate_strategy(self, X_test, y_test_encoded, candidate_model_names_for_counts, X_test_column_names, batch_size=10):
        nomad_predictions_encoded_list = []
        true_labels_so_far = []
        
        total_cost = 0.0
        model_run_counts_live = {name: 0 for name in candidate_model_names_for_counts}
        total_events = X_test.shape[0]

        self.current_dynamic_class_priors = self.initial_class_priors.copy()
        self._event_count_since_last_prior_update = 0
        self._recent_prediction_counts_by_class_name = {cn: 0 for cn in self.unique_classes_ordered}


        yield {
            "type": "setup",
            "total_events": total_events,
            "role_model_static_accuracy": float(self.role_model.accuracy),
            "role_model_cost_per_event": float(self.role_model.cost),
            "exit_classes_info": {name: list(model.exit_classes) for name, model in self.models_dict.items()},
            "class_names_ordered": self.unique_classes_ordered,
            "initial_priors": {k: float(v) for k,v in self.current_dynamic_class_priors.items()}
        }

        for i in range(total_events):
            if isinstance(X_test, pd.DataFrame):
                event_features_row = X_test.iloc[i]
                event_features_df = event_features_row.to_frame().T
            else:
                event_features_df = pd.DataFrame([X_test[i,:]], columns=X_test_column_names)
            
            pred_name_str, cost_for_event, chain_for_event = self.select_and_classify_event(event_features_df)
            
            pred_encoded = -1
            for enc_label_num, name_str_map in self.class_names_map.items():
                if name_str_map == pred_name_str:
                    pred_encoded = int(enc_label_num)
                    break
            if pred_encoded == -1 and self.role_model.label_encoder_classes_ is not None and len(self.role_model.label_encoder_classes_) > 0:
                 pred_encoded = int(self.role_model.label_encoder_classes_[0])


            nomad_predictions_encoded_list.append(pred_encoded)
            true_labels_so_far.append(y_test_encoded[i])
            
            total_cost += cost_for_event
            for model_name_in_chain in chain_for_event:
                if model_name_in_chain in model_run_counts_live:
                    model_run_counts_live[model_name_in_chain] += 1
            
            if self.adaptive_update_window > 0:
                if pred_name_str in self._recent_prediction_counts_by_class_name:
                    self._recent_prediction_counts_by_class_name[pred_name_str] += 1
                self._event_count_since_last_prior_update += 1

                if self._event_count_since_last_prior_update >= self.adaptive_update_window:
                    new_dynamic_priors_update = {}
                    total_observed_in_window = sum(self._recent_prediction_counts_by_class_name.values())

                    if total_observed_in_window > 0:
                        for class_name_str_key in self.unique_classes_ordered:
                            observed_prob_ci = float(self._recent_prediction_counts_by_class_name.get(class_name_str_key, 0)) / total_observed_in_window
                            old_prior_ci = self.current_dynamic_class_priors.get(class_name_str_key, 0.0)
                            
                            new_prior_val = (self.adaptive_beta * observed_prob_ci) + \
                                           ((1.0 - self.adaptive_beta) * old_prior_ci)
                            new_dynamic_priors_update[class_name_str_key] = new_prior_val
                        
                        current_sum_new_priors = sum(new_dynamic_priors_update.values())
                        if current_sum_new_priors > 1e-9:
                            for class_name_str_key_norm in new_dynamic_priors_update:
                                new_dynamic_priors_update[class_name_str_key_norm] /= current_sum_new_priors
                        else:
                            num_classes_norm = len(self.unique_classes_ordered)
                            uniform_prob_norm = 1.0 / num_classes_norm if num_classes_norm > 0 else 0.0
                            for class_name_str_key_norm_unif in self.unique_classes_ordered:
                                new_dynamic_priors_update[class_name_str_key_norm_unif] = uniform_prob_norm
                                
                        self.current_dynamic_class_priors = new_dynamic_priors_update
                        
                        yield {
                            "type": "prior_update",
                            "event_number": i + 1,
                            "updated_priors": {k: float(v) for k, v in self.current_dynamic_class_priors.items()}
                        }

                    self._recent_prediction_counts_by_class_name = {cn: 0 for cn in self.unique_classes_ordered}
                    self._event_count_since_last_prior_update = 0
            
            if (i + 1) % batch_size == 0 or (i + 1) == total_events:
                live_nomad_accuracy = 0.0
                if true_labels_so_far:
                    live_nomad_accuracy = float(accuracy_score(true_labels_so_far, nomad_predictions_encoded_list))
                
                yield {
                    "type": "batch_update",
                    "last_event_in_batch": i + 1,
                    "cumulative_cost": float(total_cost),
                    "model_run_counts_snapshot": {k: int(v) for k, v in model_run_counts_live.items()},
                    "live_nomad_accuracy": live_nomad_accuracy,
                }
        
        avg_cost = float(total_cost / total_events) if total_events > 0 else 0.0
        eval_labels = self.role_model.label_encoder_classes_ if self.role_model.label_encoder_classes_ is not None else [] 
        # Ensure eval_labels is not None and is a list of numeric labels for get_quality_metrics
        if not isinstance(eval_labels, list) and eval_labels is not None: # if it's numpy array for instance
            eval_labels = list(eval_labels)
        elif eval_labels is None: # Fallback if role model somehow doesn't have it (should not happen post-eval)
            # Try to get from any model, or default to sorted unique encoded labels seen in data.
            # This is a safeguard.
            if self.class_names_map: # Map is numeric -> string
                eval_labels = sorted(list(self.class_names_map.keys()))
            else: # Absolute fallback, might not be correct order for all models
                # This case indicates a more significant setup issue.
                eval_labels = []


        final_cm_nomad, final_acc_nomad, final_metrics_per_class_nomad, final_avg_metrics_nomad = get_quality_metrics(
            y_test_encoded, nomad_predictions_encoded_list,
            labels=eval_labels,
            class_names_map=self.class_names_map
        )

        yield {
            "type": "summary",
            "nomad_overall_metrics": {
                "accuracy": float(final_acc_nomad), "avg_metrics": final_avg_metrics_nomad,
                "average_cost": avg_cost, "cm": final_cm_nomad.tolist() if final_cm_nomad is not None else [],
            },
            "nomad_per_class_metrics": final_metrics_per_class_nomad,
            "role_model_performance": {
                "name": self.role_model.name, "accuracy": float(self.role_model.accuracy),
                "avg_metrics": self.role_model.avg_metrics, "cost": float(self.role_model.cost),
                "cm": self.role_model.cm.tolist() if self.role_model.cm is not None else [],
                "per_class_metrics": self.role_model.metrics_per_class
            },
            "final_model_run_counts": {k: int(v) for k, v in model_run_counts_live.items()},
        }

# --- Refactored Functions for Flask ---
def load_and_preprocess_data_for_flask(csv_file_path):
    try:
        data = pd.read_csv(csv_file_path)
    except Exception as e:
        return None, None, None, None, None, None, None, None, None, f"Error loading CSV: {str(e)}"

    if data.empty:
        return None, None, None, None, None, None, None, None, None, "CSV file is empty."
    if data.shape[1] < 2:
        return None, None, None, None, None, None, None, None, None, "CSV must have at least two columns (features and target)."

    # Simplified assumption: last column is target, rest are features.
    X = data.iloc[:, :-2]
    y_raw = data.iloc[:, -2]

    if X.empty: # Only one column was present, interpreted as y_raw. No features.
        # This scenario might be valid if the models can handle no features (e.g., DummyClassifier)
        # or if at least one feature column is expected.
        # For now, let's allow it but ColumnTransformer might complain if numerical_cols and categorical_cols are both empty.
        print("Warning: No feature columns found. X is empty. Target variable is the single column from CSV.")
        # Ensure X is an empty DataFrame with original index if it's going to be used.
        X = pd.DataFrame(index=data.index)


    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    unique_labels_encoded = np.sort(np.unique(y_encoded))
    
    class_names_map = {int(enc): str(name) for enc, name in zip(unique_labels_encoded, label_encoder.inverse_transform(unique_labels_encoded))}
    unique_class_names_ordered = [class_names_map[int(el)] for el in unique_labels_encoded]

    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Handle case where X might be empty after selection (e.g. all columns were target)
    # The ColumnTransformer needs at least one transformer.
    # If both numerical_cols and categorical_cols are empty, it means X was empty or had no identifiable num/cat columns.
    
    transformers_list = []
    if numerical_cols:
        transformers_list.append(('num', StandardScaler(), numerical_cols))
    if categorical_cols:
        transformers_list.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols))
    
    if not transformers_list and not X.empty: 
        # X has columns, but none are num/cat. They will be passed through if remainder='passthrough'.
        # Or, they could be of other types not handled.
        # If X is truly empty (no columns), then preprocessor might not be meaningful.
        print("Warning: No numerical or categorical columns identified for preprocessing. Columns might be passed through or ignored depending on 'remainder'.")

    # If X is empty, preprocessor might not be useful, but models might still work (e.g. Dummy)
    # The preprocessor object itself is fine, but fitting it on empty X might be an issue for some transformers.
    # Pipeline handles empty X passed to preprocessor correctly if it's designed for it (e.g. passthrough)

    preprocessor_for_models = ColumnTransformer(
        transformers=transformers_list, # Use dynamic list
        remainder='passthrough' 
    )
    
    return data, X, y_encoded, label_encoder, unique_labels_encoded, \
           class_names_map, unique_class_names_ordered, preprocessor_for_models, X.columns.tolist(), None


def train_evaluate_individual_models_for_flask(X_full, y_encoded_full, preprocessor, class_map_numeric_to_str, unique_enc_labels, candidate_configs):
    if X_full.empty and not any(cfg[1].__class__ == DummyClassifier for cfg in candidate_configs):
        # If X is empty and we are not just training DummyClassifiers, it's problematic
        # For DummyClassifier, empty X might be acceptable depending on strategy.
        print("Warning: X_full is empty. Only DummyClassifiers might train successfully.")
        # Depending on strictness, could raise ValueError here.

    if len(y_encoded_full) == 0:
        raise ValueError("Cannot train models with an empty target variable (y_encoded_full).")
    
    min_samples_per_class = np.min(np.unique(y_encoded_full, return_counts=True)[1]) if len(y_encoded_full) > 0 else 0
    
    test_size_param = 0.3
    # Ensure there are enough samples for a split.
    # If only 1 sample, train=1, test=0 is not possible with test_size > 0.
    # If n_samples=1, train_test_split raises error.
    # If n_samples <= number of classes for stratification, it can also fail.

    if X_full.shape[0] <= 1 or (min_samples_per_class < 2 and len(np.unique(y_encoded_full)) > 1 and X_full.shape[0] < 5) : # Simplified condition for "too small to split well"
        print(f"Warning: Dataset too small or class distribution too skewed for a meaningful train/test split (Total samples: {X_full.shape[0]}, Min samples per class: {min_samples_per_class}). Using full dataset for both training and testing. Metrics will be optimistic.")
        X_train, X_test, y_train, y_test = X_full, X_full, y_encoded_full, y_encoded_full
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_encoded_full, test_size=test_size_param, random_state=42,
            stratify=y_encoded_full if min_samples_per_class >= 2 and len(np.unique(y_encoded_full)) > 1 else None
        )

    trained_models_dict = {}
    model_summaries = []

    for name, clf_instance, cost_val in candidate_configs:
        model = Model(name, clf_instance, float(cost_val), preprocessor=preprocessor)
        try:
            # Check if X_train is empty. Some models (like Dummy) might handle it.
            if X_train.empty and not isinstance(clf_instance, DummyClassifier):
                print(f"Skipping training for {name} as X_train is empty and it's not a DummyClassifier.")
                # Populate with default/error values if needed, or skip adding to dicts
                model.accuracy = 0.0 
                # ... set other metrics to defaults ...
            else:
                model.train(X_train, y_train)
            
            if X_test.shape[0] > 0 and len(y_test) > 0: # Ensure y_test is also not empty
                model.evaluate(X_test, y_test, labels=unique_enc_labels, class_names_map=class_map_numeric_to_str)
            else:
                print(f"Warning: Test set for model {name} is empty or y_test is empty after split. Skipping evaluation.")
            
            trained_models_dict[name] = model
            model_summaries.append({
                "name": name, "accuracy": model.accuracy,
                "f1_score_weighted": model.avg_metrics.get('f1-score', 0.0),
                "cost": model.cost
            })
        except Exception as e:
            print(f"Error training or evaluating {name}: {e}")
            import traceback
            traceback.print_exc()
    return trained_models_dict, model_summaries, X_test, y_test, y_train


def run_nomad_simulation_for_flask_streamed(trained_models_dict, role_model_name, epsilon,
                                           initial_class_priors_str_keys, unique_class_names_ordered_str,
                                           class_map_numeric_to_str, quality_metric_for_ec, safety_check_type,
                                           X_test_data, y_test_encoded_data, candidate_configs,
                                           X_test_column_names, batch_size=10,
                                           adaptive_update_window=0, adaptive_beta=0.3):
    nomad_engine = NOMADEngine(
        models_dict=trained_models_dict, role_model_name=role_model_name, epsilon=epsilon,
        initial_class_priors=initial_class_priors_str_keys,
        unique_classes_ordered=unique_class_names_ordered_str,
        class_names_map=class_map_numeric_to_str,
        quality_metric_for_ec=quality_metric_for_ec,
        safety_check_type=safety_check_type,
        adaptive_update_window=adaptive_update_window,
        adaptive_beta=adaptive_beta
    )
    candidate_model_names = [name for name, _, _ in candidate_configs]
    for update_package in nomad_engine.stream_evaluate_strategy(
        X_test_data, y_test_encoded_data, candidate_model_names, X_test_column_names, batch_size=batch_size
    ):
        yield update_package