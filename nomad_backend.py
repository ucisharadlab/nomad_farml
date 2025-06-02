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
                 quality_metric_for_ec="f1-score", safety_check_type="conservative"):
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

    def _calculate_utility(self, model, current_class_priors): 
        if model.cost == 0:
            return float('inf') if sum(current_class_priors[cn_str] for cn_str in model.exit_classes if cn_str in current_class_priors) > 0 else 0.0
        sum_prob_exit_classes = sum(current_class_priors[cn_str] for cn_str in model.exit_classes if cn_str in current_class_priors)
        return float(sum_prob_exit_classes / model.cost)

    def _update_probabilities(self, p_in, softmax_output): 
        p_temp = {}
        current_sum = 0.0
        
        # Attempt to get the classifier's classes_ attribute for order
        # This relies on the model having been trained and having a pipeline structure
        # We need a reliable way to get the class order corresponding to softmax_output columns
        # For simplicity, let's assume the role model (or any evaluated model) has label_encoder_classes_ set
        model_proba_class_labels = None
        if self.role_model and self.role_model.label_encoder_classes_ is not None:
            model_proba_class_labels = self.role_model.label_encoder_classes_
        else: # Fallback: try to get from a generic model if role model not setup yet or lacks labels
            for model_val in self.models_dict.values():
                if model_val.label_encoder_classes_ is not None:
                    model_proba_class_labels = model_val.label_encoder_classes_
                    break
        
        if model_proba_class_labels is None or len(model_proba_class_labels) != len(softmax_output):
             # Fallback to unique_classes_ordered if specific model order is not determined. This is risky.
             # Or better, raise an error or log a warning.
             # For now, let's try to be robust if this occurs, but it indicates a setup issue.
            print("Warning: Could not determine class order for softmax probability update. Results may be incorrect.")
            pass


        for i, numeric_label in enumerate(model_proba_class_labels if model_proba_class_labels is not None else []): # Use the model's class order
            class_name_str = self.class_names_map.get(numeric_label) 
            if class_name_str and class_name_str in p_in: # If this numeric label maps to a known string class name
                if i < len(softmax_output): # Boundary check
                    val = p_in[class_name_str] * softmax_output[i]
                    p_temp[class_name_str] = val
                    current_sum += val
                else:
                    print(f"Warning: Index {i} out of bounds for softmax_output in _update_probabilities.")
            # else:
                # print(f"Warning: class_name_str '{class_name_str}' from numeric_label '{numeric_label}' not in p_in or not mapped.")
        
        p_out = {}
        if current_sum > 1e-9:
            for class_name_str in self.unique_classes_ordered: 
                p_out[class_name_str] = float(p_temp.get(class_name_str, 0.0) / current_sum)
        else: # Fallback to uniform if sum is too small (e.g., all probabilities became zero)
            num_classes = len(self.unique_classes_ordered)
            uniform_prob = 1.0 / num_classes if num_classes > 0 else 0.0
            for class_name_str in self.unique_classes_ordered:
                p_out[class_name_str] = uniform_prob
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
                if M_exit_for_class_j_name not in S_final_for_this_class_check: # Should not happen if logic above is correct
                    return False # Error condition: role model not found in effective chain
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
        for num_label, name_str in model_k.class_names_map.items(): # model_k.class_names_map: numeric -> string
            if name_str == class_j_true_str:
                true_label_numeric = int(num_label)
                break
        if true_label_numeric == -1 or model_k.label_encoder_classes_ is None or true_label_numeric not in model_k.label_encoder_classes_: 
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
        if pred_label_numeric != -1 and model_k.label_encoder_classes_ is not None and pred_label_numeric in model_k.label_encoder_classes_:
            try:
                cm_col_idx = list(model_k.label_encoder_classes_).index(pred_label_numeric)
                if model_k.cm is not None and cm_row_idx < model_k.cm.shape[0] and cm_col_idx < model_k.cm.shape[1]:
                     n_ji = int(model_k.cm[cm_row_idx, cm_col_idx])
            except ValueError:
                 pass 
        
        sum_n_jl = 0
        if model_k.cm is not None and cm_row_idx < model_k.cm.shape[0]:
            sum_n_jl = int(np.sum(model_k.cm[cm_row_idx, :]))

        alpha_i_val = ALPHA_SMOOTHING if class_i_pred_str in self.unique_classes_ordered else 0
        S_total_pseudo_counts = ALPHA_SMOOTHING * len(self.unique_classes_ordered) 
        
        if sum_n_jl + S_total_pseudo_counts == 0: return 0.0
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
                prob_exit_misclassify_smoothed = 0.0
                for class_i_pred_str in model_k_obj.exit_classes: 
                    prob_exit_misclassify_smoothed += self._get_smoothed_misclassification_prob(
                        model_k_obj, class_j_true_str, class_i_pred_str
                    )
                pass_through_k_refined = 1.0 - prob_exit_misclassify_smoothed
                prod_pass_through_refined *= pass_through_k_refined
            
            chain_quality_j_refined = prod_pass_through_refined * M_exit_model_obj.get_quality(class_j_true_str, "recall")
            recall_role_model_cj = self.role_model.get_quality(class_j_true_str, "recall")
            quality_bound = (1.0 - self.epsilon) * recall_role_model_cj
            if chain_quality_j_refined < quality_bound:
                return False
        return True

    def select_and_classify_event(self, event_features_df): 
        current_class_priors = self.initial_class_priors.copy()
        models_available_dict = self.models_dict.copy()
        realized_chain_names = []
        current_event_cost = 0.0
        final_prediction_name = None 

        while models_available_dict:
            utility_candidates = []
            for model_name, model_obj in models_available_dict.items():
                utility_candidates.append(
                    (self._calculate_utility(model_obj, current_class_priors), -model_obj.cost, model_name)
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
                if M_star == self.role_model: break 
                continue

            pred_current_encoded = M_star.predict(event_features_df)[0] 
            pred_current_name = self.class_names_map.get(pred_current_encoded, "UnknownPrediction") 
            
            softmax_vector = M_star.predict_proba(event_features_df)[0] 
            current_event_cost += M_star.cost
            realized_chain_names.append(M_star.name)
            final_prediction_name = pred_current_name

            if pred_current_name in M_star.exit_classes: 
                return final_prediction_name, current_event_cost, realized_chain_names
            
            del models_available_dict[M_star.name]
            current_class_priors = self._update_probabilities(current_class_priors, softmax_vector) # Pass M_star if needed by _update_probabilities
            if M_star == self.role_model: break

        if self.role_model.name not in realized_chain_names: 
            current_event_cost += self.role_model.cost
            realized_chain_names.append(self.role_model.name)
            
        pred_final_encoded = self.role_model.predict(event_features_df)[0]
        final_prediction_name = self.class_names_map.get(pred_final_encoded, "UnknownRolePrediction")
        
        return final_prediction_name, current_event_cost, realized_chain_names

    def stream_evaluate_strategy(self, X_test, y_test_encoded, candidate_model_names_for_counts, X_test_column_names, batch_size=10): # Added batch_size
        nomad_predictions_encoded_list = []
        true_labels_so_far = [] # For live accuracy calculation
        
        total_cost = 0.0
        model_run_counts_live = {name: 0 for name in candidate_model_names_for_counts}
        total_events = X_test.shape[0]

        yield {
            "type": "setup", 
            "total_events": total_events,
            "role_model_static_accuracy": float(self.role_model.accuracy),
            "role_model_cost_per_event": float(self.role_model.cost),
            "exit_classes_info": {name: list(model.exit_classes) for name, model in self.models_dict.items()},
            "class_names_ordered": self.unique_classes_ordered 
        }

        for i in range(total_events):
            # Construct DataFrame for single row correctly
            if isinstance(X_test, pd.DataFrame):
                event_features_row = X_test.iloc[i]
                event_features_df = event_features_row.to_frame().T
            else: # Assuming X_test is NumPy array
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
            
            # Yield update after each batch or if it's the last event
            if (i + 1) % batch_size == 0 or (i + 1) == total_events:
                live_nomad_accuracy = 0.0
                if true_labels_so_far: # ensure not empty
                    live_nomad_accuracy = float(accuracy_score(true_labels_so_far, nomad_predictions_encoded_list))
                
                yield {
                    "type": "batch_update", 
                    "last_event_in_batch": i + 1, # The event number that triggered this batch update
                    "cumulative_cost": float(total_cost), 
                    "model_run_counts_snapshot": {k: int(v) for k, v in model_run_counts_live.items()},
                    "live_nomad_accuracy": live_nomad_accuracy,
                }
        
        # Final summary after all batches/events
        avg_cost = float(total_cost / total_events) if total_events > 0 else 0.0
        final_cm_nomad, final_acc_nomad, final_metrics_per_class_nomad, final_avg_metrics_nomad = get_quality_metrics(
            y_test_encoded, nomad_predictions_encoded_list,
            labels=self.role_model.label_encoder_classes_, 
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

    if data.shape[1] < 2:
        return None, None, None, None, None, None, None, None, None, "CSV must have at least two columns."

    X = data.iloc[:, :-2]
    y_raw = data.iloc[:, -2]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    unique_labels_encoded = np.sort(np.unique(y_encoded))
    
    class_names_map = {int(enc): str(name) for enc, name in zip(unique_labels_encoded, label_encoder.inverse_transform(unique_labels_encoded))}
    unique_class_names_ordered = [class_names_map[int(el)] for el in unique_labels_encoded] 

    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor_for_models = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='passthrough'
    )
    return data, X, y_encoded, label_encoder, unique_labels_encoded, \
           class_names_map, unique_class_names_ordered, preprocessor_for_models, X.columns.tolist(), None

def train_evaluate_individual_models_for_flask(X_full, y_encoded_full, preprocessor, class_map_numeric_to_str, unique_enc_labels, candidate_configs):
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_encoded_full, test_size=0.3, random_state=42, stratify=y_encoded_full)
    trained_models_dict = {}
    model_summaries = []

    for name, clf_instance, cost_val in candidate_configs:
        model = Model(name, clf_instance, float(cost_val), preprocessor=preprocessor)
        try:
            model.train(X_train, y_train)
            model.evaluate(X_test, y_test, labels=unique_enc_labels, class_names_map=class_map_numeric_to_str)
            trained_models_dict[name] = model
            model_summaries.append({
                "name": name, "accuracy": model.accuracy,
                "f1_score_weighted": model.avg_metrics.get('f1-score', 0.0),
                "cost": model.cost
            })
        except Exception as e:
            print(f"Error training {name}: {e}")
            import traceback
            traceback.print_exc()
    return trained_models_dict, model_summaries, X_test, y_test, y_train

def run_nomad_simulation_for_flask_streamed(trained_models_dict, role_model_name, epsilon,
                                           initial_class_priors_str_keys, unique_class_names_ordered_str,
                                           class_map_numeric_to_str, quality_metric_for_ec, safety_check_type,
                                           X_test_data, y_test_encoded_data, candidate_configs, 
                                           X_test_column_names, batch_size=10): # Added batch_size parameter
    nomad_engine = NOMADEngine(
        models_dict=trained_models_dict, role_model_name=role_model_name, epsilon=epsilon,
        initial_class_priors=initial_class_priors_str_keys, 
        unique_classes_ordered=unique_class_names_ordered_str, 
        class_names_map=class_map_numeric_to_str, 
        quality_metric_for_ec=quality_metric_for_ec,
        safety_check_type=safety_check_type
    )
    candidate_model_names = [name for name, _, _ in candidate_configs]
    # Pass batch_size to the engine's streaming method
    for update_package in nomad_engine.stream_evaluate_strategy(
        X_test_data, y_test_encoded_data, candidate_model_names, X_test_column_names, batch_size=batch_size
    ):
        yield update_package