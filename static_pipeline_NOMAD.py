import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, classification_report

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC # SVC can be slow for large datasets, ensure probability=True
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier


# --- Configuration ---
# Define a list of models to train. Each entry is a tuple:
# (name, scikit-learn_classifier_instance, associated_cost)
# Costs are hypothetical; in a real scenario, measure inference time.
CANDIDATE_MODELS_CONFIG = [
    ("Dummy (Uniform)", DummyClassifier(strategy="uniform"), 0.1), # Very low cost, baseline
    ("Decision Tree (Shallow)", DecisionTreeClassifier(max_depth=3, random_state=42), 1.0),
    ("Gaussian Naive Bayes", GaussianNB(), 1.2),
    #("Logistic Regression", LogisticRegression(solver='liblinear', random_state=42, multi_class='auto'), 2.0),
    ("K-Nearest Neighbors (K=5)", KNeighborsClassifier(n_neighbors=5), 2.5),
    ("Decision Tree (Deeper)", DecisionTreeClassifier(max_depth=10, random_state=42), 12.0),
    ("Random Forest (Small)", RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42), 25.0),
    #("Gradient Boosting (Small)", GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42), 30.0),
    # Add a more complex/expensive model if desired, e.g., a larger RF or SVC
    # ("SVC (Linear Kernel)", SVC(kernel='linear', probability=True, random_state=42), 10.0), # Can be slow
    ("Random Forest (Large)", RandomForestClassifier(n_estimators=100, random_state=42), 35.0),
]

# For relaxed chain safety smoothing
ALPHA_SMOOTHING = 1 # Add-1 smoothing (Laplace)

# --- Helper Functions ---

def get_quality_metrics(y_true, y_pred, labels, class_names_map=None):
    """
    Calculates confusion matrix, overall accuracy, and per-class metrics.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)
    
    # precision, recall, f1, support per class
    # zero_division=0 means if a class has no true samples or no predicted samples for precision/recall,
    # the metric will be 0 for that class instead of raising a warning.
    p_r_f1_s = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    
    metrics_per_class = {}
    for i, label in enumerate(labels):
        class_name = class_names_map[label] if class_names_map else str(label)
        metrics_per_class[class_name] = {
            "precision": p_r_f1_s[0][i],
            "recall": p_r_f1_s[1][i],
            "f1-score": p_r_f1_s[2][i],
            "support": p_r_f1_s[3][i]
        }
    
    # Weighted average metrics ( mimics classification_report output)
    weighted_avg = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
    avg_metrics = {
        "precision": weighted_avg[0],
        "recall": weighted_avg[1],
        "f1-score": weighted_avg[2],
        "support": np.sum(p_r_f1_s[3]) # Total support
    }

    return cm, acc, metrics_per_class, avg_metrics

# --- Model Class ---
class Model:
    def __init__(self, name, classifier_instance, cost, preprocessor): # Added preprocessor argument
        self.name = name
        self.cost = cost
        self.classifier = classifier_instance
        self.cm = None
        self.accuracy = None
        self.metrics_per_class = {}
        self.avg_metrics = {}
        self.exit_classes = set()
        self.label_encoder_classes_ = None
        self.class_names_map = None
        self.preprocessor = preprocessor # Store the preprocessor

    def train(self, X_train, y_train):
        # Create a pipeline with the provided preprocessor and then the classifier
        pipeline_steps = []
        if self.preprocessor:
            pipeline_steps.append(('preprocessor', self.preprocessor))
        else:
            # Fallback to only scaling if no specific preprocessor is given
            # (though with ColumnTransformer, numeric scaling is part of it)
            pipeline_steps.append(('scaler', StandardScaler()))
        
        pipeline_steps.append(('classifier', self.classifier))
        
        self.trained_pipeline_ = Pipeline(steps=pipeline_steps)
        self.trained_pipeline_.fit(X_train, y_train)

    # predict, predict_proba, evaluate, get_quality methods remain largely the same
    # as they use self.trained_pipeline_ which now includes preprocessing.

    def predict(self, X_test): # No change needed
        return self.trained_pipeline_.predict(X_test)

    def predict_proba(self, X_test): # No change needed
        if hasattr(self.trained_pipeline_.named_steps['classifier'], 'predict_proba'):
            return self.trained_pipeline_.predict_proba(X_test)
        else:
            predictions = self.predict(X_test)
            n_classes = len(self.label_encoder_classes_)
            probas = np.zeros((X_test.shape[0], n_classes))
            class_indices = {label: i for i, label in enumerate(self.label_encoder_classes_)}
            for i, p_encoded in enumerate(predictions):
                if p_encoded in class_indices: # Ensure prediction is a known class
                    class_idx = class_indices[p_encoded]
                    probas[i, class_idx] = 1.0
                # else: handle unknown prediction if necessary, though classifiers usually predict known classes
            return probas


    def evaluate(self, X_test, y_test, labels, class_names_map): # No change needed
        self.label_encoder_classes_ = labels 
        self.class_names_map = class_names_map
        
        y_pred = self.predict(X_test)
        self.cm, self.accuracy, self.metrics_per_class, self.avg_metrics = get_quality_metrics(y_test, y_pred, labels, class_names_map)

    def get_quality(self, class_name, metric_type="f1-score"): # No change needed
        if class_name not in self.metrics_per_class:
            return 0.0 
        return self.metrics_per_class[class_name].get(metric_type, 0.0)

    def __repr__(self): # No change needed
        return f"Model(name='{self.name}', cost={self.cost}, accuracy={self.accuracy:.4f})"
# --- NOMAD Engine Class ---
class NOMADEngine:
    def __init__(self, models_dict, role_model_name, epsilon, 
                 initial_class_priors, unique_classes_ordered, class_names_map,
                 quality_metric_for_ec="f1-score", safety_check_type="conservative"):
        self.models_dict = models_dict # {name: Model_instance}
        self.role_model = self.models_dict[role_model_name]
        self.epsilon = epsilon
        self.initial_class_priors = initial_class_priors # dict: {class_name: prob}
        self.unique_classes_ordered = unique_classes_ordered # list of class_names in a fixed order
        self.class_names_map = class_names_map # map from encoded label to string name
        self.quality_metric_for_ec = quality_metric_for_ec
        self.safety_check_type = safety_check_type # "conservative" or "relaxed"

        if not self.role_model:
            raise ValueError(f"Role model '{role_model_name}' not found in provided models.")
            
        self._determine_all_exit_classes()

    def _determine_all_exit_classes(self):
        """
        Determines and sets the Exit Classes (EC) for each model based on the role model and epsilon.
        Definition 2.5: M_i is an exit model for C_j if M_i provides epsilon-comparable quality to M_r for C_j.
        quality(M_i, C_j) >= quality(M_r, C_j) * (1 - epsilon)
        """
        for model_name, model_obj in self.models_dict.items():
            model_obj.exit_classes = set()
            for class_name in self.unique_classes_ordered:
                quality_model = model_obj.get_quality(class_name, self.quality_metric_for_ec)
                quality_role_model = self.role_model.get_quality(class_name, self.quality_metric_for_ec)
                
                # Handle cases where role model quality might be 0 to avoid division by zero or illogical comparisons
                if quality_role_model == 0: # If role model is perfect (0 errors) for something, only perfect models can exit
                    if quality_model == 0: # Assuming metric is error-based if 0 is "perfect"
                         model_obj.exit_classes.add(class_name)
                # Or if quality_role_model is very low, any decent model might satisfy.
                # The paper definition is multiplicative, so if quality_role_model is 0,
                # any quality_model >= 0 satisfies it. This seems too loose.
                # A more robust check:
                # if quality_role_model > 1e-6: # Avoid issues with very small role model quality
                elif quality_model >= quality_role_model * (1 - self.epsilon):
                    model_obj.exit_classes.add(class_name)
                # elif quality_model > quality_role_model: # Always allow if better
                #     model_obj.exit_classes.add(class_name)

            # By definition, the role model M_r is an exit model for ALL classes (assuming epsilon >= 0)
            if model_obj == self.role_model:
                 model_obj.exit_classes = set(self.unique_classes_ordered)
            # print(f"Model {model_obj.name}: Exit Classes = {model_obj.exit_classes}")


    def _calculate_utility(self, model, current_class_priors):
        """
        Calculates utility for a model based on current class priors.
        U(M_i) = sum(p(C_j) for C_j in EC(M_i)) / cost(M_i)
        """
        if model.cost == 0: # Avoid division by zero for hypothetical zero-cost models
            return float('inf') if sum(current_class_priors[cn] for cn in model.exit_classes) > 0 else 0

        sum_prob_exit_classes = sum(current_class_priors[cn] for cn in model.exit_classes if cn in current_class_priors)
        return sum_prob_exit_classes / model.cost

    def _update_probabilities(self, p_in, softmax_output):
        """
        Updates class probabilities based on softmax output of a model.
        p_temp(C_j) = p_in(C_j) * softmax_j
        p_out(C_j) = p_temp(C_j) / sum(p_temp)
        softmax_output is a 1D array where indices map to self.unique_classes_ordered
        """
        p_temp = {}
        current_sum = 0
        for i, class_name in enumerate(self.unique_classes_ordered):
            val = p_in[class_name] * softmax_output[i]
            p_temp[class_name] = val
            current_sum += val
        
        p_out = {}
        if current_sum > 1e-9: # Avoid division by zero
            for class_name in self.unique_classes_ordered:
                p_out[class_name] = p_temp[class_name] / current_sum
        else: # Fallback: uniform distribution if sum is too small (e.g. all probabilities became zero)
            # print("Warning: Sum of probabilities too small in _update_probabilities. Resetting to uniform.")
            num_classes = len(self.unique_classes_ordered)
            for class_name in self.unique_classes_ordered:
                p_out[class_name] = 1.0 / num_classes
        return p_out

    def _check_chain_safety_conservative(self, M_new_name, S_current_names):
        """
        Algorithm 1: Check Class-Specific Chain Safety (Worst-Case / Conservative)
        S_current_names: list of names of models already in the chain for the current event.
        M_new_name: name of the new model being considered.
        """
        S_potential_names = S_current_names + [M_new_name]
        # print(f"  Safety Check (Conservative) for chain: {S_potential_names}")

        for class_name_target in self.unique_classes_ordered: # For each class C_j
            # Find the intended exit model M_exit(j) for C_j in S_potential.
            # This is the first model in S_potential for which class_name_target is an EC.
            # If no such model, M_exit(j) is effectively the role model.
            M_exit_for_class_j_name = None
            M_exit_for_class_j_idx = -1

            for idx, model_name_in_chain in enumerate(S_potential_names):
                model_in_chain = self.models_dict[model_name_in_chain]
                if class_name_target in model_in_chain.exit_classes:
                    M_exit_for_class_j_name = model_name_in_chain
                    M_exit_for_class_j_idx = idx
                    break
            
            # If no model in S_potential is an exit model for C_j,
            # then this chain *would* proceed to the role model for this class.
            # The safety check must ensure that even if it goes to M_r, the path is safe.
            if M_exit_for_class_j_name is None:
                # If M_r is not already the last element of S_potential, consider it as the exit model
                if self.role_model.name not in S_potential_names:
                    S_final_for_this_class_check = S_potential_names + [self.role_model.name]
                    M_exit_for_class_j_name = self.role_model.name
                    M_exit_for_class_j_idx = len(S_final_for_this_class_check) - 1
                elif S_potential_names[-1] == self.role_model.name: # Role model is already the potential last one
                     S_final_for_this_class_check = S_potential_names
                     M_exit_for_class_j_name = self.role_model.name
                     M_exit_for_class_j_idx = len(S_potential_names) -1
                else: # This case implies M_r was in S_potential but not as an exit for C_j, which is odd given M_r exits all.
                      # Default to checking against M_r as exit if not found.
                    S_final_for_this_class_check = S_potential_names + [self.role_model.name]
                    M_exit_for_class_j_name = self.role_model.name
                    M_exit_for_class_j_idx = len(S_final_for_this_class_check) - 1
            else:
                 S_final_for_this_class_check = S_potential_names


            M_exit_model_obj = self.models_dict[M_exit_for_class_j_name]

            # Calculate ChainRecall_j (Eq. worst_case_safety)
            # product(Recall(M_k, C_j) for M_k in S_potential up to M_exit(j)-1) * Recall(M_exit(j), C_j)
            chain_recall_j = 1.0
            for k_idx in range(M_exit_for_class_j_idx + 1): # Iterate up to and including the exit model index
                model_k_name = S_final_for_this_class_check[k_idx]
                model_k_obj = self.models_dict[model_k_name]
                recall_mk_cj = model_k_obj.get_quality(class_name_target, "recall")
                chain_recall_j *= recall_mk_cj
            
            recall_role_model_cj = self.role_model.get_quality(class_name_target, "recall")
            quality_bound = (1.0 - self.epsilon) * recall_role_model_cj
            
            # print(f"    Class '{class_name_target}': ChainRecall={chain_recall_j:.4f}, Bound={quality_bound:.4f} (Exit: {M_exit_for_class_j_name})")
            if chain_recall_j < quality_bound:
                # print(f"    UNSAFE for class '{class_name_target}'. Chain: {S_final_for_this_class_check}, Exit: {M_exit_for_class_j_name}")
                return False # Chain is not safe for this class C_j
        # print(f"  Chain SAFE.")
        return True # Chain is safe for all classes

    def _get_smoothed_misclassification_prob(self, model_k, class_j_true_name, class_i_pred_name):
        """
        Calculates P_Mk(Cj -> Ci)_smoothed using Eq. prob_misclassify_smoothed
        Requires model_k.cm, model_k.label_encoder_classes_, model_k.class_names_map
        """
        true_class_idx = np.where(np.array(self.unique_classes_ordered) == class_j_true_name)[0][0] # idx in unique_classes_ordered
        
        # Map unique_classes_ordered (string names) to indices in the model's CM
        # The model.cm is indexed by model.label_encoder_classes_
        # We need to find the actual numerical label for class_j_true_name and class_i_pred_name
        
        true_label_numeric = -1
        pred_label_numeric = -1

        for num_label, str_name in model_k.class_names_map.items():
            if str_name == class_j_true_name:
                true_label_numeric = num_label
            if str_name == class_i_pred_name:
                pred_label_numeric = num_label
        
        if true_label_numeric == -1: # Should not happen if data is consistent
            # print(f"Warning: True class name {class_j_true_name} not found in model {model_k.name}'s class map.")
            return 0.0

        # Find index of true_label_numeric in model_k.label_encoder_classes_ to index CM row
        cm_row_idx = np.where(model_k.label_encoder_classes_ == true_label_numeric)[0]
        if not cm_row_idx.size: 
            # print(f"Warning: True label {true_label_numeric} ({class_j_true_name}) not found in model {model_k.name}'s evaluated labels.")
            return 0.0 
        cm_row_idx = cm_row_idx[0]

        n_ji = 0 # Count of true class_j predicted as class_i
        if pred_label_numeric != -1:
            cm_col_idx = np.where(model_k.label_encoder_classes_ == pred_label_numeric)[0]
            if cm_col_idx.size:
                 cm_col_idx = cm_col_idx[0]
                 n_ji = model_k.cm[cm_row_idx, cm_col_idx]
            # else: pred_label_numeric was not seen by this model during its evaluation
        
        # Sum of all predictions for true class_j (row sum in CM)
        sum_n_jl = np.sum(model_k.cm[cm_row_idx, :]) 
        
        # For smoothing, S is total pseudo-counts. alpha_i is pseudo-count for class_i_pred_name.
        # Simplest: ALPHA_SMOOTHING for each class, so S = ALPHA_SMOOTHING * num_classes
        # Here, alpha_i is just ALPHA_SMOOTHING if class_i_pred_name is a valid class
        # Paper's example uses prior-based alpha. For simplicity, using constant ALPHA_SMOOTHING.
        alpha_i_val = ALPHA_SMOOTHING if pred_label_numeric != -1 else 0
        S_total_pseudo_counts = ALPHA_SMOOTHING * len(self.unique_classes_ordered)

        if sum_n_jl + S_total_pseudo_counts == 0: return 0.0 # Avoid division by zero
        
        return (n_ji + alpha_i_val) / (sum_n_jl + S_total_pseudo_counts)


    def _check_chain_safety_relaxed(self, M_new_name, S_current_names):
        """
        Implements Relaxed Chain Safety Estimation (Section 2.5)
        """
        S_potential_names = S_current_names + [M_new_name]
        # print(f"  Safety Check (Relaxed) for chain: {S_potential_names}")

        for class_j_true_name in self.unique_classes_ordered: # For each true class C_j
            M_exit_for_class_j_name = None
            M_exit_for_class_j_idx = -1
            S_final_for_this_class_check = S_potential_names # Default

            for idx, model_name_in_chain in enumerate(S_potential_names):
                model_in_chain = self.models_dict[model_name_in_chain]
                if class_j_true_name in model_in_chain.exit_classes:
                    M_exit_for_class_j_name = model_name_in_chain
                    M_exit_for_class_j_idx = idx
                    break
            
            if M_exit_for_class_j_name is None:
                if self.role_model.name not in S_potential_names:
                    S_final_for_this_class_check = S_potential_names + [self.role_model.name]
                    M_exit_for_class_j_name = self.role_model.name
                    M_exit_for_class_j_idx = len(S_final_for_this_class_check) - 1
                elif S_potential_names[-1] == self.role_model.name:
                     S_final_for_this_class_check = S_potential_names
                     M_exit_for_class_j_name = self.role_model.name
                     M_exit_for_class_j_idx = len(S_potential_names) -1
                else:
                    S_final_for_this_class_check = S_potential_names + [self.role_model.name]
                    M_exit_for_class_j_name = self.role_model.name
                    M_exit_for_class_j_idx = len(S_final_for_this_class_check) - 1

            M_exit_model_obj = self.models_dict[M_exit_for_class_j_name]
            
            # Product of refined pass-through probabilities for models *before* the exit model
            prod_pass_through_refined = 1.0
            for k_idx in range(M_exit_for_class_j_idx): # Models M_1 to M_{i-1}
                model_k_name = S_final_for_this_class_check[k_idx]
                model_k_obj = self.models_dict[model_k_name]

                # P_Mk(Cj -> EC(Mk))_smoothed = sum over i in EC(Mk) of P_Mk(Cj -> Ci)_smoothed
                prob_exit_misclassify_smoothed = 0.0
                for class_i_pred_name in model_k_obj.exit_classes: # Sum over exit classes of M_k
                    prob_exit_misclassify_smoothed += self._get_smoothed_misclassification_prob(
                        model_k_obj, class_j_true_name, class_i_pred_name
                    )
                
                # P_Mk(Cj -> ~EC(Mk))_refined = 1 - P_Mk(Cj -> EC(Mk))_smoothed
                pass_through_k_refined = 1.0 - prob_exit_misclassify_smoothed
                prod_pass_through_refined *= pass_through_k_refined
            
            # Full chain quality for C_j using refined estimate
            # [ product P_Mk(Cj -> ~EC(Mk))_refined ] * recall(M_exit, C_j)
            chain_quality_j_refined = prod_pass_through_refined * M_exit_model_obj.get_quality(class_j_true_name, "recall")
            
            recall_role_model_cj = self.role_model.get_quality(class_j_true_name, "recall")
            quality_bound = (1.0 - self.epsilon) * recall_role_model_cj
            
            # print(f"    Class '{class_j_true_name}': RefinedChainQuality={chain_quality_j_refined:.4f}, Bound={quality_bound:.4f} (Exit: {M_exit_for_class_j_name})")
            if chain_quality_j_refined < quality_bound:
                # print(f"    UNSAFE (Relaxed) for class '{class_j_true_name}'. Chain: {S_final_for_this_class_check}, Exit: {M_exit_for_class_j_name}")
                return False
        # print(f"  Chain SAFE (Relaxed).")
        return True


    def select_and_classify_event(self, event_features): # event_features is a 1D numpy array or Series for a single event
        """
        Algorithm 2: Dynamic Model Selection and Chaining for a single event.
        Returns: predicted_class_name, cost_of_chain, realized_chain_names
        """
        current_class_priors = self.initial_class_priors.copy()
        # Available models: dictionary of {name: Model_instance}
        # Make a shallow copy for modification within this method call
        models_available = {name: model for name, model in self.models_dict.items() if model != self.role_model}
        # Always ensure role model is an option if others fail, but it shouldn't be selected by utility unless it's the only one.
        # Or, it's handled by the fallback logic. The paper says M_r is in M_available.
        # Let's keep it simple: M_available initially contains all models.
        models_available_dict = self.models_dict.copy()


        realized_chain_names = []
        current_event_cost = 0
        final_prediction_name = None
        
        # Reshape event_features for scaler/predictor if it's a 1D array
        event_features_reshaped = event_features.to_frame().T


        while models_available_dict:
            # Select M* with max utility from M_available
            best_model_name = None
            max_utility = -float('inf')

            # Sort models by utility to ensure deterministic choice if utilities are equal (e.g. by cost, then name)
            # Or simply iterate and pick the first one that maximizes (Python's dict iteration order is insertion since 3.7)
            # For more deterministic behavior if utilities are same, sort available models
            
            # Create a list of (utility, cost, name) for available models to pick the best one
            utility_candidates = []
            for model_name, model_obj in models_available_dict.items():
                utility_candidates.append(
                    (self._calculate_utility(model_obj, current_class_priors), -model_obj.cost, model_name) # -cost to break ties favoring cheaper
                )
            
            if not utility_candidates: break # No models left to choose from

            # Sort by utility (desc), then by -cost (asc cost), then by name (alphabetic)
            utility_candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
            
            selected_model_name_by_utility = utility_candidates[0][2]
            M_star = self.models_dict[selected_model_name_by_utility]
            
            # Chain Safety Check (Line 9 of Algo 2 in paper)
            # Pass S_e (realized_chain_names)
            is_safe = False
            if self.safety_check_type == "conservative":
                is_safe = self._check_chain_safety_conservative(M_star.name, realized_chain_names)
            elif self.safety_check_type == "relaxed":
                 is_safe = self._check_chain_safety_relaxed(M_star.name, realized_chain_names)
            else: # Default to conservative
                is_safe = self._check_chain_safety_conservative(M_star.name, realized_chain_names)

            if not is_safe:
                # print(f"Model {M_star.name} skipped due to chain safety.")
                del models_available_dict[M_star.name] # Remove M* from consideration FOR THIS EVENT
                if not models_available_dict or (len(models_available_dict)==1 and self.role_model.name in models_available_dict and M_star.name != self.role_model.name):
                    # If only role model is left, or M_star was the last non-role model and unsafe
                    # The loop will eventually hit fallback or pick role model if it's safe.
                    # If M_star *was* the role model and it's unsafe, this is a problem with epsilon/role model.
                    if M_star.name == self.role_model.name:
                        # print(f"Warning: Role model {self.role_model.name} itself deemed unsafe. This implies issues with epsilon or base metrics.")
                        # This should ideally not happen if epsilon is reasonable as M_r vs M_r is 1.0 >= 1-epsilon.
                        # This could indicate an issue in the safety check logic for the role model itself.
                        # Fallback to role model prediction directly, ignoring safety for itself.
                        break 
                continue # Try next best utility model if chain with M* is unsafe

            # Execute M* (Line 12)
            # print(f"  Executing model: {M_star.name} (Cost: {M_star.cost})")
            pred_current_encoded = M_star.predict(event_features_reshaped)[0] # single prediction
            pred_current_name = self.class_names_map[pred_current_encoded]
            
            softmax_vector = M_star.predict_proba(event_features_reshaped)[0] # Probabilities for the single event

            current_event_cost += M_star.cost
            realized_chain_names.append(M_star.name)
            final_prediction_name = pred_current_name # Tentative final prediction

            # Check if prediction is an exit class for M* (Line 14)
            if pred_current_name in M_star.exit_classes:
                # print(f"    Exit condition met. Predicted: {pred_current_name} by {M_star.name}")
                return final_prediction_name, current_event_cost, realized_chain_names
            
            # If not an exit class (Line 17)
            del models_available_dict[M_star.name] # Remove M* from consideration for this event

            # Update probabilities (Line 20)
            current_class_priors = self._update_probabilities(current_class_priors, softmax_vector)
            # print(f"    Probabilities updated. Next priors: { {k:round(v,3) for k,v in current_class_priors.items()} }")
            
            # Optional: Break if M_r ran but didn't exit (Line 22)
            # This means M_r predicted a class not in its EC, which is impossible by def of EC(M_r) = all classes
            # So, this condition M_star == self.role_model should imply an exit.
            # The only way it wouldn't exit is if pred_current_name was somehow not in M_r.exit_classes.
            if M_star == self.role_model:
                # This implies the role model was chosen, executed, but its prediction wasn't an exit class.
                # This should not happen if EC(M_r) contains all classes.
                # If it does happen, it means the predicted class by M_r was not in its own EC set,
                # which implies an issue with _determine_all_exit_classes for M_r or the prediction mapping.
                # print(f"Warning: Role model {self.role_model.name} executed but its prediction '{pred_current_name}' was not in its exit classes {self.role_model.exit_classes}. This is unexpected.")
                # The fallback logic below will handle final prediction using M_r if loop terminates.
                break 

        # Fallback Logic (Lines 25-30)
        # If loop finishes, means no model exited or M_available became empty.
        # If final_prediction_name is not null AND M_r was run (i.e., M_r is not in models_available_dict)
        # This means M_r was the last model in a chain that didn't meet exit criteria *for M_r*.
        # This case is tricky. By definition, M_r's EC contains all classes. So if M_r runs, it *must* exit.
        # The loop termination condition `while models_available_dict` or if M_star == self.role_model (and it exits) covers this.
        # The primary fallback is if the loop exhausts models *before* M_r is chosen or M_r is unsafe.
        
        # If we are here, it means the while loop exited.
        # This typically means either all models were tried and failed to make an exit prediction,
        # or models were progressively deemed unsafe.
        # The safest bet is to use the role model.
        
        # print(f"  Fallback: Executing Role Model {self.role_model.name}")
        if self.role_model.name not in realized_chain_names: # Only add cost if not already run
            current_event_cost += self.role_model.cost
            realized_chain_names.append(self.role_model.name)
            
        pred_final_encoded = self.role_model.predict(event_features_reshaped)[0]
        final_prediction_name = self.class_names_map[pred_final_encoded]
        
        return final_prediction_name, current_event_cost, realized_chain_names

    def evaluate_strategy(self, X_test, y_test_encoded):
        """
        Applies the NOMAD strategy to the entire test set and evaluates its performance.
        y_test_encoded: true labels, numerically encoded
        """
        nomad_predictions_encoded = []
        total_cost = 0
        all_realized_chains = []

        # Map y_test_encoded back to names for comparison if needed, or keep encoded
        # For get_quality_metrics, we need encoded predictions and true values.
        
        for i in range(X_test.shape[0]):
            event_features = X_test.iloc[i] if isinstance(X_test, pd.DataFrame) else X_test[i,:]
            # print(f"\nProcessing event {i+1}/{X_test.shape[0]}")
            pred_name, cost, chain = self.select_and_classify_event(event_features)
            
            # Convert predicted name back to encoded label for metrics
            pred_encoded = -1 # Default for unknown
            for enc_label, name_val in self.class_names_map.items():
                if name_val == pred_name:
                    pred_encoded = enc_label
                    break
            if pred_encoded == -1 :
                 print(f"Warning: Predicted class name '{pred_name}' could not be mapped back to an encoded label.")
                 # Fallback to a default prediction or handle error
                 # For now, let's pick the first class if this happens
                 pred_encoded = self.role_model.label_encoder_classes_[0]


            nomad_predictions_encoded.append(pred_encoded)
            total_cost += cost
            all_realized_chains.append(chain)
        
        avg_cost = total_cost / X_test.shape[0]
        
        # Evaluate NOMAD's predictions
        # Ensure unique_classes_ordered_numeric corresponds to the actual numeric labels used in y_test_encoded
        # and nomad_predictions_encoded. These are the `labels` argument for metrics.
        # The `self.role_model.label_encoder_classes_` should hold these from its evaluation.
        
        cm_nomad, acc_nomad, metrics_per_class_nomad, avg_metrics_nomad = get_quality_metrics(
            y_test_encoded, 
            nomad_predictions_encoded, 
            labels=self.role_model.label_encoder_classes_, # Use the numeric labels
            class_names_map=self.class_names_map
        )
        
        return {
            "cm": cm_nomad,
            "accuracy": acc_nomad,
            "metrics_per_class": metrics_per_class_nomad,
            "avg_metrics": avg_metrics_nomad,
            "average_cost": avg_cost,
            "all_realized_chains": all_realized_chains, # For inspection
            "nomad_predictions_encoded": nomad_predictions_encoded # For further analysis
        }

# --- Main Execution ---
if __name__ == '__main__':
    # 1. Load Data
    csv_file_path = input("Enter path to your CSV file: ")
    try:
        data = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_file_path}")
        exit()
    except Exception as e:
        print(f"Error loading CSV: {e}")
        exit()

    if data.shape[1] < 2:
        print("Error: CSV must have at least two columns (one feature, one target).")
        exit()

    X = data.iloc[:, :-2]
    y_raw = data.iloc[:, -2]

     # 2. Preprocess Labels (target)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    unique_labels_encoded = np.sort(np.unique(y))
    class_names_map = {encoded_label: original_name for encoded_label, original_name in zip(unique_labels_encoded, label_encoder.inverse_transform(unique_labels_encoded))}
    unique_class_names_ordered = [class_names_map[el] for el in unique_labels_encoded]

    print(f"Target classes: {unique_class_names_ordered}")

    # --- FEATURE PREPROCESSING SETUP ---
    # Identify categorical and numerical columns IN THE ORIGINAL X
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Identified numerical feature columns: {numerical_cols}")
    print(f"Identified categorical feature columns: {categorical_cols}")

    # Create a preprocessor object using ColumnTransformer
    # Numerical features will be scaled.
    # Categorical features will be one-hot encoded.
    # handle_unknown='ignore' will prevent errors if new categories appear in test data (they become all zeros)
    # sparse_output=False to make it easier to work with standard scalers/classifiers,
    # but can be True if memory is an issue and downstream components support sparse data.
    preprocessor_for_models = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='passthrough' # Use 'drop' if you want to ensure only specified columns are used
    )
    # --- END OF FEATURE PREPROCESSING SETUP ---


    # Split data AFTER identifying column types from the full X, but BEFORE applying transformations
    # The preprocessor will be fit on X_train and transform X_train and X_test within the model's pipeline
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print(f"Data loaded: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples.")
    print(f"Classes found: {unique_class_names_ordered}")
    # print(f"Class mapping (encoded to name): {class_names_map}") # Already printed
    print(type(X_test))

    # 3. Train Models and Gather Metrics
    trained_models_dict = {}
    print("\n--- Training and Evaluating Individual Models ---")
    for name, clf_instance, cost_val in CANDIDATE_MODELS_CONFIG:
        print(f"Training {name}...")
        # Pass the preprocessor to each model instance
        model = Model(name, clf_instance, cost_val, preprocessor=preprocessor_for_models)
        try:
            model.train(X_train, y_train) # X_train is raw here, preprocessing happens in pipeline
            model.evaluate(X_test, y_test, labels=unique_labels_encoded, class_names_map=class_names_map)
            trained_models_dict[name] = model
            print(f"  {name}: Accuracy = {model.accuracy:.4f}, Cost = {model.cost}")
        except Exception as e:
            print(f"  Error training or evaluating {name}: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            
    if not trained_models_dict:
        print("No models were successfully trained. Exiting.")
        exit()

    print("\n--- Individual Model Performance Summary ---")
    header = f"{'Model Name':<30} | {'Accuracy':<10} | {'W. F1-Score':<12} | {'Cost':<5} | "
    for cn in unique_class_names_ordered: header += f"{'F1('+cn[:5]+')':<10} | " # Shorten class name for header
    print(header)
    print("-" * len(header))

    for name, model in trained_models_dict.items():
        row = f"{name:<30} | {model.accuracy:<10.4f} | {model.avg_metrics['f1-score']:<12.4f} | {model.cost:<5.1f} | "
        for cn in unique_class_names_ordered:
            row += f"{model.get_quality(cn, 'f1-score'):<10.4f} | "
        print(row)

    # 4. User Input for NOMAD Setup
    print("\n--- NOMAD Engine Setup ---")
    role_model_name = ""
    while role_model_name not in trained_models_dict:
        print("Available models for Role Model:")
        for name in trained_models_dict.keys(): print(f" - {name}")
        role_model_name = input("Select Role Model name from the list: ")
        if role_model_name not in trained_models_dict:
            print("Invalid model name. Please choose from the list.")

    epsilon_str = input(f"Enter quality tolerance epsilon (e.g., 0.05 for 5% degradation compared to '{role_model_name}'): ")
    try:
        epsilon = float(epsilon_str)
        if not (0 <= epsilon < 1):
            raise ValueError("Epsilon must be between 0 (inclusive) and 1 (exclusive).")
    except ValueError as e:
        print(f"Invalid epsilon: {e}. Using default 0.05.")
        epsilon = 0.05
        
    quality_metric_for_ec_input = input(f"Enter quality metric for Exit Class determination (precision, recall, or f1-score) [default: f1-score]: ").lower()
    if quality_metric_for_ec_input not in ["precision", "recall", "f1-score", ""]:
        print("Invalid metric. Using 'f1-score'.")
        quality_metric_for_ec_input = "f1-score"
    elif quality_metric_for_ec_input == "":
        quality_metric_for_ec_input = "f1-score"

    safety_check_type_input = input("Use 'conservative' or 'relaxed' chain safety check? [default: conservative]: ").lower()
    if safety_check_type_input not in ["conservative", "relaxed", ""]:
        print("Invalid safety check type. Using 'conservative'.")
        safety_check_type_input = "conservative"
    elif safety_check_type_input == "":
        safety_check_type_input = "conservative"


    # Initial class priors (can be from training data distribution or other domain knowledge)
    _, class_counts = np.unique(y_train, return_counts=True)
    initial_class_priors_list = class_counts / len(y_train)
    initial_class_priors_dict = {
        unique_class_names_ordered[i]: initial_class_priors_list[i] 
        for i in range(len(unique_class_names_ordered))
    }
    print(f"Using initial class priors from training data: { {k:round(v,3) for k,v in initial_class_priors_dict.items()} }")


    # 5. Initialize and Run NOMAD Engine
    nomad_engine = NOMADEngine(
        models_dict=trained_models_dict,
        role_model_name=role_model_name,
        epsilon=epsilon,
        initial_class_priors=initial_class_priors_dict,
        unique_classes_ordered=unique_class_names_ordered,
        class_names_map=class_names_map,
        quality_metric_for_ec=quality_metric_for_ec_input,
        safety_check_type=safety_check_type_input
    )
    
    print(f"\nNOMAD Engine initialized with Role Model: '{nomad_engine.role_model.name}', Epsilon: {nomad_engine.epsilon}, EC Metric: '{nomad_engine.quality_metric_for_ec}', Safety: '{nomad_engine.safety_check_type}'.")
    print("Determined Exit Classes for each model:")
    for name, model in nomad_engine.models_dict.items():
        print(f"  {name}: EC = {model.exit_classes if model.exit_classes else '{}'}")


    print("\n--- Evaluating NOMAD Strategy on Test Set ---")
    nomad_results = nomad_engine.evaluate_strategy(X_test, y_test)

    # 6. Display Results
    print("\n--- NOMAD Strategy Performance ---")
    print(f"Overall Accuracy: {nomad_results['accuracy']:.4f}")
    print(f"Weighted Avg Precision: {nomad_results['avg_metrics']['precision']:.4f}")
    print(f"Weighted Avg Recall: {nomad_results['avg_metrics']['recall']:.4f}")
    print(f"Weighted Avg F1-Score: {nomad_results['avg_metrics']['f1-score']:.4f}")
    print(f"Average Cost per Event: {nomad_results['average_cost']:.4f}")
    
    print("\nPer-Class Metrics for NOMAD Strategy:")
    print(f"{'Class':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<10}")
    print("-" * 70)
    for class_name, metrics in nomad_results['metrics_per_class'].items():
        print(f"{class_name:<15} | {metrics['precision']:<10.4f} | {metrics['recall']:<10.4f} | {metrics['f1-score']:<10.4f} | {int(metrics['support']):<10}")

    print("\nConfusion Matrix for NOMAD Strategy (Rows: True, Cols: Predicted):")
    # Display CM with labels
    cm_df_nomad = pd.DataFrame(nomad_results['cm'], index=unique_class_names_ordered, columns=unique_class_names_ordered)
    print(cm_df_nomad)

    print("\n--- Role Model Performance (for comparison) ---")
    role_model_obj = nomad_engine.role_model
    print(f"Role Model: {role_model_obj.name}")
    print(f"Overall Accuracy: {role_model_obj.accuracy:.4f}")
    print(f"Weighted Avg Precision: {role_model_obj.avg_metrics['precision']:.4f}")
    print(f"Weighted Avg Recall: {role_model_obj.avg_metrics['recall']:.4f}")
    print(f"Weighted Avg F1-Score: {role_model_obj.avg_metrics['f1-score']:.4f}")
    print(f"Cost: {role_model_obj.cost:.4f}")
    
    print("\nPer-Class Metrics for Role Model:")
    print(f"{'Class':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<10}")
    print("-" * 70)
    for class_name, metrics in role_model_obj.metrics_per_class.items():
        print(f"{class_name:<15} | {metrics['precision']:<10.4f} | {metrics['recall']:<10.4f} | {metrics['f1-score']:<10.4f} | {int(metrics['support']):<10}")
    
    print("\nConfusion Matrix for Role Model (Rows: True, Cols: Predicted):")
    cm_df_role = pd.DataFrame(role_model_obj.cm, index=unique_class_names_ordered, columns=unique_class_names_ordered)
    print(cm_df_role)

    # Example of chains (first 5 events)
    print("\nExample Realized Chains (first 5 events):")
    for i, chain in enumerate(nomad_results['all_realized_chains'][:5]):
        true_label_name = class_names_map[y_test[i]]
        pred_label_encoded = nomad_results['nomad_predictions_encoded'][i]
        pred_label_name = class_names_map[pred_label_encoded]
        print(f"  Event {i+1}: True='{true_label_name}', Predicted='{pred_label_name}', Chain={chain}")