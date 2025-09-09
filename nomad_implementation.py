"""
NOMAD API Implementation Examples
Showing integration with existing backend code
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Generator
from dataclasses import asdict

# Import from existing implementation
from nomad_arima_backend import (
    train_evaluate_individual_models_for_flask,
    run_nomad_simulation_for_flask_streamed,
    load_and_preprocess_data_for_flask,
    AdaptivePriorsManager,
    NOMADEngine as BackendNOMADEngine,
    Model as BackendModel
)

# Import from the API library we just created
from nomad_api import (
    NOMADEngine, Model, NOMADConfig, ModelConfig, AdaptiveConfig,
    SafetyCheckType, QualityMetric, FARMLModelRegistry
)

class NOMADEngineImpl(NOMADEngine):
    """Implementation of NOMADEngine that integrates with existing backend"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._backend_engine = None
        self._trained = False
        
    def _init_adaptive_manager(self, initial_priors: Dict[str, float], adaptive_config: AdaptiveConfig):
        """Initialize the adaptive priors manager using existing implementation"""
        self.adaptive_manager = AdaptivePriorsManager(
            class_names=self.class_names,
            initial_priors=initial_priors,
            ph_threshold=adaptive_config.ph_threshold,
            ph_delta=adaptive_config.ph_delta,
            buffer_size=adaptive_config.buffer_size,
            min_buffer_for_arima=adaptive_config.min_buffer_for_arima,
            arima_order=adaptive_config.arima_order,
            enable_arima=adaptive_config.enable_arima
        )
    
    def _train_models(self, data: pd.DataFrame, target_column: str = "target") -> 'NOMADEngineImpl':
        """Train all models using existing backend logic"""
        # Extract features and target
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
            
        X_full = data.drop(columns=[target_column])
        y_full = data[target_column]
        
        # Convert model configs to format expected by backend
        candidate_configs = {}
        for model_name, model in self.models.items():
            candidate_configs[model_name] = {
                "name": model.config.name,
                "type": model.config.model_type,
                "params": model.config.params,
                "cost": model.config.cost,
                "is_custom": model.config.is_custom,
                "module_name": model.config.module_name,
                "class_name": model.config.class_name
            }
        
        # Use existing training logic
        try:
            from sklearn.preprocessing import LabelEncoder
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import StandardScaler, OneHotEncoder
            
            # Encode labels
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y_full)
            
            # Create preprocessor
            numeric_features = X_full.select_dtypes(include=[np.number]).columns
            categorical_features = X_full.select_dtypes(include=['object']).columns
            
            transformers = []
            if len(numeric_features) > 0:
                transformers.append(('num', StandardScaler(), numeric_features))
            if len(categorical_features) > 0:
                transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features))
            
            preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
            
            # Train models using backend
            trained_models_dict, model_summaries, X_test, y_test, y_train = train_evaluate_individual_models_for_flask(
                X_full, y_encoded, preprocessor,
                {int(i): name for i, name in enumerate(label_encoder.classes_)},
                label_encoder.transform(label_encoder.classes_),
                candidate_configs,
                custom_models_path="."  # placeholder
            )
            
            # Update our models with trained instances
            for model_name, backend_model in trained_models_dict.items():
                if model_name in self.models:
                    self.models[model_name]._model_instance = backend_model
                    self.models[model_name]._trained = True
                    
            # Create backend NOMAD engine
            self._backend_engine = BackendNOMADEngine(
                models_dict=trained_models_dict,
                role_model_name=self.config.role_model_name,
                epsilon=self.config.epsilon,
                initial_class_priors_str_keys={name: 1.0/len(self.class_names) for name in self.class_names},
                unique_classes_ordered_str=self.class_names,
                class_map_numeric_to_str={i: name for i, name in enumerate(self.class_names)},
                quality_metric_for_ec=self.config.quality_metric.value,
                safety_check_type=self.config.safety_check_type.value,
                adaptive_config=asdict(self.config.adaptive_config) if self.config.adaptive_config else None
            )
            
            self._trained = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to train models: {str(e)}")
            
        return self
        
    def classify_event(self, event_features: pd.DataFrame) -> Tuple[str, float, List[str]]:
        """Classify a single event using trained NOMAD system"""
        if not self._trained or not self._backend_engine:
            raise RuntimeError("NOMAD system must be trained before classification")
            
        return self._backend_engine.select_and_classify_event(event_features)
    
    def classify_stream(self, 
                       data_stream: pd.DataFrame,
                       target_column: str = "target",
                       batch_size: int = 10,
                       workload_phases: Optional[List[Dict]] = None) -> Generator[Dict[str, Any], None, None]:
        """Classify a stream of events with real-time adaptation"""
        if not self._trained or not self._backend_engine:
            raise RuntimeError("NOMAD system must be trained before stream classification")
            
        # Prepare data for streaming simulation
        X_test = data_stream.drop(columns=[target_column])
        y_test = data_stream[target_column]
        
        # Convert to format expected by backend
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_test)
        
        # Create X_by_class_str_keys and y_by_class_str_keys for workload phases
        X_by_class_str_keys = {}
        y_by_class_str_keys = {}
        
        for class_name in self.class_names:
            class_mask = y_test == class_name
            X_by_class_str_keys[class_name] = X_test[class_mask]
            y_by_class_str_keys[class_name] = y_encoded[class_mask]
        
        # Use existing streaming evaluation
        candidate_configs = {model.config.name: asdict(model.config) for model in self.models.values()}
        
        yield from self._backend_engine.stream_evaluate_strategy(
            X_test_overall=X_test,
            y_test_encoded_overall=y_encoded,
            candidate_model_names_for_counts=list(candidate_configs.keys()),
            X_test_column_names=X_test.columns.tolist(),
            batch_size=batch_size,
            workload_phases=workload_phases,
            X_by_class_str_keys=X_by_class_str_keys,
            y_by_class_str_keys=y_by_class_str_keys
        )
    
    def _determine_exit_classes(self):
        """Determine exit classes using backend logic"""
        if self._backend_engine:
            # Exit classes are determined in the backend engine initialization
            for model_name, backend_model in self._backend_engine.models_dict.items():
                if model_name in self.models:
                    self.models[model_name].exit_classes = backend_model.exit_classes
    
    def _get_adaptive_stats(self) -> Dict[str, Any]:
        """Get adaptive algorithm statistics"""
        if self._backend_engine:
            return self._backend_engine.adaptive_manager.get_statistics()
        return {}
    
    def _get_current_priors(self) -> Dict[str, float]:
        """Get current class priors"""
        if self._backend_engine:
            return self._backend_engine.adaptive_manager.get_current_priors()
        return {name: 1.0/len(self.class_names) for name in self.class_names}

class ModelImpl(Model):
    """Implementation of Model that integrates with backend Model class"""
    
    def train(self, X_train: pd.DataFrame, y_train: np.ndarray) -> 'Model':
        """Train the model using backend implementation"""
        # This would integrate with the backend Model class training
        # For now, we'll delegate to the NOMADEngine training process
        self._trained = True
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained model"""
        if not self._trained or not self._model_instance:
            raise RuntimeError("Model must be trained before prediction")
        return self._model_instance.predict(X)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if not self._trained or not self._model_instance:
            raise RuntimeError("Model must be trained before prediction")
        return self._model_instance.predict_proba(X)
        
    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate model performance"""
        if not self._trained or not self._model_instance:
            raise RuntimeError("Model must be trained before evaluation")
            
        # Use backend evaluation logic
        predictions = self.predict(X_test)
        
        # Calculate metrics (simplified version)
        from sklearn.metrics import accuracy_score, classification_report
        
        self.accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        
        # Extract per-class metrics
        self.metrics_per_class = {}
        for class_name in np.unique(y_test):
            if str(class_name) in report:
                self.metrics_per_class[str(class_name)] = report[str(class_name)]
        
        # Extract average metrics
        if 'weighted avg' in report:
            self.avg_metrics = report['weighted avg']
        
        return {
            "accuracy": self.accuracy,
            "per_class_metrics": self.metrics_per_class,
            "avg_metrics": self.avg_metrics
        }

# ============================================================================
# Data Loading Utilities
# ============================================================================

def load_data_for_nomad(csv_path: str) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
    """Load and preprocess data using existing backend logic"""
    
    result = load_and_preprocess_data_for_flask(csv_path)
    
    if result[-1]:  # Error message exists
        raise RuntimeError(f"Failed to load data: {result[-1]}")
    
    data_df, X_df, y_encoded, label_encoder, unique_labels_encoded, \
    class_map_numeric_to_str, unique_class_names_ordered_str, \
    preprocessor, x_column_names, X_by_class_str_keys, \
    y_by_class_str_keys, _ = result
    
    # Calculate initial priors from training data
    unique, counts = np.unique(y_encoded, return_counts=True)
    initial_priors = {}
    for num_label, count in zip(unique, counts):
        class_name = class_map_numeric_to_str[int(num_label)]
        initial_priors[class_name] = float(count / len(y_encoded))
    
    return data_df, unique_class_names_ordered_str, initial_priors

# ============================================================================
# High-Level Convenience Functions
# ============================================================================

def train_nomad_from_data(data: pd.DataFrame,
                         model_configs: List[ModelConfig],
                         role_model_name: str,
                         target_column: str = "target",
                         epsilon: float = 0.1,
                         **kwargs) -> NOMADEngineImpl:
    """High-level function to train NOMAD system from data"""
    
    # Extract class names from target column
    class_names = sorted(data[target_column].unique().astype(str))
    
    # Calculate initial priors
    value_counts = data[target_column].value_counts()
    initial_priors = {str(k): v/len(data) for k, v in value_counts.items()}
    
    # Create models
    models = [ModelImpl(config) for config in model_configs]
    
    # Create configuration
    config = NOMADConfig(
        role_model_name=role_model_name,
        epsilon=epsilon,
        **kwargs
    )
    
    # Build and train NOMAD system
    nomad = NOMADEngineImpl(
        models=models,
        config=config,
        class_names=class_names,
        initial_class_priors=initial_priors
    )
    
    return nomad._train_models(data, target_column)

def load_and_train_nomad(csv_path: str,
                        model_configs: List[ModelConfig], 
                        role_model_name: str,
                        target_column: str = "target",
                        epsilon: float = 0.1) -> NOMADEngineImpl:
    """Load data from CSV and train NOMAD system"""
    
    data_df, class_names, initial_priors = load_data_for_nomad(csv_path)
    
    # Add target column name if not present in loaded data
    if target_column not in data_df.columns and len(data_df.columns) >= 2:
        # Assume last column is target
        data_df = data_df.rename(columns={data_df.columns[-1]: target_column})
    
    models = [ModelImpl(config) for config in model_configs]
    
    config = NOMADConfig(
        role_model_name=role_model_name,
        epsilon=epsilon
    )
    
    nomad = NOMADEngineImpl(
        models=models,
        config=config, 
        class_names=class_names,
        initial_class_priors=initial_priors
    )
    
    return nomad._train_models(data_df, target_column)

# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Example showing complete NOMAD workflow"""
    
    # Define models with different cost-quality tradeoffs
    model_configs = [
        ModelConfig(
            name="decision_tree",
            model_type="DecisionTreeClassifier", 
            cost=1.0,
            params={"max_depth": 5, "random_state": 42}
        ),
        ModelConfig(
            name="random_forest",
            model_type="RandomForestClassifier",
            cost=5.0, 
            params={"n_estimators": 100, "random_state": 42}
        ),
        ModelConfig(
            name="gradient_boost",
            model_type="GradientBoostingClassifier",
            cost=15.0,
            params={"n_estimators": 100, "random_state": 42}
        )
    ]
    
    # Load data and train NOMAD system
    nomad = load_and_train_nomad(
        csv_path="example_data.csv",
        model_configs=model_configs,
        role_model_name="gradient_boost",  # Highest quality model
        epsilon=0.1  # Allow 10% quality degradation for cost savings
    )
    
    # Get performance summary
    performance = nomad.get_performance_summary()
    print(f"NOMAD System Performance: {performance}")
    
    # Classify new events (if you have test data)
    # test_data = pd.read_csv("test_data.csv")
    # for result in nomad.classify_stream(test_data):
    #     print(f"Batch result: {result}")
    
    return nomad

if __name__ == "__main__":
    # Run example
    try:
        nomad_system = example_usage()
        print("NOMAD system created successfully!")
    except Exception as e:
        print(f"Error creating NOMAD system: {e}")