"""
NOMAD - Navigating Optimal Model Application to Datastreams
A Python API library for cost-efficient real-time multi-class classification

Part of the FARML Toolbox ecosystem
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import warnings

# ============================================================================
# Core Data Structures and Enums
# ============================================================================

class SafetyCheckType(Enum):
    """Types of chain safety checks available in NOMAD"""
    CONSERVATIVE = "conservative"
    RELAXED = "relaxed"

class QualityMetric(Enum):
    """Quality metrics for model evaluation and exit conditions"""
    F1_SCORE = "f1-score"
    PRECISION = "precision"
    RECALL = "recall"
    ACCURACY = "accuracy"

@dataclass
class ModelConfig:
    """Configuration for individual models in NOMAD"""
    name: str
    model_type: str
    params: Dict[str, Any]
    cost: float
    is_custom: bool = False
    module_name: Optional[str] = None
    class_name: Optional[str] = None
    prerequisites: Optional[List[str]] = None

@dataclass
class AdaptiveConfig:
    """Configuration for adaptive distribution tracking"""
    ph_threshold: float = 50.0
    ph_delta: float = 0.005
    buffer_size: int = 100
    min_buffer_for_arima: int = 20
    arima_order: Tuple[int, int, int] = (1, 0, 1)
    enable_arima: bool = True

@dataclass
class NOMADConfig:
    """Main configuration for NOMAD engine"""
    role_model_name: str
    epsilon: float = 0.05
    quality_metric: QualityMetric = QualityMetric.F1_SCORE
    safety_check_type: SafetyCheckType = SafetyCheckType.CONSERVATIVE
    adaptive_config: AdaptiveConfig = None
    enable_dependent_models: bool = False

# ============================================================================
# FARML Integration Interfaces (Placeholders)
# ============================================================================

class FARMLDataConnector(ABC):
    """Interface for connecting to FARML data sources"""
    
    @abstractmethod
    def load_data(self, source_config: Dict[str, Any]) -> pd.DataFrame:
        """Load data from configured FARML data source"""
        pass
    
    @abstractmethod
    def stream_data(self, source_config: Dict[str, Any]):
        """Stream data from configured FARML data source"""
        pass

class FARMLModelRegistry(ABC):
    """Interface for FARML model registry integration"""
    
    @abstractmethod
    def register_model(self, model_config: ModelConfig, model_artifact: Any) -> str:
        """Register a trained model in FARML registry"""
        pass
    
    @abstractmethod
    def load_model(self, model_id: str) -> Any:
        """Load a model from FARML registry"""
        pass
    
    @abstractmethod
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models in registry"""
        pass

class FARMLWorkflowEngine(ABC):
    """Interface for FARML workflow automation"""
    
    @abstractmethod
    def create_pipeline(self, pipeline_config: Dict[str, Any]) -> str:
        """Create a new FARML pipeline"""
        pass
    
    @abstractmethod
    def execute_pipeline(self, pipeline_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a FARML pipeline"""
        pass

# ============================================================================
# Core NOMAD API Classes
# ============================================================================

class Model:
    """Wrapper for individual models in NOMAD system"""
    
    def __init__(self, config: ModelConfig, farml_registry: Optional[FARMLModelRegistry] = None):
        self.config = config
        self.farml_registry = farml_registry
        self._model_instance = None
        self._trained = False
        
        # Performance metrics (populated after evaluation)
        self.accuracy = 0.0
        self.metrics_per_class = {}
        self.avg_metrics = {}
        self.exit_classes = set()
        
    def train(self, X_train: pd.DataFrame, y_train: np.ndarray) -> 'Model':
        """Train the model with provided data"""
        # Implementation would use the existing training logic
        # from nomad_arima_backend.py
        pass
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on input data"""
        pass
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        pass
        
    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate model performance and update metrics"""
        pass
        
    def get_quality(self, class_name: str, metric_type: str = "f1-score") -> float:
        """Get quality metric for specific class"""
        return self.metrics_per_class.get(class_name, {}).get(metric_type, 0.0)

class NOMADEngine:
    """Main NOMAD engine for cost-efficient classification"""
    
    def __init__(self, 
                 models: List[Model], 
                 config: NOMADConfig,
                 class_names: List[str],
                 initial_class_priors: Optional[Dict[str, float]] = None,
                 farml_registry: Optional[FARMLModelRegistry] = None):
        
        self.models = {model.config.name: model for model in models}
        self.config = config
        self.class_names = class_names
        self.farml_registry = farml_registry
        
        # Validate role model exists
        if config.role_model_name not in self.models:
            raise ValueError(f"Role model '{config.role_model_name}' not found in provided models")
        
        self.role_model = self.models[config.role_model_name]
        
        # Initialize class priors
        if initial_class_priors is None:
            initial_class_priors = {name: 1.0/len(class_names) for name in class_names}
        
        # Initialize adaptive manager (using existing implementation)
        adaptive_config = config.adaptive_config or AdaptiveConfig()
        self._init_adaptive_manager(initial_class_priors, adaptive_config)
        
        # Determine exit classes for all models
        self._determine_exit_classes()
        
    def _init_adaptive_manager(self, initial_priors: Dict[str, float], adaptive_config: AdaptiveConfig):
        """Initialize the adaptive priors manager"""
        # This would use the existing AdaptivePriorsManager from the backend
        pass
        
    def _determine_exit_classes(self):
        """Determine exit classes for all models based on epsilon-comparability"""
        # Implementation from existing NOMADEngine._determine_all_exit_classes
        pass
    
    def classify_event(self, event_features: pd.DataFrame) -> Tuple[str, float, List[str]]:
        """
        Classify a single event using NOMAD strategy
        
        Returns:
            - Predicted class name
            - Total cost incurred
            - List of models used in the chain
        """
        # Implementation from existing select_and_classify_event method
        pass
    
    def classify_stream(self, 
                       data_stream, 
                       batch_size: int = 10,
                       workload_phases: Optional[List[Dict]] = None) -> Any:
        """
        Classify a stream of events with real-time adaptation
        
        Yields classification results and performance metrics
        """
        # Implementation from existing stream_evaluate_strategy method
        pass
    
    def add_model(self, model: Model) -> 'NOMADEngine':
        """Add a new model to the NOMAD system"""
        self.models[model.config.name] = model
        self._determine_exit_classes()  # Recalculate exit classes
        return self
    
    def remove_model(self, model_name: str) -> 'NOMADEngine':
        """Remove a model from the NOMAD system"""
        if model_name == self.config.role_model_name:
            raise ValueError("Cannot remove the role model")
        if model_name in self.models:
            del self.models[model_name]
            self._determine_exit_classes()
        return self
    
    def update_config(self, new_config: NOMADConfig) -> 'NOMADEngine':
        """Update NOMAD configuration"""
        self.config = new_config
        if new_config.role_model_name not in self.models:
            raise ValueError(f"New role model '{new_config.role_model_name}' not available")
        self.role_model = self.models[new_config.role_model_name]
        self._determine_exit_classes()
        return self
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return {
            "adaptive_stats": self._get_adaptive_stats(),
            "model_performance": {name: model.avg_metrics for name, model in self.models.items()},
            "current_priors": self._get_current_priors()
        }
    
    def _get_adaptive_stats(self) -> Dict[str, Any]:
        """Get adaptive algorithm statistics"""
        pass
    
    def _get_current_priors(self) -> Dict[str, float]:
        """Get current class priors"""
        pass

# ============================================================================
# High-Level NOMAD Builder API
# ============================================================================

class NOMADBuilder:
    """Builder pattern for easy NOMAD system construction"""
    
    def __init__(self):
        self._models = []
        self._config = NOMADConfig(role_model_name="")
        self._class_names = []
        self._initial_priors = None
        self._farml_registry = None
        
    def with_models(self, models: List[ModelConfig]) -> 'NOMADBuilder':
        """Add model configurations"""
        self._models = [Model(config) for config in models]
        return self
        
    def with_farml_registry(self, registry: FARMLModelRegistry) -> 'NOMADBuilder':
        """Connect to FARML model registry"""
        self._farml_registry = registry
        return self
        
    def with_role_model(self, role_model_name: str) -> 'NOMADBuilder':
        """Set the role model (highest quality benchmark)"""
        self._config.role_model_name = role_model_name
        return self
        
    def with_epsilon(self, epsilon: float) -> 'NOMADBuilder':
        """Set quality tolerance (0.0 to 1.0)"""
        self._config.epsilon = epsilon
        return self
        
    def with_safety_check(self, safety_type: SafetyCheckType) -> 'NOMADBuilder':
        """Set chain safety check type"""
        self._config.safety_check_type = safety_type
        return self
        
    def with_quality_metric(self, metric: QualityMetric) -> 'NOMADBuilder':
        """Set quality metric for evaluation"""
        self._config.quality_metric = metric
        return self
        
    def with_class_names(self, class_names: List[str]) -> 'NOMADBuilder':
        """Set target class names"""
        self._class_names = class_names
        return self
        
    def with_initial_priors(self, priors: Dict[str, float]) -> 'NOMADBuilder':
        """Set initial class probability priors"""
        self._initial_priors = priors
        return self
        
    def with_adaptive_config(self, adaptive_config: AdaptiveConfig) -> 'NOMADBuilder':
        """Configure adaptive distribution tracking"""
        self._config.adaptive_config = adaptive_config
        return self
        
    def enable_dependent_models(self, enable: bool = True) -> 'NOMADBuilder':
        """Enable support for dependent models (e.g., early-exit DNNs)"""
        self._config.enable_dependent_models = enable
        return self
        
    def build(self) -> NOMADEngine:
        """Build the NOMAD engine"""
        if not self._config.role_model_name:
            raise ValueError("Role model must be specified")
        if not self._class_names:
            raise ValueError("Class names must be specified")
        if not self._models:
            raise ValueError("At least one model must be provided")
            
        return NOMADEngine(
            models=self._models,
            config=self._config,
            class_names=self._class_names,
            initial_class_priors=self._initial_priors,
            farml_registry=self._farml_registry
        )

# ============================================================================
# FARML Workflow Integration
# ============================================================================

class NOMADWorkflowComponent:
    """NOMAD component for FARML workflow automation"""
    
    def __init__(self, 
                 workflow_engine: FARMLWorkflowEngine,
                 data_connector: FARMLDataConnector,
                 model_registry: FARMLModelRegistry):
        self.workflow_engine = workflow_engine
        self.data_connector = data_connector
        self.model_registry = model_registry
        
    def create_training_pipeline(self, 
                                models: List[ModelConfig],
                                data_source_config: Dict[str, Any],
                                training_config: Dict[str, Any]) -> str:
        """Create a FARML pipeline for training NOMAD models"""
        pipeline_config = {
            "type": "nomad_training",
            "models": models,
            "data_source": data_source_config,
            "training": training_config
        }
        return self.workflow_engine.create_pipeline(pipeline_config)
    
    def create_inference_pipeline(self,
                                 nomad_config: NOMADConfig,
                                 data_source_config: Dict[str, Any],
                                 output_config: Dict[str, Any]) -> str:
        """Create a FARML pipeline for NOMAD inference"""
        pipeline_config = {
            "type": "nomad_inference", 
            "nomad_config": nomad_config,
            "data_source": data_source_config,
            "output": output_config
        }
        return self.workflow_engine.create_pipeline(pipeline_config)
    
    def register_nomad_system(self, nomad_engine: NOMADEngine) -> str:
        """Register a complete NOMAD system in FARML registry"""
        # Placeholder for integration with FARML model registry
        pass

# ============================================================================
# Convenience Functions and Utilities
# ============================================================================

def create_model_config(name: str, 
                       model_type: str, 
                       cost: float,
                       params: Optional[Dict[str, Any]] = None,
                       **kwargs) -> ModelConfig:
    """Convenience function to create model configurations"""
    return ModelConfig(
        name=name,
        model_type=model_type,
        params=params or {},
        cost=cost,
        **kwargs
    )

def load_nomad_from_config(config_path: str, 
                          farml_registry: Optional[FARMLModelRegistry] = None) -> NOMADEngine:
    """Load NOMAD system from configuration file"""
    # Implementation would parse JSON/YAML config and build NOMAD system
    pass

def train_models_on_data(model_configs: List[ModelConfig], 
                        data: pd.DataFrame,
                        target_column: str = "target",
                        test_size: float = 0.3) -> List[Model]:
    """Train multiple models on provided data"""
    # Implementation would use existing training logic
    pass

# ============================================================================
# Example Usage and Quick Start
# ============================================================================

def create_simple_nomad_example():
    """Example showing simple NOMAD usage"""
    
    # Define models with different cost-quality tradeoffs
    models = [
        create_model_config("fast_tree", "DecisionTreeClassifier", cost=1.0, 
                           params={"max_depth": 3}),
        create_model_config("random_forest", "RandomForestClassifier", cost=5.0,
                           params={"n_estimators": 50}),
        create_model_config("gradient_boost", "GradientBoostingClassifier", cost=15.0,
                           params={"n_estimators": 100})
    ]
    
    # Build NOMAD system
    nomad = (NOMADBuilder()
             .with_models(models)
             .with_role_model("gradient_boost")  # Highest quality model
             .with_epsilon(0.1)  # 10% quality tolerance
             .with_class_names(["class_A", "class_B", "class_C"])
             .with_safety_check(SafetyCheckType.CONSERVATIVE)
             .build())
    
    return nomad

# Export public API
__all__ = [
    # Core classes
    'NOMADEngine', 'Model', 'NOMADBuilder',
    
    # Configuration classes
    'NOMADConfig', 'ModelConfig', 'AdaptiveConfig',
    
    # Enums
    'SafetyCheckType', 'QualityMetric',
    
    # FARML integration
    'NOMADWorkflowComponent', 'FARMLDataConnector', 'FARMLModelRegistry', 'FARMLWorkflowEngine',
    
    # Utilities
    'create_model_config', 'load_nomad_from_config', 'train_models_on_data', 
    'create_simple_nomad_example'
]