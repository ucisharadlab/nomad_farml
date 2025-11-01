# NOMAD: Navigating Optimal Model Application to Datastreams

**Complete Documentation & User Guide**

---

## Table of Contents

1. [Overview & Introduction](#overview--introduction)
2. [Theoretical Foundation](#theoretical-foundation)
3. [System Architecture](#system-architecture)
4. [Installation & Setup](#installation--setup)
5. [Python API Reference](#python-api-reference)
6. [Web Application Guide](#web-application-guide)
7. [FARML Ecosystem Integration](#farml-ecosystem-integration)
8. [Advanced Configuration](#advanced-configuration)
9. [Examples & Tutorials](#examples--tutorials)
10. [Performance Optimization](#performance-optimization)
11. [Troubleshooting](#troubleshooting)
12. [Contributing & Extending](#contributing--extending)

---

## Overview & Introduction

### What is NOMAD?

NOMAD (Navigating Optimal Model Application to Datastreams) is an intelligent framework for cost-efficient, real-time multi-class classification. It addresses the fundamental challenge in machine learning deployments: **How can we achieve the performance of our most powerful models while only paying the cost of our cheapest ones?**

NOMAD dynamically sequences machine learning models with varying cost-quality tradeoffs, using cheaper models as initial filters and proceeding to more expensive models only when necessary. This approach can achieve significant computational savings (often 2-5x speedup) while maintaining classification quality comparable to the most accurate model.

### Key Features

- **Cost-Quality Optimization**: Automatically balances computational cost against prediction accuracy
- **Dynamic Model Chaining**: Intelligently sequences models based on real-time performance and class distributions
- **Adaptive Learning**: Automatically detects and adapts to changes in data distribution using ARIMA models and Page-Hinkley tests
- **Safety Guarantees**: Provides formal guarantees that output quality stays within ε-tolerance of a role model
- **Multi-Modal Support**: Works with any scikit-learn compatible models and supports custom model implementations
- **Real-Time Processing**: Designed for streaming data with event-by-event processing
- **Dependent Model Support**: Handles complex model relationships like early-exit DNNs and stacking ensembles

### Use Cases

- **Real-time cybersecurity**: Network intrusion detection with varying threat levels
- **IoT data processing**: Efficient classification of sensor data streams
- **Content moderation**: Scalable text/image classification with quality guarantees
- **Financial fraud detection**: Real-time transaction analysis with cost constraints
- **Edge computing**: Resource-constrained environments requiring intelligent model selection

---

## Theoretical Foundation

### Core Algorithm

NOMAD is based on predicate-ordering techniques from database query processing, adapted for multi-class machine learning. The system maintains:

1. **Model Portfolio**: A set of models M = {M₁, M₂, ..., Mₙ} with increasing costs
2. **Role Model**: A benchmark model Mᵣ representing the quality standard
3. **Exit Classes**: For each model Mᵢ, a set of classes where it provides ε-comparable quality to Mᵣ
4. **Utility Function**: U(Mᵢ) = Σ P(Cⱼ) / cost(Mᵢ) for Cⱼ in exit classes of Mᵢ

### Key Concepts

#### ε-Comparability
A model Mᵢ provides ε-comparable quality to Mⱼ for class Cₖ if:
```
Q(Mᵢ, Cₖ) ≥ Q(Mⱼ, Cₖ) × (1 - ε)
```
Where Q(M, C) is the quality metric (F1-score, precision, etc.) of model M on class C.

#### Chain Safety
NOMAD ensures that any model chain maintains quality guarantees through two safety bounds:

**Conservative Bound**: Uses recall-based estimates
```
Chain_Recall_j = Π Recall(Mₖ, Cⱼ) for all models in chain
```

**Relaxed Bound**: Uses full confusion matrix analysis
```
Chain_Quality_j = Π (1 - P_misclassify_to_exit(Mₖ, Cⱼ)) × Quality(M_exit, Cⱼ)
```

#### Adaptive Distribution Tracking
NOMAD uses a two-tier adaptation strategy:

1. **Per-Event Updates**: Lightweight ARIMA model updates after each event
2. **Drift Detection**: Page-Hinkley test triggers full model retraining when distribution shifts

### Algorithm Workflow

```python
def nomad_classify_event(event, models, current_priors):
    available_models = models.copy()
    realized_chain = []
    
    while available_models:
        # Select model with highest utility
        selected_model = max(available_models, key=lambda m: utility(m, current_priors))
        
        # Check chain safety
        if not is_chain_safe(selected_model, realized_chain):
            available_models.remove(selected_model)
            continue
            
        # Execute model
        prediction, probabilities = selected_model.predict(event)
        realized_chain.append(selected_model)
        
        # Check exit condition
        if prediction in selected_model.exit_classes:
            return prediction, realized_chain
            
        # Update beliefs and continue
        current_priors = update_beliefs(current_priors, probabilities)
        available_models.remove(selected_model)
    
    # Fallback to role model
    return role_model.predict(event), realized_chain + [role_model]
```

---

## System Architecture

### Component Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Frontend  │    │   Python API     │    │  Core Engine    │
│                 │    │                  │    │                 │
│ • React UI      │◄──►│ • NOMADEngine    │◄──►│ • Model Chain   │
│ • Visualizations│    │ • Model Configs  │    │ • Safety Checks │
│ • Interactions  │    │ • Builder API    │    │ • Adaptivity    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Flask Backend │    │  FARML Toolbox   │    │  Data Sources   │
│                 │    │                  │    │                 │
│ • REST API      │    │ • Workflow Mgmt  │    │ • CSV Files     │
│ • Model Training│    │ • Model Registry │    │ • Streams       │
│ • Simulation    │    │ • Data Connectors│    │ • Databases     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Core Components

#### 1. NOMAD Engine (`nomad_arima_backend.py`)
- **NOMADEngine**: Main orchestration class
- **Model**: Individual model wrapper with performance tracking
- **AdaptivePriorsManager**: Handles distribution tracking and drift detection
- **PageHinkleyTest**: Statistical change point detection
- **Safety validation**: Conservative and relaxed bounds checking

#### 2. Python API Library (`nomad_api.py`)
- **High-level API**: `NOMADBuilder` for easy system construction
- **Configuration classes**: Type-safe configuration management  
- **FARML integration**: Interfaces for workflow automation
- **Convenience functions**: One-line training and deployment

#### 3. Web Application
- **Backend**: Flask REST API for model management and simulation
- **Frontend**: React interface with real-time visualizations
- **Model management**: Add/remove models, upload custom classes
- **Simulation**: Interactive workload design and results visualization

#### 4. FARML Integration
- **Workflow automation**: Drag-and-drop pipeline construction
- **Model registry**: Centralized model artifact management
- **Data connectors**: Multiple data source integrations

---

## Installation & Setup

### Prerequisites

- **Python**: 3.7+ with pip
- **Node.js**: 16+ with npm (for web interface)
- **System Requirements**: 16GB+ RAM, Multi-core (Please see Evaluation Hardware in the paper for precise reproducability of results). Note: System performance will vary based on the CPU and RAM available to the system as well as the cost of each classifier inference on each system.

### Quick Start (Python API Only)

```bash
# Clone repository
git clone <repository-url>
cd nomad

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Quick test
python -c "
from nomad_api import create_simple_nomad_example
nomad = create_simple_nomad_example()
print('NOMAD system created successfully!')
"
```

### Complete Installation (Web Interface)

#### Backend Setup

1. **Prepare Python environment**:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Install additional ML dependencies**:
```bash
pip install scikit-learn pandas numpy flask flask-cors
pip install statsmodels  # For ARIMA models
```

3. **Start Flask backend**:
```bash
python app.py
# Backend runs at http://127.0.0.1:5001
```

#### Frontend Setup

1. **Create React project**:
```bash
npm create vite@latest nomad-frontend -- --template react
cd nomad-frontend
```

2. **Install dependencies**:
```bash
npm install && npm install -D tailwindcss @tailwindcss/vite react-chartjs-2 chart.js @heroicons/react
```

3. **Configure Tailwind CSS v4**:

Create `tailwind.config.js`:
```javascript
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: { extend: {} },
  plugins: [],
}
```

Update `vite.config.js`:
```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
})
```

Update `src/index.css`:
```css
@import 'tailwindcss';
```

4. **Start development server**:
```bash
npm run dev
# Frontend runs at http://localhost:5173
```

### Docker Installation (Optional)

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5001

CMD ["python", "app.py"]
```

```bash
# Build and run
docker build -t nomad-backend .
docker run -p 5001:5001 nomad-backend
```

### Dependencies Overview

#### Python Dependencies (`requirements.txt`)
```
flask>=2.0.0
flask-cors>=4.0.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
statsmodels>=0.12.0  # For ARIMA
scipy>=1.7.0
```

#### Node.js Dependencies (`package.json`)
```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-chartjs-2": "^5.2.0",
    "chart.js": "^4.3.0"
  },
  "devDependencies": {
    "@tailwindcss/vite": "^4.0.0",
    "@heroicons/react": "^2.0.0",
    "vite": "^4.4.5",
    "tailwindcss": "^4.0.0"
  }
}
```

---

## Python API Reference

### Quick Start Example

```python
from nomad_api import NOMADBuilder, ModelConfig, SafetyCheckType

# Define models with different cost-quality tradeoffs
models = [
    ModelConfig("fast_tree", "DecisionTreeClassifier", cost=1.0, 
                params={"max_depth": 3}),
    ModelConfig("random_forest", "RandomForestClassifier", cost=5.0,
                params={"n_estimators": 50}),
    ModelConfig("gradient_boost", "GradientBoostingClassifier", cost=15.0,
                params={"n_estimators": 100})
]

# Build NOMAD system
nomad = (NOMADBuilder()
         .with_models(models)
         .with_role_model("gradient_boost")
         .with_epsilon(0.1)  # 10% quality tolerance
         .with_class_names(["normal", "attack", "suspicious"])
         .with_safety_check(SafetyCheckType.CONSERVATIVE)
         .build())

# Train on your data
nomad.train(training_data, target_column="label")

# Classify events
prediction, cost, chain = nomad.classify_event(test_event)
print(f"Predicted: {prediction}, Cost: {cost}, Models used: {chain}")
```

### Core Classes

#### NOMADEngine

Main orchestration class for the NOMAD system.

**Constructor Parameters:**
- `models`: List of Model instances
- `config`: NOMADConfig instance
- `class_names`: List of target class names
- `initial_class_priors`: Optional dict of initial class probabilities

**Key Methods:**

```python
def classify_event(self, event_features: pd.DataFrame) -> Tuple[str, float, List[str]]:
    """
    Classify a single event using NOMAD strategy.
    
    Returns:
        - Predicted class name
        - Total computational cost incurred
        - List of models used in the chain
    """

def classify_stream(self, data_stream, batch_size=10, workload_phases=None):
    """
    Classify a stream of events with real-time adaptation.
    
    Args:
        data_stream: DataFrame with features and labels
        batch_size: Events per batch for progress reporting
        workload_phases: List of workload phase configurations
        
    Yields:
        Dictionary with batch results and performance metrics
    """

def add_model(self, model: Model) -> 'NOMADEngine':
    """Add a new model to the system and recalculate exit classes."""

def get_performance_summary(self) -> Dict[str, Any]:
    """Get current performance statistics including adaptive stats."""
```

#### NOMADBuilder

Fluent builder pattern for easy NOMAD construction.

```python
nomad = (NOMADBuilder()
         .with_models([model1, model2, model3])
         .with_role_model("best_model_name")
         .with_epsilon(0.15)
         .with_class_names(["class1", "class2"])
         .with_safety_check(SafetyCheckType.RELAXED)
         .with_adaptive_config(AdaptiveConfig(
             ph_threshold=75.0,
             enable_arima=True
         ))
         .enable_dependent_models(True)
         .build())
```

#### ModelConfig

Configuration for individual models.

```python
config = ModelConfig(
    name="my_model",
    model_type="RandomForestClassifier",
    cost=10.0,
    params={"n_estimators": 100, "max_depth": 10},
    is_custom=False,  # Set to True for custom model classes
    prerequisites=["dependency_model"]  # For dependent models
)
```

#### AdaptiveConfig

Configuration for adaptive distribution tracking.

```python
adaptive_config = AdaptiveConfig(
    ph_threshold=50.0,      # Page-Hinkley detection threshold
    ph_delta=0.005,         # Page-Hinkley delta parameter
    buffer_size=100,        # Event buffer size for retraining
    min_buffer_for_arima=20,  # Minimum events before ARIMA training
    arima_order=(1, 0, 1),  # ARIMA(p,d,q) parameters
    enable_arima=True       # Enable ARIMA-based adaptation
)
```

### High-Level Convenience Functions

#### Training from Data

```python
from nomad_api_implementation import train_nomad_from_data

# Train directly from DataFrame
nomad = train_nomad_from_data(
    data=df,
    model_configs=model_configs,
    role_model_name="best_model",
    target_column="label",
    epsilon=0.1,
    safety_check_type=SafetyCheckType.CONSERVATIVE
)
```

#### Loading from CSV

```python
from nomad_api_implementation import load_and_train_nomad

nomad = load_and_train_nomad(
    csv_path="training_data.csv",
    model_configs=model_configs,
    role_model_name="best_model",
    epsilon=0.1
)
```

### Custom Model Integration

#### Creating Custom Models

```python
# custom_model.py
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class MyCustomModel(BaseEstimator, ClassifierMixin):
    def __init__(self, param1=1.0):
        self.param1 = param1
    
    def fit(self, X, y):
        # Your training logic
        self.classes_ = np.unique(y)
        return self
    
    def predict(self, X):
        # Your prediction logic
        return np.random.choice(self.classes_, size=len(X))
    
    def predict_proba(self, X):
        # Your probability prediction logic (optional but recommended)
        n_classes = len(self.classes_)
        return np.random.dirichlet(np.ones(n_classes), size=len(X))
```

```python
# Using custom models
custom_config = ModelConfig(
    name="my_custom_model",
    model_type="MyCustomModel",  # Class name
    cost=8.0,
    params={"param1": 2.0},
    is_custom=True,
    module_name="custom_model",  # File name without .py
    class_name="MyCustomModel"
)
```

### Advanced Configuration

#### Dependent Models (Early-Exit DNNs, Stacking)

```python
# Define model dependencies
base_model = ModelConfig("base_cnn", "SimpleCNN", cost=5.0)
early_exit_1 = ModelConfig("early_exit_1", "EarlyExitCNN", cost=3.0, 
                          prerequisites=["base_cnn"])
early_exit_2 = ModelConfig("early_exit_2", "EarlyExitCNN", cost=4.0,
                          prerequisites=["base_cnn"])
full_model = ModelConfig("full_cnn", "FullCNN", cost=10.0,
                        prerequisites=["base_cnn"])

# Enable dependent model support
nomad = (NOMADBuilder()
         .with_models([base_model, early_exit_1, early_exit_2, full_model])
         .enable_dependent_models(True)
         .build())
```

#### Workload Phases for Complex Scenarios

```python
# Define varying workload phases
workload_phases = [
    {
        "duration": 1000,  # Number of events
        "target_distribution": {
            "normal": 0.8,
            "attack": 0.15, 
            "suspicious": 0.05
        }
    },
    {
        "duration": 500,
        "target_distribution": {
            "normal": 0.4,
            "attack": 0.5,
            "suspicious": 0.1
        }
    }
]

# Run simulation with phases
for result in nomad.classify_stream(test_data, workload_phases=workload_phases):
    if result["type"] == "phase_shift":
        print(f"Entering phase {result['phase_index']}")
    elif result["type"] == "batch_update":
        print(f"Processed {result['last_event_in_batch']} events")
```

---

## Web Application Guide

### Application Overview

The NOMAD Web Visualizer provides an interactive interface for exploring cost-quality tradeoffs in machine learning model deployment. It consists of several key sections:

1. **Model Management**: Add, configure, and remove models
2. **Data Upload & Training**: Train models on your datasets
3. **Simulation Configuration**: Configure NOMAD parameters and workloads
4. **Live Results**: Real-time visualization of the simulation
5. **Final Analysis**: Comprehensive results summary

### Step-by-Step Walkthrough

#### 1. Model Management

**Default Models**: The application starts with three pre-configured models representing different cost-quality tradeoffs:
- Decision Tree (Shallow): Low cost, basic accuracy
- Random Forest (Small): Medium cost, good accuracy  
- Gradient Boosting: High cost, best accuracy

**Adding Standard Models**:
1. Click "Add New Model" 
2. Enter a unique name
3. Set cost (abstract computational units)
4. Select model type from dropdown
5. Configure hyperparameters in the dynamic form
6. Click "Add Model"

**Adding Custom Models**:
1. Select "-- Custom Model (from file) --" 
2. Enter the exact Python class name
3. Upload a `.py` file with your scikit-learn compatible model
4. The model must implement `.fit()`, `.predict()`, and optionally `.predict_proba()`

**Model Configuration Examples**:

*Support Vector Machine*:
```
Name: svm_classifier
Cost: 12.0
Type: SVC
Parameters:
  - C: 1.0
  - kernel: rbf
  - gamma: scale
```

*Custom Neural Network*:
```python
# Upload file: neural_net.py
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier

class CustomNeuralNet(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layers=(100,), learning_rate=0.001):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            learning_rate_init=learning_rate
        )
    
    def fit(self, X, y):
        self.mlp.fit(X, y)
        return self
    
    def predict(self, X):
        return self.mlp.predict(X)
    
    def predict_proba(self, X):
        return self.mlp.predict_proba(X)
```

#### 2. Data Upload & Training

**Data Format Requirements**:
- CSV format with headers
- Target variable (class labels) in the **second-to-last column**
- Features in all other columns except the last
- No missing values in critical columns

**Sample Data Structure**:
```csv
feature1,feature2,feature3,target_class,metadata
1.2,0.5,2.1,normal,extra_info
0.8,1.2,1.5,attack,extra_info
2.1,0.3,0.9,suspicious,extra_info
```

**Training Process**:
1. Click "Choose File" and select your CSV
2. Click "Upload and Train"
3. System automatically:
   - Splits data (70% train, 30% test)
   - Trains all models in your portfolio
   - Evaluates performance on test set
   - Displays results in "Individual Model Performance" table

**Understanding Training Results**:
- **Accuracy**: Overall classification accuracy
- **F1-Score (Weighted)**: Weighted average F1-score across classes
- **Cost**: Computational cost per prediction
- **Best performers** become candidates for the Role Model

#### 3. Simulation Configuration

**Role Model Selection**:
The Role Model serves as the quality benchmark. Typically choose:
- Highest accuracy model from training results
- Most balanced high-performer
- Model representing your quality requirements

**Epsilon (ε) Configuration**:
- **0.05 (5%)**: Very strict quality requirements
- **0.10 (10%)**: Balanced cost-quality tradeoff
- **0.20 (20%)**: Aggressive cost savings
- **Higher values**: More cost savings, lower quality guarantee

**Safety Check Types**:
- **Conservative**: Uses recall-based estimates (stricter, safer)
- **Relaxed**: Uses full confusion matrix analysis (more optimistic)

**Workload Phase Design**:

*Steady State Workload*:
```
Duration: 2000 events
Distribution: 
  - normal: 80%
  - attack: 15%
  - suspicious: 5%
```

*Attack Scenario*:
```
Phase 1 (1000 events):
  - normal: 70%
  - attack: 25%
  - suspicious: 5%

Phase 2 (800 events):  
  - normal: 30%
  - attack: 60%
  - suspicious: 10%
```

*Gradual Shift*:
```
Phase 1: normal: 90%, attack: 8%, suspicious: 2%
Phase 2: normal: 70%, attack: 25%, suspicious: 5%  
Phase 3: normal: 50%, attack: 40%, suspicious: 10%
```

#### 4. Live Results Interpretation

**Key Visualizations**:

*Cumulative Cost Chart*:
- **NOMAD Line**: Should grow slower than Role Model
- **Steep increases**: Indicate periods where expensive models are needed
- **Plateau regions**: NOMAD found efficient model sequences

*Live Accuracy Chart*:
- **NOMAD fluctuations**: Normal adaptive behavior
- **Role Model line**: Static baseline to beat
- **Convergence**: NOMAD accuracy should approach Role Model over time

*Model Execution Frequency*:
- **Bar heights**: How often each model is used
- **Changes over time**: Adaptation to workload shifts
- **Ideal pattern**: Frequent use of cheaper models, strategic use of expensive ones

**Performance Indicators**:
- **Speedup > 2.0x**: Excellent cost savings
- **Speedup 1.5-2.0x**: Good optimization
- **Speedup < 1.3x**: Limited benefit (check ε or data characteristics)
- **Accuracy gap < ε**: Quality guarantee maintained

#### 5. Final Analysis

**Summary Statistics**:
```
NOMAD Performance:
  Overall Accuracy: 94.2%
  Average Cost per Event: 3.8 units
  Total Events Processed: 2000

Role Model Baseline:
  Overall Accuracy: 95.1%
  Cost per Event: 15.0 units
  
Performance Comparison:
  Accuracy Difference: -0.9% (within ε=10%)
  Cost Reduction: 74.7%
  Speedup: 3.9x
```

**Model Usage Breakdown**:
- Decision Tree: 45% of events
- Random Forest: 35% of events  
- Gradient Boost: 20% of events

**Interpretation Guidelines**:
- **High speedup + small accuracy loss**: NOMAD working optimally
- **Low speedup**: Data may not benefit from model chaining (homogeneous difficulty)
- **Accuracy loss > ε**: Potential configuration issue or data drift

### Troubleshooting Common Issues

#### Training Failures
```
Error: "Model training error: No numerical features found"
Solution: Ensure CSV has numeric columns for features
```

```
Error: "Cannot determine numeric labels for final evaluation"
Solution: Check target column format and class names
```

#### Simulation Issues
```
Warning: "No models available in dict"
Solution: Ensure at least one model trained successfully
```

```
Error: "Role model not found in provided models"
Solution: Select role model from successfully trained models only
```

#### Performance Issues
```
Observation: Very low speedup (<1.2x)
Possible Causes:
- ε too strict (try increasing to 0.15-0.20)
- Dataset too easy (all models perform similarly)
- Role model already cheapest/fastest
```

#### Data Issues
```
Error: "CSV file is empty" or "Must have at least two columns"
Solution: 
- Verify CSV format and structure
- Ensure proper headers
- Check file encoding (UTF-8 recommended)
```

---

## FARML Ecosystem Integration

### Overview

NOMAD is designed as a core component of the FARML (Fair, Accountable, Robust Machine Learning) Toolbox, providing specialized functionality for cost-efficient model deployment within a broader ML lifecycle management system.

### Integration Points

#### 1. Workflow Automation

```python
from nomad_api import NOMADWorkflowComponent

# Create workflow component
nomad_workflow = NOMADWorkflowComponent(
    workflow_engine=farml_workflow_engine,
    data_connector=farml_data_connector, 
    model_registry=farml_model_registry
)

# Create training pipeline
training_pipeline_id = nomad_workflow.create_training_pipeline(
    models=model_configs,
    data_source_config={
        "type": "streaming_kafka",
        "topic": "ml_training_data",
        "batch_size": 1000
    },
    training_config={
        "validation_split": 0.2,
        "cross_validation_folds": 5
    }
)

# Create inference pipeline  
inference_pipeline_id = nomad_workflow.create_inference_pipeline(
    nomad_config=nomad_config,
    data_source_config={
        "type": "real_time_stream", 
        "endpoint": "kafka://events:9092"
    },
    output_config={
        "type": "database",
        "connection": "postgres://results_db"
    }
)
```

#### 2. Model Registry Integration

```python
# Register trained NOMAD system
class FARMLModelRegistryImpl(FARMLModelRegistry):
    def register_model(self, model_config, model_artifact):
        """Register individual model in FARML registry"""
        return self.registry_client.upload_model(
            name=model_config.name,
            version="1.0.0", 
            artifact=model_artifact,
            metadata={
                "cost": model_config.cost,
                "type": model_config.model_type,
                "parameters": model_config.params,
                "performance_metrics": model_artifact.avg_metrics
            }
        )
    
    def register_nomad_system(self, nomad_engine):
        """Register complete NOMAD system as composite model"""
        system_metadata = {
            "type": "nomad_system",
            "role_model": nomad_engine.config.role_model_name,
            "epsilon": nomad_engine.config.epsilon,
            "models": list(nomad_engine.models.keys()),
            "class_names": nomad_engine.class_names,
            "performance": nomad_engine.get_performance_summary()
        }
        
        return self.registry_client.upload_composite_model(
            name=f"nomad_system_{datetime.now().isoformat()}",
            components=nomad_engine.models,
            orchestrator=nomad_engine,
            metadata=system_metadata
        )
```

#### 3. Data Connector Integration

```python
class FARMLDataConnectorImpl(FARMLDataConnector):
    def stream_data(self, source_config):
        """Stream data for NOMAD real-time processing"""
        if source_config["type"] == "kafka":
            from kafka import KafkaConsumer
            consumer = KafkaConsumer(
                source_config["topic"],
                bootstrap_servers=source_config["servers"],
                value_deserializer=lambda m: json.loads(m.decode('utf-8'))
            )
            
            for message in consumer:
                yield pd.DataFrame([message.value])
                
        elif source_config["type"] == "database_stream":
            # Implement database streaming logic
            pass
```

### Drag-and-Drop Pipeline Construction

The FARML Workflow Automation system supports visual pipeline construction where NOMAD appears as a specialized component:

```
Data Source → Data Preprocessing → NOMAD Training → Model Evaluation
     ↓                ↓                   ↓              ↓
[File Upload]   [Feature Scaling]   [Multi-Model]   [Metrics Display]
[Stream Input]  [Encoding]          [Training]      [Validation]
[Database]      [Selection]         [Cost Config]   [Reporting]

                              ↓
                    
Live Data → NOMAD Inference → Actions/Alerts → Results Storage
    ↓             ↓                ↓                ↓
[Real-time]   [Model Chain]    [Notifications]  [Dashboard]
[Batch]       [Adaptation]     [Decisions]      [Analytics]
[API]         [Monitoring]     [Routing]        [Archival]
```

### Configuration Management

#### Pipeline Templates

```json
{
  "pipeline_type": "nomad_classification",
  "components": [
    {
      "id": "data_source",
      "type": "farml_data_connector",
      "config": {
        "source_type": "streaming",
        "format": "json"
      }
    },
    {
      "id": "nomad_core",
      "type": "nomad_engine", 
      "config": {
        "role_model": "gradient_boost",
        "epsilon": 0.1,
        "models": [
          {"name": "fast_tree", "cost": 1.0},
          {"name": "svm", "cost": 8.0},
          {"name": "gradient_boost", "cost": 15.0}
        ]
      },
      "inputs": ["data_source.output"],
      "outputs": ["predictions", "costs", "model_chain"]
    },
    {
      "id": "results_processor",
      "type": "farml_results_handler",
      "config": {
        "storage": "time_series_db",
        "alerts": {
          "accuracy_threshold": 0.90,
          "cost_threshold": 10.0
        }
      },
      "inputs": ["nomad_core.predictions", "nomad_core.costs"]
    }
  ]
}
```

#### Multi-Stage Automation

```python
# Complex FARML pipeline with NOMAD
pipeline_definition = {
    "stages": [
        {
            "name": "data_ingestion",
            "components": ["kafka_connector", "data_validator"]
        },
        {
            "name": "preprocessing", 
            "components": ["feature_scaler", "encoder", "selector"]
        },
        {
            "name": "nomad_training",
            "components": ["nomad_trainer", "hyperparameter_tuner"]
        },
        {
            "name": "validation",
            "components": ["cross_validator", "performance_analyzer"]
        },
        {
            "name": "deployment",
            "components": ["nomad_inference_engine", "model_monitor"]
        }
    ],
    "schedule": "continuous",
    "resource_constraints": {
        "max_cost_per_event": 12.0,
        "min_accuracy": 0.92
    }
}
```

### Monitoring & Observability

```python
class NOMADMonitor:
    def __init__(self, nomad_engine, farml_monitor):
        self.nomad = nomad_engine
        self.monitor = farml_monitor
        
    def track_performance(self):
        """Continuous performance tracking"""
        return {
            "cost_efficiency": self.calculate_cost_efficiency(),
            "quality_maintenance": self.check_quality_bounds(),
            "adaptation_effectiveness": self.measure_adaptation(),
            "model_utilization": self.get_model_usage_stats(),
            "drift_detection": self.get_drift_alerts()
        }
    
    def generate_alerts(self, performance_data):
        """Generate alerts for FARML monitoring system"""
        alerts = []
        
        if performance_data["cost_efficiency"] < 1.5:
            alerts.append({
                "level": "warning",
                "message": "NOMAD cost efficiency below expected threshold"
            })
            
        if performance_data["quality_maintenance"] > self.nomad.config.epsilon:
            alerts.append({
                "level": "critical", 
                "message": "Quality degradation exceeds epsilon tolerance"
            })
            
        return alerts
```

---

## Advanced Configuration

### Hyperparameter Tuning

#### Epsilon Optimization

```python
def find_optimal_epsilon(nomad_system, test_data, cost_budget):
    """Find optimal epsilon for given cost budget"""
    epsilons = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    results = []
    
    for eps in epsilons:
        nomad_system.config.epsilon = eps
        nomad_system._determine_exit_classes()
        
        total_cost = 0
        correct_predictions = 0
        
        for _, row in test_data.iterrows():
            event_df = pd.DataFrame([row.drop('target')])
            pred, cost, _ = nomad_system.classify_event(event_df)
            
            total_cost += cost
            if pred == row['target']:
                correct_predictions += 1
        
        avg_cost = total_cost / len(test_data)
        accuracy = correct_predictions / len(test_data)
        
        results.append({
            'epsilon': eps,
            'avg_cost': avg_cost,
            'accuracy': accuracy,
            'within_budget': avg_cost <= cost_budget
        })
    
    # Find best epsilon within budget
    valid_results = [r for r in results if r['within_budget']]
    return max(valid_results, key=lambda x: x['accuracy']) if valid_results else None
```

#### Adaptive Configuration Tuning

```python
def tune_adaptive_parameters(historical_data, drift_points):
    """Tune Page-Hinkley parameters based on historical drift data"""
    from scipy.optimize import minimize
    
    def objective(params):
        ph_threshold, ph_delta = params
        
        # Simulate drift detection with these parameters
        detector = PageHinkleyTest(threshold=ph_threshold, delta=ph_delta)
        
        false_positives = 0
        false_negatives = 0
        detection_delays = []
        
        for i, (value, is_drift_point) in enumerate(historical_data):
            drift_detected = detector.update(value)
            
            if drift_detected and not is_drift_point:
                false_positives += 1
            elif is_drift_point and not drift_detected:
                false_negatives += 1
            elif drift_detected and is_drift_point:
                # Calculate detection delay
                delay = i - drift_points[is_drift_point]
                detection_delays.append(delay)
        
        # Objective: minimize false positives + false negatives + avg delay
        avg_delay = np.mean(detection_delays) if detection_delays else 0
        return false_positives + false_negatives + avg_delay * 0.1
    
    # Optimize
    result = minimize(
        objective, 
        x0=[50.0, 0.005],  # Initial guess
        bounds=[(10.0, 200.0), (0.001, 0.05)],  # Bounds
        method='L-BFGS-B'
    )
    
    return {
        'ph_threshold': result.x[0],
        'ph_delta': result.x[1],
        'optimization_success': result.success
    }
```

### Custom Safety Bounds

```python
class CustomSafetyChecker:
    """Custom implementation of chain safety checking"""
    
    def __init__(self, epsilon, class_weights=None):
        self.epsilon = epsilon
        self.class_weights = class_weights or {}
        
    def check_weighted_safety(self, chain, class_name):
        """Safety check with class-specific weights"""
        weight = self.class_weights.get(class_name, 1.0)
        adjusted_epsilon = self.epsilon * weight
        
        # Calculate chain quality with adjusted epsilon
        projected_quality = self.calculate_chain_quality(chain, class_name)
        role_quality = self.get_role_quality(class_name)
        
        return projected_quality >= role_quality * (1 - adjusted_epsilon)
        
    def check_temporal_safety(self, chain, class_name, time_window):
        """Safety check considering recent performance trends"""
        recent_performance = self.get_recent_performance(
            chain, class_name, time_window
        )
        
        # Adjust epsilon based on recent trends
        if recent_performance['trend'] == 'declining':
            safety_epsilon = self.epsilon * 0.8  # Stricter
        else:
            safety_epsilon = self.epsilon * 1.2  # More lenient
            
        return self.check_standard_safety(chain, class_name, safety_epsilon)
```

### Multi-Objective Optimization

```python
def pareto_optimal_configuration(models, test_data, objectives=['cost', 'accuracy', 'fairness']):
    """Find Pareto-optimal NOMAD configurations"""
    from scipy.optimize import differential_evolution
    import numpy as np
    
    def evaluate_configuration(config):
        """Evaluate configuration on multiple objectives"""
        epsilon, ph_threshold, ph_delta = config
        
        # Configure NOMAD system
        nomad_config = NOMADConfig(
            role_model_name="best_model",
            epsilon=epsilon,
            adaptive_config=AdaptiveConfig(
                ph_threshold=ph_threshold,
                ph_delta=ph_delta
            )
        )
        
        nomad = create_nomad_system(models, nomad_config)
        
        # Evaluate on multiple objectives
        results = []
        total_cost = 0
        predictions = []
        true_labels = []
        
        for _, row in test_data.iterrows():
            event_df = pd.DataFrame([row.drop('target')])
            pred, cost, _ = nomad.classify_event(event_df)
            
            total_cost += cost
            predictions.append(pred)
            true_labels.append(row['target'])
        
        # Calculate objectives
        avg_cost = total_cost / len(test_data)
        accuracy = accuracy_score(true_labels, predictions)
        fairness = calculate_fairness_metric(true_labels, predictions)
        
        return [avg_cost, 1-accuracy, 1-fairness]  # Minimize all
    
    # Multi-objective optimization
    bounds = [
        (0.05, 0.3),    # epsilon
        (10.0, 200.0),  # ph_threshold  
        (0.001, 0.05)   # ph_delta
    ]
    
    # Use NSGA-II or similar multi-objective algorithm
    pareto_solutions = []
    
    # Simplified: evaluate grid of configurations
    for epsilon in [0.1, 0.15, 0.2]:
        for ph_thresh in [50, 100, 150]:
            for ph_delta in [0.005, 0.01, 0.02]:
                config = [epsilon, ph_thresh, ph_delta]
                objectives = evaluate_configuration(config)
                pareto_solutions.append({
                    'config': config,
                    'objectives': objectives
                })
    
    return pareto_solutions
```

### Performance Optimization

#### Model Caching

```python
class CachedNOMADEngine(NOMADEngine):
    """NOMAD engine with intelligent model caching"""
    
    def __init__(self, *args, cache_size=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.prediction_cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        
    def classify_event_cached(self, event_features):
        """Classify with caching support"""
        # Create cache key from features
        cache_key = self._create_cache_key(event_features)
        
        if cache_key in self.prediction_cache:
            self.cache_hits += 1
            return self.prediction_cache[cache_key]
        
        # Not in cache, compute prediction
        result = self.classify_event(event_features)
        
        # Add to cache
        if len(self.prediction_cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.prediction_cache))
            del self.prediction_cache[oldest_key]
            
        self.prediction_cache[cache_key] = result
        self.cache_misses += 1
        
        return result
        
    def _create_cache_key(self, event_features):
        """Create hashable cache key from features"""
        return tuple(event_features.iloc[0].round(6))  # Round for stability
        
    def get_cache_stats(self):
        """Get cache performance statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            'hit_rate': hit_rate,
            'cache_size': len(self.prediction_cache),
            'total_requests': total
        }
```

#### Batch Processing Optimization

```python
def optimize_batch_processing(nomad_engine, data_stream, batch_sizes=[1, 10, 50, 100]):
    """Find optimal batch size for stream processing"""
    results = {}
    
    for batch_size in batch_sizes:
        start_time = time.time()
        total_cost = 0
        processed_events = 0
        
        # Process in batches
        for batch in chunk_dataframe(data_stream, batch_size):
            batch_predictions = []
            batch_costs = []
            
            for _, row in batch.iterrows():
                event_df = pd.DataFrame([row.drop('target')])
                pred, cost, _ = nomad_engine.classify_event(event_df)
                
                batch_predictions.append(pred)
                batch_costs.append(cost)
                processed_events += 1
            
            total_cost += sum(batch_costs)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        results[batch_size] = {
            'processing_time': processing_time,
            'events_per_second': processed_events / processing_time,
            'avg_cost': total_cost / processed_events,
            'total_events': processed_events
        }
    
    # Find optimal batch size
    optimal_batch = max(results.keys(), key=lambda k: results[k]['events_per_second'])
    return optimal_batch, results
```

#### Memory Optimization

```python
class MemoryOptimizedNOMAD(NOMADEngine):
    """Memory-optimized NOMAD for resource-constrained environments"""
    
    def __init__(self, *args, max_memory_mb=512, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_memory = max_memory_mb * 1024 * 1024  # Convert to bytes
        self.model_cache = {}
        
    def lazy_load_model(self, model_name):
        """Load model only when needed"""
        if model_name not in self.model_cache:
            # Check memory usage
            current_memory = self.get_memory_usage()
            if current_memory + self.estimate_model_size(model_name) > self.max_memory:
                self.evict_least_used_model()
            
            # Load model
            model = self.models[model_name]
            self.model_cache[model_name] = {
                'model': model,
                'last_used': time.time(),
                'usage_count': 0
            }
        
        # Update usage stats
        self.model_cache[model_name]['last_used'] = time.time()
        self.model_cache[model_name]['usage_count'] += 1
        
        return self.model_cache[model_name]['model']
        
    def evict_least_used_model(self):
        """Evict least recently used model from cache"""
        if not self.model_cache:
            return
            
        lru_model = min(
            self.model_cache.keys(),
            key=lambda k: self.model_cache[k]['last_used']
        )
        del self.model_cache[lru_model]
        
    def get_memory_usage(self):
        """Estimate current memory usage"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss
```

---

## Examples & Tutorials

### Tutorial 1: Basic NOMAD System

**Objective**: Create a simple NOMAD system for binary classification with three models of different complexities.

```python
import pandas as pd
from nomad_api import NOMADBuilder, ModelConfig

# Step 1: Prepare sample data
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
data['target'] = y

# Step 2: Define model portfolio
models = [
    ModelConfig("simple_tree", "DecisionTreeClassifier", cost=1.0,
                params={"max_depth": 3, "random_state": 42}),
    ModelConfig("balanced_forest", "RandomForestClassifier", cost=5.0, 
                params={"n_estimators": 50, "random_state": 42}),
    ModelConfig("complex_boost", "GradientBoostingClassifier", cost=15.0,
                params={"n_estimators": 100, "random_state": 42})
]

# Step 3: Build NOMAD system
nomad = (NOMADBuilder()
         .with_models(models)
         .with_role_model("complex_boost")  # Best model as benchmark
         .with_epsilon(0.15)  # 15% quality tolerance
         .with_class_names(['class_0', 'class_1'])
         .build())

# Step 4: Train system
from nomad_api_implementation import train_nomad_from_data
nomad = train_nomad_from_data(data, models, "complex_boost", epsilon=0.15)

# Step 5: Test single prediction
test_event = data.iloc[[0]].drop('target', axis=1)
prediction, cost, chain = nomad.classify_event(test_event)
print(f"Prediction: {prediction}, Cost: {cost}, Models used: {chain}")

# Step 6: Evaluate on test stream
test_data = data.iloc[800:].copy()  # Last 200 samples
results = []

for result in nomad.classify_stream(test_data):
    if result["type"] == "summary":
        results.append(result)
        break

print(f"NOMAD Accuracy: {results[0]['nomad_overall_metrics']['accuracy']:.3f}")
print(f"Average Cost: {results[0]['nomad_overall_metrics']['average_cost']:.2f}")
print(f"Role Model Cost: {results[0]['role_model_performance']['cost']}")
```

### Tutorial 2: Multi-Class Cybersecurity Classification

**Objective**: Build a NOMAD system for network intrusion detection with multiple attack types.

```python
# Step 1: Load cybersecurity dataset (example structure)
data = pd.read_csv('network_traffic.csv')
# Columns: duration, protocol_type, service, flag, src_bytes, dst_bytes, ..., attack_type

# Step 2: Define specialized models for cybersecurity
models = [
    # Fast screening model
    ModelConfig("fast_screen", "DecisionTreeClassifier", cost=0.5,
                params={"max_depth": 5, "min_samples_split": 100}),
    
    # Pattern recognition model  
    ModelConfig("pattern_svm", "SVC", cost=3.0,
                params={"kernel": "rbf", "C": 1.0, "probability": True}),
    
    # Statistical anomaly detection
    ModelConfig("isolation_forest", "IsolationForest", cost=2.0,
                params={"contamination": 0.1, "random_state": 42}),
    
    # Deep analysis model
    ModelConfig("ensemble_model", "GradientBoostingClassifier", cost=12.0,
                params={"n_estimators": 200, "learning_rate": 0.1})
]

# Step 3: Configure for cybersecurity scenario
nomad = (NOMADBuilder()
         .with_models(models)
         .with_role_model("ensemble_model")
         .with_epsilon(0.1)  # Strict quality for security
         .with_class_names(['normal', 'dos', 'probe', 'r2l', 'u2r'])
         .with_safety_check(SafetyCheckType.CONSERVATIVE)  # Conservative for security
         .build())

# Step 4: Train with cybersecurity data
nomad = train_nomad_from_data(data, models, "ensemble_model", 
                             target_column="attack_type", epsilon=0.1)

# Step 5: Simulate varying threat scenarios
threat_scenarios = [
    {
        "name": "normal_operations", 
        "duration": 1000,
        "target_distribution": {"normal": 0.95, "dos": 0.02, "probe": 0.02, "r2l": 0.005, "u2r": 0.005}
    },
    {
        "name": "dos_attack",
        "duration": 500, 
        "target_distribution": {"normal": 0.60, "dos": 0.35, "probe": 0.03, "r2l": 0.01, "u2r": 0.01}
    },
    {
        "name": "advanced_persistent_threat",
        "duration": 300,
        "target_distribution": {"normal": 0.70, "dos": 0.05, "probe": 0.10, "r2l": 0.10, "u2r": 0.05}
    }
]

# Step 6: Run threat simulation
test_data = data.iloc[8000:].copy()  # Test set
performance_log = []

for result in nomad.classify_stream(test_data, workload_phases=threat_scenarios):
    if result["type"] == "phase_shift":
        print(f"\n=== Entering {threat_scenarios[result['phase_index']]['name']} ===")
        
    elif result["type"] == "batch_update":
        performance_log.append({
            'events_processed': result['last_event_in_batch'],
            'cumulative_cost': result['cumulative_cost'],
            'live_accuracy': result['live_nomad_accuracy'],
            'model_usage': result['model_run_counts_snapshot']
        })
        
        if result['last_event_in_batch'] % 500 == 0:
            print(f"Processed {result['last_event_in_batch']} events, "
                  f"Accuracy: {result['live_nomad_accuracy']:.3f}, "
                  f"Avg Cost: {result['cumulative_cost']/result['last_event_in_batch']:.2f}")

# Step 7: Analyze security-specific metrics
final_results = [r for r in nomad.classify_stream(test_data, workload_phases=threat_scenarios) 
                if r["type"] == "summary"][0]

security_analysis = {
    'detection_accuracy': final_results['nomad_overall_metrics']['accuracy'],
    'false_positive_rate': calculate_false_positive_rate(final_results),
    'cost_efficiency': final_results['nomad_overall_metrics']['average_cost'],
    'model_utilization': final_results['final_model_run_counts'],
    'threat_response_time': calculate_avg_response_time(performance_log)
}

print("\n=== Security Analysis ===")
for metric, value in security_analysis.items():
    print(f"{metric}: {value}")
```

### Tutorial 3: Custom Model Integration

**Objective**: Integrate a custom neural network model with specific preprocessing requirements.

```python
# Step 1: Define custom model class
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.neural_network import MLPClassifier

class CustomNeuralNetwork(BaseEstimator, ClassifierMixin):
    """Custom neural network with specialized preprocessing"""
    
    def __init__(self, hidden_layers=(100, 50), dropout_rate=0.2, learning_rate=0.001):
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
    def fit(self, X, y):
        # Custom preprocessing
        X_processed = self._preprocess_features(X)
        
        # Initialize MLPClassifier with custom parameters
        self.mlp = MLPClassifier(
            hidden_layer_sizes=self.hidden_layers,
            learning_rate_init=self.learning_rate,
            max_iter=500,
            random_state=42
        )
        
        self.mlp.fit(X_processed, y)
        self.classes_ = self.mlp.classes_
        return self
    
    def predict(self, X):
        X_processed = self._preprocess_features(X)
        return self.mlp.predict(X_processed)
    
    def predict_proba(self, X):
        X_processed = self._preprocess_features(X) 
        return self.mlp.predict_proba(X_processed)
    
    def _preprocess_features(self, X):
        """Custom feature preprocessing"""
        # Example: Apply domain-specific transformations
        X_processed = X.copy()
        
        # Log transform for positive features
        positive_features = ['feature_1', 'feature_3', 'feature_5']
        for feat in positive_features:
            if feat in X_processed.columns:
                X_processed[feat] = np.log1p(X_processed[feat])
        
        # Normalize specific feature ranges
        range_features = ['feature_2', 'feature_4']
        for feat in range_features:
            if feat in X_processed.columns:
                X_processed[feat] = (X_processed[feat] - X_processed[feat].min()) / (X_processed[feat].max() - X_processed[feat].min())
        
        return X_processed

# Step 2: Save custom model to file
with open('custom_neural_net.py', 'w') as f:
    f.write('''
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.neural_network import MLPClassifier

class CustomNeuralNetwork(BaseEstimator, ClassifierMixin):
    # ... (copy the class definition above)
    pass
''')

# Step 3: Create model configuration for custom model
custom_model_config = ModelConfig(
    name="custom_neural_net",
    model_type="CustomNeuralNetwork",  # Class name
    cost=8.0,  # Medium-high cost
    params={
        "hidden_layers": (128, 64, 32),
        "dropout_rate": 0.3,
        "learning_rate": 0.001
    },
    is_custom=True,
    module_name="custom_neural_net",  # File name without .py
    class_name="CustomNeuralNetwork"
)

# Step 4: Build NOMAD system with custom model
models = [
    ModelConfig("logistic", "LogisticRegression", cost=1.0),
    ModelConfig("random_forest", "RandomForestClassifier", cost=4.0, 
                params={"n_estimators": 100}),
    custom_model_config,  # Our custom model
    ModelConfig("gradient_boost", "GradientBoostingClassifier", cost=12.0,
                params={"n_estimators": 150})
]

nomad = (NOMADBuilder()
         .with_models(models)
         .with_role_model("gradient_boost")
         .with_epsilon(0.12)
         .build())

# Step 5: Test the custom model integration
sample_data = pd.read_csv('your_data.csv')
nomad_trained = train_nomad_from_data(sample_data, models, "gradient_boost")

# Verify custom model is working
test_event = sample_data.iloc[[0]].drop('target', axis=1)
prediction, cost, chain = nomad_trained.classify_event(test_event)

print(f"Custom model integration successful!")
print(f"Prediction: {prediction}")
print(f"Models in chain: {chain}")
print(f"Custom model available: {'custom_neural_net' in nomad_trained.models}")
```

### Tutorial 4: Real-Time Streaming with Kafka

**Objective**: Deploy NOMAD for real-time stream processing with Kafka integration.

```python
import json
from kafka import KafkaConsumer, KafkaProducer
import threading
import time

class RealTimeNOMADProcessor:
    """Real-time NOMAD processor for Kafka streams"""
    
    def __init__(self, nomad_engine, input_topic, output_topic, kafka_config):
        self.nomad = nomad_engine
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.kafka_config = kafka_config
        self.running = False
        
        # Initialize Kafka consumer/producer
        self.consumer = KafkaConsumer(
            input_topic,
            bootstrap_servers=kafka_config['bootstrap_servers'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id=kafka_config.get('group_id', 'nomad_processor')
        )
        
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_config['bootstrap_servers'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        # Performance tracking
        self.stats = {
            'messages_processed': 0,
            'total_cost': 0.0,
            'start_time': None,
            'avg_processing_time': 0.0
        }
        
    def start_processing(self):
        """Start real-time processing"""
        self.running = True
        self.stats['start_time'] = time.time()
        
        print(f"Starting NOMAD real-time processor...")
        print(f"Input topic: {self.input_topic}")
        print(f"Output topic: {self.output_topic}")
        
        try:
            for message in self.consumer:
                if not self.running:
                    break
                    
                start_time = time.time()
                
                # Extract features from Kafka message
                event_data = message.value
                features_df = self._extract_features(event_data)
                
                # Process with NOMAD
                prediction, cost, model_chain = self.nomad.classify_event(features_df)
                
                # Create output message
                output_message = {
                    'timestamp': time.time(),
                    'input_event_id': event_data.get('id', 'unknown'),
                    'prediction': prediction,
                    'confidence': self._calculate_confidence(features_df),
                    'cost': cost,
                    'model_chain': model_chain,
                    'processing_time_ms': (time.time() - start_time) * 1000
                }
                
                # Send to output topic
                self.producer.send(self.output_topic, output_message)
                
                # Update stats
                self.stats['messages_processed'] += 1
                self.stats['total_cost'] += cost
                
                processing_time = time.time() - start_time
                self.stats['avg_processing_time'] = (
                    (self.stats['avg_processing_time'] * (self.stats['messages_processed'] - 1) + processing_time) 
                    / self.stats['messages_processed']
                )
                
                # Log progress
                if self.stats['messages_processed'] % 100 == 0:
                    self._log_progress()
                    
        except KeyboardInterrupt:
            print("Stopping processor...")
        finally:
            self.stop_processing()
    
    def _extract_features(self, event_data):
        """Extract features from Kafka message"""
        # Implement feature extraction logic based on your message format
        features = {
            'feature_1': event_data.get('metric1', 0.0),
            'feature_2': event_data.get('metric2', 0.0),
            # ... extract other features
        }
        return pd.DataFrame([features])
    
    def _calculate_confidence(self, features_df):
        """Calculate prediction confidence"""
        # Use the role model to get probability distribution
        probabilities = self.nomad.role_model.predict_proba(features_df)[0]
        return float(np.max(probabilities))
    
    def _log_progress(self):
        """Log processing statistics"""
        runtime = time.time() - self.stats['start_time']
        messages_per_sec = self.stats['messages_processed'] / runtime
        avg_cost = self.stats['total_cost'] / self.stats['messages_processed']
        
        print(f"Processed {self.stats['messages_processed']} messages")
        print(f"Rate: {messages_per_sec:.2f} msg/sec")
        print(f"Average cost: {avg_cost:.3f}")
        print(f"Average processing time: {self.stats['avg_processing_time']*1000:.2f} ms")
        print("---")
    
    def stop_processing(self):
        """Stop processing and cleanup"""
        self.running = False
        self.consumer.close()
        self.producer.close()
        print("NOMAD processor stopped")

# Usage example
if __name__ == "__main__":
    # Step 1: Set up NOMAD system (assume already trained)
    nomad_system = load_trained_nomad_system()
    
    # Step 2: Configure Kafka
    kafka_config = {
        'bootstrap_servers': ['localhost:9092'],
        'group_id': 'nomad_ml_processor'
    }
    
    # Step 3: Create processor
    processor = RealTimeNOMADProcessor(
        nomad_engine=nomad_system,
        input_topic='ml_events',
        output_topic='ml_predictions', 
        kafka_config=kafka_config
    )
    
    # Step 4: Start processing
    processor.start_processing()
```

### Tutorial 5: A/B Testing Framework

**Objective**: Compare different NOMAD configurations using an A/B testing framework.

```python
import uuid
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from scipy import stats

class TestVariant(Enum):
    CONTROL = "control"  # Baseline NOMAD config
    VARIANT_A = "variant_a"  # Different epsilon
    VARIANT_B = "variant_b"  # Different safety check
    VARIANT_C = "variant_c"  # Different adaptive config

@dataclass
class ABTestResult:
    variant: TestVariant
    accuracy: float
    avg_cost: float
    processing_time: float
    model_utilization: Dict[str, int]
    sample_size: int

class NOMADABTester:
    """A/B testing framework for NOMAD configurations"""
    
    def __init__(self, base_models, test_data, traffic_split=None):
        self.base_models = base_models
        self.test_data = test_data
        self.traffic_split = traffic_split or {
            TestVariant.CONTROL: 0.4,
            TestVariant.VARIANT_A: 0.2,
            TestVariant.VARIANT_B: 0.2,
            TestVariant.VARIANT_C: 0.2
        }
        self.results = {}
        
    def create_test_variants(self):
        """Create different NOMAD configurations for testing"""
        variants = {}
        
        # Control: Conservative configuration
        variants[TestVariant.CONTROL] = (NOMADBuilder()
                                        .with_models(self.base_models)
                                        .with_role_model("gradient_boost")
                                        .with_epsilon(0.1)
                                        .with_safety_check(SafetyCheckType.CONSERVATIVE)
                                        .build())
        
        # Variant A: More aggressive cost savings
        variants[TestVariant.VARIANT_A] = (NOMADBuilder()
                                          .with_models(self.base_models)
                                          .with_role_model("gradient_boost")
                                          .with_epsilon(0.2)  # Higher epsilon
                                          .with_safety_check(SafetyCheckType.CONSERVATIVE)
                                          .build())
        
        # Variant B: Relaxed safety checks
        variants[TestVariant.VARIANT_B] = (NOMADBuilder()
                                          .with_models(self.base_models)
                                          .with_role_model("gradient_boost") 
                                          .with_epsilon(0.1)
                                          .with_safety_check(SafetyCheckType.RELAXED)  # Relaxed
                                          .build())
        
        # Variant C: Optimized adaptive parameters
        adaptive_config = AdaptiveConfig(
            ph_threshold=75.0,  # More sensitive drift detection
            ph_delta=0.01,
            buffer_size=150
        )
        variants[TestVariant.VARIANT_C] = (NOMADBuilder()
                                          .with_models(self.base_models)
                                          .with_role_model("gradient_boost")
                                          .with_epsilon(0.15)
                                          .with_adaptive_config(adaptive_config)
                                          .build())
        
        return variants
    
    def run_ab_test(self, duration_events=2000):
        """Run A/B test across variants"""
        variants = self.create_test_variants()
        
        # Train all variants
        print("Training variants...")
        for variant, nomad_system in variants.items():
            print(f"Training {variant.value}...")
            # Assume training logic here
            
        print(f"\nStarting A/B test with {duration_events} events...")
        
        # Assign events to variants based on traffic split
        variant_assignments = self._assign_traffic(duration_events)
        
        # Process events for each variant
        for variant, nomad_system in variants.items():
            print(f"\nTesting {variant.value}...")
            
            # Get events assigned to this variant
            variant_events = variant_assignments[variant]
            variant_data = self.test_data.iloc[variant_events].copy()
            
            # Run NOMAD on variant data
            start_time = time.time()
            results = self._evaluate_variant(nomad_system, variant_data)
            processing_time = time.time() - start_time
            
            # Store results
            self.results[variant] = ABTestResult(
                variant=variant,
                accuracy=results['accuracy'],
                avg_cost=results['avg_cost'],
                processing_time=processing_time,
                model_utilization=results['model_usage'],
                sample_size=len(variant_events)
            )
            
            print(f"{variant.value} - Accuracy: {results['accuracy']:.3f}, "
                  f"Avg Cost: {results['avg_cost']:.2f}")
        
        return self.results
    
    def _assign_traffic(self, total_events):
        """Assign events to variants based on traffic split"""
        assignments = {variant: [] for variant in TestVariant}
        
        for event_idx in range(total_events):
            # Use hash-based assignment for consistent routing
            event_hash = hash(f"event_{event_idx}") % 100
            
            cumulative_prob = 0
            for variant, probability in self.traffic_split.items():
                cumulative_prob += probability * 100
                if event_hash < cumulative_prob:
                    assignments[variant].append(event_idx)
                    break
        
        return assignments
    
    def _evaluate_variant(self, nomad_system, test_data):
        """Evaluate a single variant"""
        total_cost = 0
        correct_predictions = 0
        model_usage = {}
        
        for _, row in test_data.iterrows():
            event_df = pd.DataFrame([row.drop('target')])
            true_label = row['target']
            
            prediction, cost, model_chain = nomad_system.classify_event(event_df)
            
            total_cost += cost
            if prediction == true_label:
                correct_predictions += 1
            
            # Track model usage
            for model_name in model_chain:
                model_usage[model_name] = model_usage.get(model_name, 0) + 1
        
        return {
            'accuracy': correct_predictions / len(test_data),
            'avg_cost': total_cost / len(test_data),
            'model_usage': model_usage
        }
    
    def analyze_results(self):
        """Perform statistical analysis of A/B test results"""
        if len(self.results) < 2:
            return "Need at least 2 variants for comparison"
        
        control_result = self.results[TestVariant.CONTROL]
        analysis = {
            'summary': {},
            'statistical_significance': {},
            'recommendations': []
        }
        
        # Compare each variant to control
        for variant, result in self.results.items():
            if variant == TestVariant.CONTROL:
                continue
            
            # Statistical tests
            accuracy_improvement = result.accuracy - control_result.accuracy
            cost_reduction = control_result.avg_cost - result.avg_cost
            cost_reduction_pct = (cost_reduction / control_result.avg_cost) * 100
            
            # Simple significance test (in practice, use proper statistical tests)
            accuracy_significant = abs(accuracy_improvement) > 0.01  # 1% threshold
            cost_significant = abs(cost_reduction_pct) > 5  # 5% threshold
            
            analysis['summary'][variant.value] = {
                'accuracy_change': accuracy_improvement,
                'cost_reduction_%': cost_reduction_pct,
                'processing_time_ratio': result.processing_time / control_result.processing_time
            }
            
            analysis['statistical_significance'][variant.value] = {
                'accuracy_significant': accuracy_significant,
                'cost_significant': cost_significant
            }
            
            # Generate recommendations
            if cost_reduction_pct > 10 and accuracy_improvement > -0.02:
                analysis['recommendations'].append(
                    f"{variant.value}: Excellent cost savings ({cost_reduction_pct:.1f}%) "
                    f"with minimal accuracy impact. Consider deploying."
                )
            elif accuracy_improvement > 0.02:
                analysis['recommendations'].append(
                    f"{variant.value}: Significant accuracy improvement ({accuracy_improvement:.3f}). "
                    f"Worth considering if cost increase is acceptable."
                )
        
        return analysis
    
    def generate_report(self):
        """Generate comprehensive A/B test report"""
        if not self.results:
            return "No test results available"
        
        report = []
        report.append("=== NOMAD A/B Test Report ===\n")
        
        # Results summary
        report.append("Test Results Summary:")
        report.append("-" * 50)
        for variant, result in self.results.items():
            report.append(f"{variant.value:12} | "
                         f"Accuracy: {result.accuracy:.3f} | "
                         f"Avg Cost: {result.avg_cost:.2f} | "
                         f"Sample Size: {result.sample_size}")
        
        # Statistical analysis
        analysis = self.analyze_results()
        report.append("\nStatistical Analysis:")
        report.append("-" * 50)
        
        for variant, summary in analysis.get('summary', {}).items():
            report.append(f"{variant}:")
            report.append(f"  Accuracy Change: {summary['accuracy_change']:+.3f}")
            report.append(f"  Cost Reduction: {summary['cost_reduction_%']:+.1f}%")
            report.append(f"  Speed Ratio: {summary['processing_time_ratio']:.2f}x")
        
        # Recommendations
        if analysis.get('recommendations'):
            report.append("\nRecommendations:")
            report.append("-" * 50)
            for rec in analysis['recommendations']:
                report.append(f"• {rec}")
        
        return "\n".join(report)

# Usage example
if __name__ == "__main__":
    # Load test data and models
    test_data = pd.read_csv('test_dataset.csv')
    base_models = [
        ModelConfig("tree", "DecisionTreeClassifier", cost=1.0),
        ModelConfig("forest", "RandomForestClassifier", cost=5.0),
        ModelConfig("gradient_boost", "GradientBoostingClassifier", cost=15.0)
    ]
    
    # Run A/B test
    ab_tester = NOMADABTester(base_models, test_data)
    results = ab_tester.run_ab_test(duration_events=1000)
    
    # Generate and print report
    report = ab_tester.generate_report()
    print(report)
    
    # Save detailed results
    with open('nomad_ab_test_results.json', 'w') as f:
        json.dump({variant.value: {
            'accuracy': result.accuracy,
            'avg_cost': result.avg_cost,
            'processing_time': result.processing_time,
            'sample_size': result.sample_size
        } for variant, result in results.items()}, f, indent=2)
```

---

## Performance Optimization

### Computational Performance

#### 1. Model Selection Optimization

The most expensive operation in NOMAD is the utility calculation and model selection at each step. Optimize this by:

```python
class OptimizedNOMADEngine(NOMADEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._utility_cache = {}
        self._sorted_models_cache = None
        
    def _calculate_utility_optimized(self, model, current_priors):
        """Optimized utility calculation with caching"""
        # Create cache key from current priors
        priors_key = tuple(sorted(current_priors.items()))
        cache_key = (model.config.name, priors_key)
        
        if cache_key in self._utility_cache:
            return self._utility_cache[cache_key]
        
        # Calculate utility
        exit_prob_sum = sum(current_priors.get(class_name, 0.0) 
                           for class_name in model.exit_classes)
        
        if model.config.cost == 0:
            utility = float('inf') if exit_prob_sum > 0 else 0.0
        else:
            utility = exit_prob_sum / model.config.cost
        
        # Cache result
        self._utility_cache[cache_key] = utility
        return utility
    
    def _select_next_model_optimized(self, available_models, current_priors):
        """Optimized model selection"""
        # Pre-sort models by cost (cheaper models first for tie-breaking)
        if self._sorted_models_cache is None:
            self._sorted_models_cache = sorted(
                available_models, 
                key=lambda m: (m.config.cost, m.config.name)
            )
        
        best_model = None
        best_utility = -1
        
        for model in self._sorted_models_cache:
            if model not in available_models:
                continue
                
            utility = self._calculate_utility_optimized(model, current_priors)
            
            if utility > best_utility:
                best_utility = utility
                best_model = model
                
            # Early termination for infinite utility
            if utility == float('inf'):
                break
        
        return best_model
```

#### 2. Safety Check Optimization

Safety checks can be computationally expensive. Optimize by pre-computing and caching:

```python
class CachedSafetyChecker:
    def __init__(self, nomad_engine):
        self.nomad = nomad_engine
        self.safety_cache = {}
        self.passthrough_cache = {}
        
    def check_chain_safety_cached(self, model_chain, class_name):
        """Cached chain safety checking"""
        chain_key = tuple(m.config.name for m in model_chain)
        cache_key = (chain_key, class_name)
        
        if cache_key in self.safety_cache:
            return self.safety_cache[cache_key]
        
        # Compute safety
        is_safe = self._compute_chain_safety(model_chain, class_name)
        self.safety_cache[cache_key] = is_safe
        
        return is_safe
    
    def _compute_chain_safety(self, model_chain, class_name):
        """Optimized safety computation"""
        # Find exit model for this class
        exit_model = None
        exit_index = -1
        
        for idx, model in enumerate(model_chain):
            if class_name in model.exit_classes:
                exit_model = model
                exit_index = idx
                break
        
        if exit_model is None:
            # Use role model as fallback
            exit_model = self.nomad.role_model
            exit_index = len(model_chain)
        
        # Calculate cumulative passthrough probability
        cumulative_passthrough = 1.0
        
        for i in range(exit_index):
            model = model_chain[i]
            passthrough_prob = self._get_passthrough_probability(model, class_name)
            cumulative_passthrough *= passthrough_prob
        
        # Check safety condition
        projected_quality = cumulative_passthrough * exit_model.get_quality(class_name, self.nomad.config.quality_metric.value)
        role_quality = self.nomad.role_model.get_quality(class_name, self.nomad.config.quality_metric.value)
        
        return projected_quality >= role_quality * (1 - self.nomad.config.epsilon)
    
    def _get_passthrough_probability(self, model, class_name):
        """Cached passthrough probability calculation"""
        cache_key = (model.config.name, class_name)
        
        if cache_key in self.passthrough_cache:
            return self.passthrough_cache[cache_key]
        
        # Use recall as conservative estimate
        passthrough_prob = model.get_quality(class_name, "recall")
        
        self.passthrough_cache[cache_key] = passthrough_prob
        return passthrough_prob
```

#### 3. Batch Processing Optimization

For high-throughput scenarios, implement efficient batch processing:

```python
class BatchNOMADProcessor:
    def __init__(self, nomad_engine, batch_size=100):
        self.nomad = nomad_engine
        self.batch_size = batch_size
        self.feature_cache = {}
        
    def process_batch(self, events_df):
        """Process multiple events efficiently"""
        results = []
        
        # Group events by feature similarity for cache efficiency
        event_groups = self._group_similar_events(events_df)
        
        for group_features, event_indices in event_groups.items():
            # Process group leader to determine likely model chain
            leader_idx = event_indices[0]
            leader_event = events_df.iloc[[leader_idx]]
            
            leader_pred, leader_cost, leader_chain = self.nomad.classify_event(leader_event)
            
            # Try to reuse the same model chain for similar events
            for event_idx in event_indices[1:]:
                event_df = events_df.iloc[[event_idx]]
                
                # Check if leader's chain works for this event
                if self._can_reuse_chain(event_df, leader_chain):
                    pred, cost = self._execute_fixed_chain(event_df, leader_chain)
                    results.append((event_idx, pred, cost, leader_chain))
                else:
                    # Fallback to full NOMAD processing
                    pred, cost, chain = self.nomad.classify_event(event_df)
                    results.append((event_idx, pred, cost, chain))
            
            # Add leader result
            results.append((leader_idx, leader_pred, leader_cost, leader_chain))
        
        return sorted(results, key=lambda x: x[0])  # Sort by original index
    
    def _group_similar_events(self, events_df):
        """Group events by feature similarity"""
        # Simple grouping by quantized features
        groups = {}
        
        for idx, row in events_df.iterrows():
            # Create feature signature (quantized)
            feature_sig = tuple(
                round(row[col], 2) if pd.api.types.is_numeric_dtype(events_df[col]) 
                else str(row[col])
                for col in events_df.columns
            )
            
            if feature_sig not in groups:
                groups[feature_sig] = []
            groups[feature_sig].append(idx)
        
        return groups
    
    def _can_reuse_chain(self, event_df, model_chain):
        """Check if a model chain can be reused for an event"""
        # Simple heuristic: check if the first model in the chain
        # would make the same decision
        if not model_chain:
            return False
        
        first_model = model_chain[0]
        prediction = first_model.predict(event_df)[0]
        
        # If first model's prediction is in its exit classes, 
        # the chain would likely be the same
        return prediction in first_model.exit_classes
    
    def _execute_fixed_chain(self, event_df, model_chain):
        """Execute a pre-determined model chain"""
        total_cost = 0
        
        for model in model_chain:
            prediction = model.predict(event_df)[0]
            total_cost += model.config.cost
            
            # Check exit condition
            if prediction in model.exit_classes:
                return prediction, total_cost
        
        # Fallback to last model's prediction
        final_prediction = model_chain[-1].predict(event_df)[0]
        return final_prediction, total_cost
```

### Memory Optimization

#### 1. Model Memory Management

```python
import gc
import psutil
from sklearn.externals import joblib

class MemoryEfficientNOMAD:
    def __init__(self, models, config, memory_limit_mb=1024):
        self.models_config = {m.config.name: m.config for m in models}
        self.config = config
        self.memory_limit = memory_limit_mb * 1024 * 1024  # Convert to bytes
        
        # Only keep lightweight model metadata in memory
        self.loaded_models = {}
        self.model_files = {}  # Store serialized models
        self.access_times = {}
        
        # Initialize by saving all models to disk
        self._serialize_models(models)
        
    def _serialize_models(self, models):
        """Serialize models to disk to save memory"""
        for model in models:
            if model._trained:
                model_file = f"temp_model_{model.config.name}.joblib"
                joblib.dump(model, model_file)
                self.model_files[model.config.name] = model_file
                print(f"Serialized {model.config.name} to {model_file}")
        
    def _load_model_on_demand(self, model_name):
        """Load model from disk when needed"""
        if model_name in self.loaded_models:
            self.access_times[model_name] = time.time()
            return self.loaded_models[model_name]
        
        # Check memory usage before loading
        if self._get_memory_usage() > self.memory_limit * 0.8:
            self._evict_least_used_model()
        
        # Load model from disk
        if model_name in self.model_files:
            model = joblib.load(self.model_files[model_name])
            self.loaded_models[model_name] = model
            self.access_times[model_name] = time.time()
            return model
        
        return None
    
    def _evict_least_used_model(self):
        """Remove least recently used model from memory"""
        if not self.loaded_models:
            return
        
        lru_model = min(self.access_times.keys(), 
                       key=lambda k: self.access_times[k])
        
        del self.loaded_models[lru_model]
        del self.access_times[lru_model]
        gc.collect()  # Force garbage collection
        
        print(f"Evicted {lru_model} from memory")
    
    def _get_memory_usage(self):
        """Get current memory usage in bytes"""
        process = psutil.Process()
        return process.memory_info().rss
    
    def classify_event(self, event_features):
        """Memory-efficient event classification"""
        available_models = list(self.models_config.keys())
        realized_chain = []
        
        while available_models:
            # Select next model (simplified utility calculation)
            selected_model_name = self._select_model_memory_efficient(available_models)
            
            # Load model on demand
            selected_model = self._load_model_on_demand(selected_model_name)
            if selected_model is None:
                available_models.remove(selected_model_name)
                continue
            
            # Make prediction
            prediction = selected_model.predict(event_features)[0]
            realized_chain.append(selected_model_name)
            
            # Check exit condition
            if prediction in selected_model.exit_classes:
                cost = sum(self.models_config[name].cost for name in realized_chain)
                return prediction, cost, realized_chain
            
            available_models.remove(selected_model_name)
        
        # Fallback to role model
        role_model = self._load_model_on_demand(self.config.role_model_name)
        prediction = role_model.predict(event_features)[0]
        realized_chain.append(self.config.role_model_name)
        
        cost = sum(self.models_config[name].cost for name in realized_chain)
        return prediction, cost, realized_chain
    
    def _select_model_memory_efficient(self, available_models):
        """Simplified model selection to reduce memory usage"""
        # Prefer models already in memory
        for model_name in available_models:
            if model_name in self.loaded_models:
                return model_name
        
        # Otherwise, select cheapest available model
        return min(available_models, 
                  key=lambda name: self.models_config[name].cost)
    
    def cleanup(self):
        """Clean up temporary files"""
        for model_file in self.model_files.values():
            if os.path.exists(model_file):
                os.remove(model_file)
```

### Scalability Optimizations

#### 1. Parallel Processing

```python
from multiprocessing import Pool, Queue, Process
import threading

class ParallelNOMADProcessor:
    def __init__(self, nomad_engine, n_workers=4):
        self.nomad = nomad_engine
        self.n_workers = n_workers
        self.result_queue = Queue()
        
    def process_events_parallel(self, events_df):
        """Process events using multiple worker processes"""
        # Split events into chunks for parallel processing
        chunk_size = len(events_df) // self.n_workers
        event_chunks = [
            events_df.iloc[i:i+chunk_size] 
            for i in range(0, len(events_df), chunk_size)
        ]
        
        # Process chunks in parallel
        with Pool(self.n_workers) as pool:
            chunk_results = pool.map(self._process_chunk, event_chunks)
        
        # Combine results
        all_results = []
        for chunk_result in chunk_results:
            all_results.extend(chunk_result)
        
        return all_results
    
    def _process_chunk(self, events_chunk):
        """Process a chunk of events in a single worker"""
        results = []
        
        # Each worker needs its own NOMAD instance to avoid conflicts
        worker_nomad = self._create_worker_nomad()
        
        for idx, row in events_chunk.iterrows():
            event_df = pd.DataFrame([row])
            prediction, cost, chain = worker_nomad.classify_event(event_df)
            
            results.append({
                'index': idx,
                'prediction': prediction,
                'cost': cost,
                'chain': chain
            })
        
        return results
    
    def _create_worker_nomad(self):
        """Create a copy of NOMAD engine for worker process"""
        # Note: This is simplified - in practice, you'd need to handle
        # model serialization/deserialization properly
        return copy.deepcopy(self.nomad)

class AsyncNOMADProcessor:
    """Asynchronous NOMAD processing using threading"""
    
    def __init__(self, nomad_engine, max_concurrent=10):
        self.nomad = nomad_engine
        self.max_concurrent = max_concurrent
        self.semaphore = threading.Semaphore(max_concurrent)
        self.results = {}
        
    def process_event_async(self, event_id, event_df, callback=None):
        """Process single event asynchronously"""
        def worker():
            with self.semaphore:
                try:
                    prediction, cost, chain = self.nomad.classify_event(event_df)
                    result = {
                        'event_id': event_id,
                        'prediction': prediction,
                        'cost': cost,
                        'chain': chain,
                        'status': 'completed'
                    }
                    
                    if callback:
                        callback(result)
                    else:
                        self.results[event_id] = result
                        
                except Exception as e:
                    error_result = {
                        'event_id': event_id,
                        'error': str(e),
                        'status': 'failed'
                    }
                    
                    if callback:
                        callback(error_result)
                    else:
                        self.results[event_id] = error_result
        
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        return thread
    
    def process_batch_async(self, events_df, callback=None):
        """Process multiple events asynchronously"""
        threads = []
        
        for idx, row in events_df.iterrows():
            event_df = pd.DataFrame([row])
            thread = self.process_event_async(idx, event_df, callback)
            threads.append(thread)
        
        return threads
    
    def wait_for_completion(self, threads, timeout=None):
        """Wait for all threads to complete"""
        for thread in threads:
            thread.join(timeout=timeout)
        
        return list(self.results.values())
```

#### 2. Distributed Processing

```python
import redis
import pickle
from celery import Celery

# Redis-based distributed NOMAD processing
class DistributedNOMADProcessor:
    """Distributed NOMAD processing using Redis and Celery"""
    
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        self.app = Celery('nomad_processor', 
                         broker=f'redis://{redis_host}:{redis_port}/0')
        self._register_tasks()
    
    def _register_tasks(self):
        """Register Celery tasks"""
        
        @self.app.task
        def process_nomad_event(serialized_nomad, event_data):
            """Celery task for processing single event"""
            nomad_engine = pickle.loads(serialized_nomad)
            event_df = pd.DataFrame([event_data])
            
            prediction, cost, chain = nomad_engine.classify_event(event_df)
            
            return {
                'prediction': prediction,
                'cost': cost,
                'chain': chain
            }
        
        self.process_event_task = process_nomad_event
    
    def distribute_processing(self, nomad_engine, events_df):
        """Distribute processing across multiple workers"""
        # Serialize NOMAD engine once
        serialized_nomad = pickle.dumps(nomad_engine)
        
        # Submit tasks to workers
        job_results = []
        for _, row in events_df.iterrows():
            result = self.process_event_task.delay(
                serialized_nomad, 
                row.to_dict()
            )
            job_results.append(result)
        
        # Collect results
        results = []
        for job_result in job_results:
            try:
                result = job_result.get(timeout=30)
                results.append(result)
            except Exception as e:
                results.append({'error': str(e)})
        
        return results
```

#### 3. Database Integration Optimization

```python
import sqlite3
import pandas as pd
from sqlalchemy import create_engine

class DatabaseOptimizedNOMAD:
    """NOMAD with optimized database integration"""
    
    def __init__(self, nomad_engine, db_connection_string):
        self.nomad = nomad_engine
        self.engine = create_engine(db_connection_string)
        self.batch_buffer = []
        self.batch_size = 1000
        
    def process_streaming_data(self, query, batch_callback=None):
        """Process streaming data from database efficiently"""
        
        # Use chunked reading for memory efficiency
        chunk_iter = pd.read_sql(query, self.engine, chunksize=self.batch_size)
        
        results_buffer = []
        total_processed = 0
        
        for chunk in chunk_iter:
            # Process chunk with NOMAD
            chunk_results = self._process_chunk_optimized(chunk)
            results_buffer.extend(chunk_results)
            
            total_processed += len(chunk)
            
            # Batch insert results back to database
            if len(results_buffer) >= self.batch_size:
                self._batch_insert_results(results_buffer)
                results_buffer = []
            
            # Optional callback for progress monitoring
            if batch_callback:
                batch_callback(total_processed, chunk_results)
        
        # Insert remaining results
        if results_buffer:
            self._batch_insert_results(results_buffer)
        
        return total_processed
    
    def _process_chunk_optimized(self, chunk_df):
        """Process database chunk with optimizations"""
        results = []
        
        # Group similar events to reuse computations
        feature_groups = chunk_df.groupby(
            chunk_df.select_dtypes(include=[np.number]).columns.tolist()
        )
        
        for group_key, group_events in feature_groups:
            # Process representative event
            representative = group_events.iloc[[0]]
            pred, cost, chain = self.nomad.classify_event(representative)
            
            # Apply result to all events in group
            for idx, _ in group_events.iterrows():
                results.append({
                    'event_id': idx,
                    'prediction': pred,
                    'cost': cost,
                    'model_chain': ','.join(chain),
                    'processing_timestamp': pd.Timestamp.now()
                })
        
        return results
    
    def _batch_insert_results(self, results):
        """Efficiently insert results using batch operations"""
        results_df = pd.DataFrame(results)
        results_df.to_sql('nomad_results', self.engine, 
                         if_exists='append', index=False, method='multi')
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Training and Setup Issues

**Problem**: `ImportError: No module named 'nomad_api'`
```python
# Solution: Ensure proper installation
pip install -e .  # If installing from source
# Or add to Python path
import sys
sys.path.append('/path/to/nomad')
```

**Problem**: Models fail to train with error: "Model training error: No numerical features found"
```python
# Solution: Check data format and preprocessing
data = pd.read_csv('your_data.csv')
print("Data info:")
print(data.info())
print("Data types:")
print(data.dtypes)

# Ensure numeric features exist
numeric_cols = data.select_dtypes(include=[np.number]).columns
print(f"Numeric columns: {numeric_cols}")

if len(numeric_cols) == 0:
    print("WARNING: No numeric features found. Check data encoding.")
```

**Problem**: "Role model not found in provided models"
```python
# Solution: Verify model names match exactly
trained_models = nomad.models.keys()
print(f"Available models: {list(trained_models)}")

# Ensure role model name is in the list
role_model_name = "your_role_model"
if role_model_name not in trained_models:
    print(f"ERROR: {role_model_name} not in {list(trained_models)}")
    # Use an available model name instead
```

**Problem**: Custom model upload fails
```python
# Solution: Verify custom model interface
class YourCustomModel:
    def fit(self, X, y):
        # Must implement fit method
        return self
    
    def predict(self, X):
        # Must implement predict method
        return predictions
    
    def predict_proba(self, X):  # Recommended
        # Should implement predict_proba for better performance
        return probabilities
    
    def get_params(self, deep=True):  # Required for sklearn compatibility
        return {}
    
    def set_params(self, **params):  # Required for sklearn compatibility
        return self
```

#### 2. Configuration Issues

**Problem**: Very low speedup (<1.2x) despite reasonable epsilon
```python
# Diagnostic approach
def diagnose_low_speedup(nomad_system, test_data):
    print("=== NOMAD Speedup Diagnostic ===")
    
    # Check model cost ratios
    costs = [model.config.cost for model in nomad_system.models.values()]
    print(f"Model costs: {costs}")
    print(f"Cost ratio (max/min): {max(costs)/min(costs):.2f}")
    
    if max(costs)/min(costs) < 2.0:
        print("ISSUE: Low cost variation between models")
        print("SOLUTION: Increase cost differences or use models with more varied complexity")
    
    # Check model accuracy differences  
    accuracies = [model.accuracy for model in nomad_system.models.values()]
    print(f"Model accuracies: {accuracies}")
    print(f"Accuracy range: {max(accuracies) - min(accuracies):.3f}")
    
    if max(accuracies) - min(accuracies) < 0.05:
        print("ISSUE: Models have very similar accuracy")
        print("SOLUTION: Use more diverse models or increase dataset complexity")
    
    # Check epsilon setting
    print(f"Current epsilon: {nomad_system.config.epsilon}")
    if nomad_system.config.epsilon < 0.1:
        print("ISSUE: Epsilon too restrictive")
        print("SOLUTION: Try increasing epsilon to 0.15-0.20")
    
    # Check exit class distribution
    total_exit_classes = sum(len(model.exit_classes) for model in nomad_system.models.values())
    avg_exit_classes = total_exit_classes / len(nomad_system.models)
    print(f"Average exit classes per model: {avg_exit_classes:.1f}")
    
    if avg_exit_classes < 1.5:
        print("ISSUE: Models have too few exit classes")
        print("SOLUTION: Reduce epsilon or retrain models")

# Usage
diagnose_low_speedup(your_nomad_system, your_test_data)
```

**Problem**: Quality degradation exceeds epsilon tolerance
```python
# Solution: Debug chain safety and model performance
def debug_quality_issues(nomad_system, test_data):
    print("=== Quality Degradation Debug ===")
    
    # Test individual model performance
    for model_name, model in nomad_system.models.items():
        if hasattr(model, 'accuracy'):
            print(f"{model_name}: Accuracy = {model.accuracy:.3f}")
    
    # Check safety bounds
    print(f"\nSafety check type: {nomad_system.config.safety_check_type}")
    
    # Test with different safety bounds
    original_safety = nomad_system.config.safety_check_type
    
    nomad_system.config.safety_check_type = SafetyCheckType.RELAXED
    relaxed_results = evaluate_nomad_sample(nomad_system, test_data.head(100))
    
    nomad_system.config.safety_check_type = SafetyCheckType.CONSERVATIVE  
    conservative_results = evaluate_nomad_sample(nomad_system, test_data.head(100))
    
    nomad_system.config.safety_check_type = original_safety
    
    print(f"Relaxed safety accuracy: {relaxed_results['accuracy']:.3f}")
    print(f"Conservative safety accuracy: {conservative_results['accuracy']:.3f}")

def evaluate_nomad_sample(nomad_system, sample_data):
    correct = 0
    total_cost = 0
    
    for _, row in sample_data.iterrows():
        event_df = pd.DataFrame([row.drop('target')])
        pred, cost, _ = nomad_system.classify_event(event_df)
        
        if pred == row['target']:
            correct += 1
        total_cost += cost
    
    return {
        'accuracy': correct / len(sample_data),
        'avg_cost': total_cost / len(sample_data)
    }
```

#### 3. Performance Issues

**Problem**: Memory usage grows continuously during streaming
```python
# Solution: Implement memory monitoring and cleanup
class MemoryMonitoredNOMAD:
    def __init__(self, nomad_engine, memory_threshold_mb=1024):
        self.nomad = nomad_engine
        self.memory_threshold = memory_threshold_mb * 1024 * 1024
        self.processed_events = 0
        
    def classify_event_monitored(self, event_df):
        """Classify with memory monitoring"""
        # Check memory usage periodically
        if self.processed_events % 100 == 0:
            self._check_memory_usage()
        
        result = self.nomad.classify_event(event_df)
        self.processed_events += 1
        
        return result
    
    def _check_memory_usage(self):
        """Monitor and manage memory usage"""
        import psutil
        process = psutil.Process()
        memory_usage = process.memory_info().rss
        
        print(f"Memory usage: {memory_usage / 1024 / 1024:.1f} MB")
        
        if memory_usage > self.memory_threshold:
            print("WARNING: Memory usage high. Running cleanup...")
            self._cleanup_memory()
    
    def _cleanup_memory(self):
        """Clean up memory usage"""
        import gc
        
        # Clear adaptive manager buffers if they exist
        if hasattr(self.nomad, 'adaptive_manager'):
            if hasattr(self.nomad.adaptive_manager, 'event_buffer'):
                buffer_size = len(self.nomad.adaptive_manager.event_buffer)
                if buffer_size > 1000:
                    # Keep only recent events
                    recent_events = list(self.nomad.adaptive_manager.event_buffer)[-500:]
                    self.nomad.adaptive_manager.event_buffer.clear()
                    self.nomad.adaptive_manager.event_buffer.extend(recent_events)
                    print(f"Reduced event buffer from {buffer_size} to 500 events")
        
        # Force garbage collection
        gc.collect()
```

**Problem**: Slow processing speed for real-time applications
```python
# Solution: Profile and optimize bottlenecks
import cProfile
import pstats

def profile_nomad_performance(nomad_system, test_data):
    """Profile NOMAD performance to identify bottlenecks"""
    
    # Profile classification process
    profiler = cProfile.Profile()
    
    print("Profiling NOMAD classification...")
    profiler.enable()
    
    # Run sample classification
    sample_size = min(100, len(test_data))
    for i in range(sample_size):
        event_df = test_data.iloc[[i]].drop('target', axis=1)
        nomad_system.classify_event(event_df)
    
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    
    print(f"\nTop 10 time-consuming functions:")
    stats.print_stats(10)
    
    # Identify common bottlenecks
    print("\n=== Performance Analysis ===")
    print("Common bottlenecks and solutions:")
    print("1. Model.predict() calls - Consider model caching")
    print("2. Safety check calculations - Use cached safety bounds")
    print("3. Utility calculations - Implement utility caching")
    print("4. Feature preprocessing - Optimize data transformations")

# Usage
profile_nomad_performance(your_nomad_system, your_test_data)
```

#### 4. Accuracy and Reliability Issues

**Problem**: Inconsistent results across runs
```python
# Solution: Set random seeds and check for non-deterministic behavior
def ensure_reproducible_results():
    """Set all random seeds for reproducible results"""
    import random
    import numpy as np
    from sklearn.utils import check_random_state
    
    # Set seeds
    random.seed(42)
    np.random.seed(42)
    
    # For sklearn models, ensure random_state parameter is set
    for model_config in your_model_configs:
        if 'random_state' not in model_config.params:
            model_config.params['random_state'] = 42
    
    print("Random seeds set for reproducible results")

def validate_nomad_consistency(nomad_system, test_event, num_runs=10):
    """Test NOMAD consistency across multiple runs"""
    results = []
    
    for run in range(num_runs):
        pred, cost, chain = nomad_system.classify_event(test_event.copy())
        results.append((pred, cost, tuple(chain)))
    
    # Check consistency
    unique_results = set(results)
    
    if len(unique_results) == 1:
        print("✓ NOMAD results are consistent across runs")
    else:
        print("✗ NOMAD results are inconsistent:")
        for result in unique_results:
            print(f"  {result}")
        print("Check for non-deterministic model behavior")
    
    return len(unique_results) == 1
```

#### 5. Integration Issues

**Problem**: Web interface connection errors
```python
# Backend debugging
def debug_flask_backend():
    """Debug common Flask backend issues"""
    try:
        response = requests.get('http://127.0.0.1:5001/api/hello')
        print(f"Backend status: {response.status_code}")
        print(f"Response: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to Flask backend")
        print("Solutions:")
        print("1. Ensure Flask app is running: python app.py")
        print("2. Check port 5001 is not in use")
        print("3. Verify Flask-CORS is installed")
    except Exception as e:
        print(f"Backend error: {e}")

# Frontend debugging  
def debug_react_frontend():
    """Debug common React frontend issues"""
    print("Frontend debugging checklist:")
    print("1. Ensure npm dependencies are installed: npm install")
    print("2. Check Tailwind CSS is configured correctly")
    print("3. Verify API endpoints in frontend match backend")
    print("4. Check browser console for JavaScript errors")
    print("5. Ensure CORS is enabled on backend")
```

### Error Recovery and Resilience

#### Handling Model Failures

```python
class ResilientNOMADEngine(NOMADEngine):
    """NOMAD engine with enhanced error handling and recovery"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failed_models = set()
        self.error_counts = {}
        self.max_errors_per_model = 5
        
    def classify_event_resilient(self, event_features):
        """Classify with error handling and recovery"""
        available_models = [m for m in self.models.values() 
                           if m.config.name not in self.failed_models]
        realized_chain = []
        
        while available_models:
            try:
                selected_model = self._select_next_model_safe(available_models)
                if selected_model is None:
                    break
                
                # Attempt prediction with error handling
                prediction, probabilities = self._predict_with_fallback(
                    selected_model, event_features
                )
                
                realized_chain.append(selected_model.config.name)
                
                # Check exit condition
                if prediction in selected_model.exit_classes:
                    cost = sum(self.models[name].config.cost for name in realized_chain)
                    return prediction, cost, realized_chain
                
                # Update beliefs for next iteration
                current_priors = self._update_beliefs_safe(probabilities)
                available_models.remove(selected_model)
                
            except Exception as e:
                # Handle model failure
                self._handle_model_error(selected_model, e)
                if selected_model in available_models:
                    available_models.remove(selected_model)
                continue
        
        # Fallback to role model with error handling
        try:
            role_prediction = self.role_model.predict(event_features)[0]
            realized_chain.append(self.role_model.config.name)
            cost = sum(self.models[name].config.cost for name in realized_chain)
            return role_prediction, cost, realized_chain
            
        except Exception as e:
            # Ultimate fallback - return most common class
            print(f"CRITICAL: Role model failed: {e}")
            fallback_prediction = self._get_fallback_prediction()
            return fallback_prediction, 0, ['fallback']
    
    def _predict_with_fallback(self, model, event_features):
        """Make prediction with fallback for probability estimation"""
        try:
            prediction = model.predict(event_features)[0]
            
            # Try to get probabilities
            try:
                probabilities = model.predict_proba(event_features)[0]
            except:
                # Fallback to uniform probabilities
                num_classes = len(self.class_names)
                probabilities = np.ones(num_classes) / num_classes
                
            return prediction, probabilities
            
        except Exception as e:
            # Re-raise for higher-level error handling
            raise RuntimeError(f"Model {model.config.name} prediction failed: {e}")
    
    def _handle_model_error(self, model, error):
        """Handle model errors and decide whether to disable model"""
        model_name = model.config.name
        self.error_counts[model_name] = self.error_counts.get(model_name, 0) + 1
        
        print(f"Model {model_name} error #{self.error_counts[model_name]}: {error}")
        
        if self.error_counts[model_name] >= self.max_errors_per_model:
            print(f"Disabling model {model_name} due to repeated failures")
            self.failed_models.add(model_name)
    
    def _get_fallback_prediction(self):
        """Get fallback prediction when all models fail"""
        # Return most common class from training data
        if hasattr(self, '_fallback_class'):
            return self._fallback_class
        
        # Default to first class
        return self.class_names[0] if self.class_names else "unknown"
    
    def get_system_health(self):
        """Get system health status"""
        total_models = len(self.models)
        active_models = total_models - len(self.failed_models)
        
        return {
            'total_models': total_models,
            'active_models': active_models,
            'failed_models': list(self.failed_models),
            'error_counts': dict(self.error_counts),
            'health_percentage': (active_models / total_models) * 100
        }
```

### Monitoring and Alerting

```python
import logging
from datetime import datetime, timedelta

class NOMADMonitor:
    """Monitoring and alerting system for NOMAD"""
    
    def __init__(self, nomad_engine, alert_thresholds=None):
        self.nomad = nomad_engine
        self.thresholds = alert_thresholds or {
            'accuracy_drop': 0.05,  # 5% accuracy drop
            'cost_increase': 2.0,   # 2x cost increase
            'error_rate': 0.1       # 10% error rate
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('nomad_monitoring.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.baseline_accuracy = None
        self.baseline_cost = None
        self.recent_performance = []
        self.alert_history = []
        
    def track_performance(self, predictions, true_labels, costs):
        """Track performance metrics"""
        accuracy = sum(p == t for p, t in zip(predictions, true_labels)) / len(predictions)
        avg_cost = sum(costs) / len(costs)
        
        # Store performance data
        performance_data = {
            'timestamp': datetime.now(),
            'accuracy': accuracy,
            'avg_cost': avg_cost,
            'num_events': len(predictions)
        }
        
        self.recent_performance.append(performance_data)
        
        # Keep only recent data (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.recent_performance = [
            p for p in self.recent_performance 
            if p['timestamp'] > cutoff_time
        ]
        
        # Set baselines if not set
        if self.baseline_accuracy is None:
            self.baseline_accuracy = accuracy
            self.baseline_cost = avg_cost
            self.logger.info(f"Baseline set - Accuracy: {accuracy:.3f}, Cost: {avg_cost:.2f}")
        
        # Check for alerts
        self._check_alerts(performance_data)
        
        return performance_data
    
    def _check_alerts(self, current_performance):
        """Check for alert conditions"""
        alerts = []
        
        # Check accuracy drop
        accuracy_drop = self.baseline_accuracy - current_performance['accuracy']
        if accuracy_drop > self.thresholds['accuracy_drop']:
            alerts.append({
                'type': 'accuracy_drop',
                'severity': 'high',
                'message': f"Accuracy dropped by {accuracy_drop:.3f} "
                          f"(current: {current_performance['accuracy']:.3f}, "
                          f"baseline: {self.baseline_accuracy:.3f})"
            })
        
        # Check cost increase
        cost_ratio = current_performance['avg_cost'] / self.baseline_cost
        if cost_ratio > self.thresholds['cost_increase']:
            alerts.append({
                'type': 'cost_increase',
                'severity': 'medium',
                'message': f"Cost increased by {cost_ratio:.1f}x "
                          f"(current: {current_performance['avg_cost']:.2f}, "
                          f"baseline: {self.baseline_cost:.2f})"
            })
        
        # Process alerts
        for alert in alerts:
            self._send_alert(alert)
    
    def _send_alert(self, alert):
        """Send alert notification"""
        alert['timestamp'] = datetime.now()
        self.alert_history.append(alert)
        
        # Log alert
        self.logger.warning(f"ALERT [{alert['type']}]: {alert['message']}")
        
        # In production, integrate with alerting system (email, Slack, etc.)
        # self._send_email_alert(alert)
        # self._send_slack_alert(alert)
    
    def get_performance_report(self, hours=24):
        """Generate performance report"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_data = [p for p in self.recent_performance if p['timestamp'] > cutoff_time]
        
        if not recent_data:
            return "No performance data available"
        
        # Calculate statistics
        accuracies = [p['accuracy'] for p in recent_data]
        costs = [p['avg_cost'] for p in recent_data]
        
        report = {
            'time_period': f"Last {hours} hours",
            'total_events': sum(p['num_events'] for p in recent_data),
            'accuracy_stats': {
                'mean': np.mean(accuracies),
                'min': np.min(accuracies),
                'max': np.max(accuracies),
                'std': np.std(accuracies)
            },
            'cost_stats': {
                'mean': np.mean(costs),
                'min': np.min(costs),
                'max': np.max(costs),
                'std': np.std(costs)
            },
            'alerts_count': len([a for a in self.alert_history 
                               if a['timestamp'] > cutoff_time]),
            'system_health': self.nomad.get_system_health() if hasattr(self.nomad, 'get_system_health') else 'N/A'
        }
        
        return report
    
    def export_metrics(self, filename=None):
        """Export metrics for external analysis"""
        if filename is None:
            filename = f"nomad_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df = pd.DataFrame(self.recent_performance)
        df.to_csv(filename, index=False)
        
        self.logger.info(f"Metrics exported to {filename}")
        return filename
```

---

## Contributing & Extending

### Development Setup

#### Setting Up Development Environment

```bash
# Clone repository
git clone <nomad-repository-url>
cd nomad

# Create development environment
python -m venv venv_dev
source venv_dev/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
python -m pytest tests/ -v

# Run linting
flake8 nomad_api/
black nomad_api/

# Type checking
mypy nomad_api/
```

#### Development Dependencies (`requirements-dev.txt`)

```
# Core dependencies
-r requirements.txt

# Development tools
pytest>=6.0.0
pytest-cov>=2.10.0
black>=21.0.0
flake8>=3.8.0
mypy>=0.812
pre-commit>=2.15.0

# Documentation
sphinx>=4.0.0
sphinx-rtd-theme>=0.5.0

# Testing utilities
factory-boy>=3.2.0
parameterized>=0.8.0
mock>=4.0.0
```

### Architecture for Extensions

#### Plugin System Design

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import importlib

class NOMADPlugin(ABC):
    """Base class for NOMAD plugins"""
    
    @abstractmethod
    def get_name(self) -> str:
        """Return plugin name"""
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        """Return plugin version"""
        pass
    
    @abstractmethod
    def initialize(self, nomad_engine: 'NOMADEngine') -> None:
        """Initialize plugin with NOMAD engine"""
        pass

class ModelPlugin(NOMADPlugin):
    """Plugin interface for custom model implementations"""
    
    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """Return list of model types this plugin supports"""
        pass
    
    @abstractmethod
    def create_model(self, model_config: 'ModelConfig') -> Any:
        """Create model instance from configuration"""
        pass

class AdaptivePlugin(NOMADPlugin):
    """Plugin interface for custom adaptive algorithms"""
    
    @abstractmethod
    def create_adaptive_manager(self, config: 'AdaptiveConfig') -> Any:
        """Create adaptive manager instance"""
        pass

class SafetyPlugin(NOMADPlugin):
    """Plugin interface for custom safety checking algorithms"""
    
    @abstractmethod
    def check_chain_safety(self, chain: List['Model'], class_name: str, 
                          epsilon: float) -> bool:
        """Implement custom chain safety checking"""
        pass

# Plugin manager
class NOMADPluginManager:
    """Manages NOMAD plugins"""
    
    def __init__(self):
        self.plugins = {}
        self.model_plugins = {}
        self.adaptive_plugins = {}
        self.safety_plugins = {}
    
    def register_plugin(self, plugin: NOMADPlugin):
        """Register a plugin"""
        plugin_name = plugin.get_name()
        self.plugins[plugin_name] = plugin
        
        # Categorize plugins
        if isinstance(plugin, ModelPlugin):
            for model_type in plugin.get_supported_models():
                self.model_plugins[model_type] = plugin
        elif isinstance(plugin, AdaptivePlugin):
            self.adaptive_plugins[plugin_name] = plugin
        elif isinstance(plugin, SafetyPlugin):
            self.safety_plugins[plugin_name] = plugin
    
    def load_plugins_from_directory(self, plugin_dir: str):
        """Load all plugins from directory"""
        import os
        import sys
        
        if plugin_dir not in sys.path:
            sys.path.append(plugin_dir)
        
        for filename in os.listdir(plugin_dir):
            if filename.endswith('.py') and not filename.startswith('_'):
                module_name = filename[:-3]
                try:
                    module = importlib.import_module(module_name)
                    
                    # Look for plugin classes
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and 
                            issubclass(attr, NOMADPlugin) and 
                            attr != NOMADPlugin):
                            plugin_instance = attr()
                            self.register_plugin(plugin_instance)
                            
                except Exception as e:
                    print(f"Failed to load plugin {module_name}: {e}")
    
    def get_plugin(self, plugin_name: str) -> NOMADPlugin:
        """Get plugin by name"""
        return self.plugins.get(plugin_name)
    
    def create_model(self, model_config: 'ModelConfig'):
        """Create model using appropriate plugin"""
        model_type = model_config.model_type
        
        if model_type in self.model_plugins:
            plugin = self.model_plugins[model_type]
            return plugin.create_model(model_config)
        
        # Fallback to built-in models
        from nomad_api import CLASSIFIER_MAP
        if model_type in CLASSIFIER_MAP:
            return CLASSIFIER_MAP[model_type](**model_config.params)
        
        raise ValueError(f"Unknown model type: {model_type}")
```

#### Example Plugin Implementation

```python
# plugins/neural_network_plugin.py
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from nomad_api import ModelPlugin

class PyTorchModel(BaseEstimator, ClassifierMixin):
    """PyTorch model wrapper for NOMAD"""
    
    def __init__(self, hidden_sizes=[64, 32], learning_rate=0.001, epochs=100):
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = None
        self.classes_ = None
        
    def fit(self, X, y):
        n_features = X.shape[1]
        n_classes = len(np.unique(y))
        
        # Create PyTorch model
        layers = []
        prev_size = n_features
        
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, n_classes))
        self.model = nn.Sequential(*layers)
        
        # Train model
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X)
        y_tensor = torch.LongTensor(y)
        
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        self.classes_ = np.unique(y)
        return self
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
        
        return self.classes_[predictions.numpy()]
    
    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model not trained")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.numpy()

class NeuralNetworkPlugin(ModelPlugin):
    """Plugin for PyTorch neural networks"""
    
    def get_name(self) -> str:
        return "neural_network_plugin"
    
    def get_version(self) -> str:
        return "1.0.0"
    
    def initialize(self, nomad_engine):
        print(f"Initialized {self.get_name()} v{self.get_version()}")
    
    def get_supported_models(self) -> List[str]:
        return ["PyTorchModel", "DeepNeuralNetwork"]
    
    def create_model(self, model_config):
        if model_config.model_type == "PyTorchModel":
            return PyTorchModel(**model_config.params)
        elif model_config.model_type == "DeepNeuralNetwork":
            # Create deeper network
            params = model_config.params.copy()
            params['hidden_sizes'] = params.get('hidden_sizes', [128, 64, 32, 16])
            return PyTorchModel(**params)
        
        raise ValueError(f"Unsupported model type: {model_config.model_type}")

# Usage
plugin_manager = NOMADPluginManager()
plugin_manager.register_plugin(NeuralNetworkPlugin())

# Now PyTorch models can be used in NOMAD
pytorch_config = ModelConfig(
    name="deep_pytorch",
    model_type="PyTorchModel",
    cost=20.0,
    params={
        "hidden_sizes": [128, 64, 32],
        "learning_rate": 0.001,
        "epochs": 50
    }
)
```

### Testing Framework

#### Unit Testing Structure

```python
# tests/test_nomad_engine.py
import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from nomad_api import NOMADEngine, ModelConfig, NOMADConfig
from nomad_api_implementation import NOMADEngineImpl, ModelImpl

class TestNOMADEngine(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'target': ['A', 'B', 'A', 'B', 'A']
        })
        
        self.model_configs = [
            ModelConfig("model1", "DecisionTreeClassifier", 1.0, 
                       params={"max_depth": 3}),
            ModelConfig("model2", "RandomForestClassifier", 5.0,
                       params={"n_estimators": 10}),
            ModelConfig("model3", "LogisticRegression", 2.0)
        ]
        
        self.nomad_config = NOMADConfig(
            role_model_name="model2",
            epsilon=0.1
        )
    
    def test_nomad_initialization(self):
        """Test NOMAD engine initialization"""
        models = [ModelImpl(config) for config in self.model_configs]
        
        nomad = NOMADEngineImpl(
            models=models,
            config=self.nomad_config,
            class_names=['A', 'B']
        )
        
        self.assertEqual(len(nomad.models), 3)
        self.assertEqual(nomad.role_model.config.name, "model2")
        self.assertEqual(nomad.config.epsilon, 0.1)
    
    def test_model_training(self):
        """Test model training process"""
        nomad = self._create_test_nomad()
        
        # Mock training data
        trained_nomad = nomad._train_models(self.sample_data, 'target')
        
        # Verify training completed
        self.assertTrue(trained_nomad._trained)
        self.assertIsNotNone(trained_nomad._backend_engine)
    
    def test_event_classification(self):
        """Test single event classification"""
        nomad = self._create_trained_nomad()
        
        test_event = pd.DataFrame([[1, 2]], columns=['feature1', 'feature2'])
        
        prediction, cost, chain = nomad.classify_event(test_event)
        
        # Verify output format
        self.assertIsInstance(prediction, str)
        self.assertIsInstance(cost, (int, float))
        self.assertIsInstance(chain, list)
        self.assertTrue(len(chain) > 0)
    
    def test_epsilon_tolerance(self):
        """Test epsilon tolerance behavior"""
        # Test with strict epsilon
        strict_config = NOMADConfig(role_model_name="model2", epsilon=0.05)
        strict_nomad = self._create_nomad_with_config(strict_config)
        
        # Test with lenient epsilon
        lenient_config = NOMADConfig(role_model_name="model2", epsilon=0.3)
        lenient_nomad = self._create_nomad_with_config(lenient_config)
        
        test_event = pd.DataFrame([[1, 2]], columns=['feature1', 'feature2'])
        
        strict_pred, strict_cost, strict_chain = strict_nomad.classify_event(test_event)
        lenient_pred, lenient_cost, lenient_chain = lenient_nomad.classify_event(test_event)
        
        # Lenient configuration should generally use cheaper models
        self.assertLessEqual(lenient_cost, strict_cost * 1.5)  # Allow some variance
    
    def test_stream_processing(self):
        """Test stream processing functionality"""
        nomad = self._create_trained_nomad()
        
        results = list(nomad.classify_stream(self.sample_data))
        
        # Verify we get setup and summary messages
        setup_msgs = [r for r in results if r.get('type') == 'setup']
        summary_msgs = [r for r in results if r.get('type') == 'summary']
        
        self.assertEqual(len(setup_msgs), 1)
        self.assertEqual(len(summary_msgs), 1)
        
        # Verify summary contains expected fields
        summary = summary_msgs[0]
        self.assertIn('nomad_overall_metrics', summary)
        self.assertIn('role_model_performance', summary)
    
    def test_model_addition_removal(self):
        """Test dynamic model management"""
        nomad = self._create_test_nomad()
        
        initial_count = len(nomad.models)
        
        # Add new model
        new_model_config = ModelConfig("new_model", "GaussianNB", 3.0)
        new_model = ModelImpl(new_model_config)
        nomad.add_model(new_model)
        
        self.assertEqual(len(nomad.models), initial_count + 1)
        self.assertIn("new_model", nomad.models)
        
        # Remove model
        nomad.remove_model("new_model")
        self.assertEqual(len(nomad.models), initial_count)
        self.assertNotIn("new_model", nomad.models)
        
        # Test removing role model (should fail)
        with self.assertRaises(ValueError):
            nomad.remove_model("model2")
    
    def _create_test_nomad(self):
        """Helper to create test NOMAD instance"""
        models = [ModelImpl(config) for config in self.model_configs]
        return NOMADEngineImpl(
            models=models,
            config=self.nomad_config,
            class_names=['A', 'B']
        )
    
    def _create_trained_nomad(self):
        """Helper to create trained NOMAD instance"""
        nomad = self._create_test_nomad()
        return nomad._train_models(self.sample_data, 'target')
    
    def _create_nomad_with_config(self, config):
        """Helper to create NOMAD with specific config"""
        models = [ModelImpl(cfg) for cfg in self.model_configs]
        nomad = NOMADEngineImpl(models=models, config=config, class_names=['A', 'B'])
        return nomad._train_models(self.sample_data, 'target')

# Integration tests
class TestNOMADIntegration(unittest.TestCase):
    
    def test_end_to_end_workflow(self):
        """Test complete NOMAD workflow"""
        from nomad_api_implementation import load_and_train_nomad
        
        # Create test CSV
        test_data = pd.DataFrame({
            'f1': np.random.randn(100),
            'f2': np.random.randn(100),
            'f3': np.random.randn(100),
            'target': np.random.choice(['X', 'Y', 'Z'], 100)
        })
        
        test_csv = 'test_data.csv'
        test_data.to_csv(test_csv, index=False)
        
        try:
            model_configs = [
                ModelConfig("tree", "DecisionTreeClassifier", 1.0),
                ModelConfig("forest", "RandomForestClassifier", 5.0)
            ]
            
            # Test loading and training
            nomad = load_and_train_nomad(
                csv_path=test_csv,
                model_configs=model_configs,
                role_model_name="forest"
            )
            
            # Test classification
            test_event = test_data.iloc[[0]].drop('target', axis=1)
            prediction, cost, chain = nomad.classify_event(test_event)
            
            # Verify results
            self.assertIn(prediction, ['X', 'Y', 'Z'])
            self.assertGreater(cost, 0)
            self.assertTrue(len(chain) > 0)
            
        finally:
            import os
            if os.path.exists(test_csv):
                os.remove(test_csv)

if __name__ == '__main__':
    unittest.main()
```

#### Performance Testing

```python
# tests/test_performance.py
import time
import pytest
import numpy as np
import pandas as pd
from nomad_api_implementation import train_nomad_from_data
from nomad_api import ModelConfig

class TestPerformance:
    
    @pytest.fixture
    def large_dataset(self):
        """Create large test dataset"""
        np.random.seed(42)
        n_samples = 10000
        n_features = 50
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.choice(['A', 'B', 'C', 'D'], n_samples)
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        df['target'] = y
        
        return df
    
    @pytest.fixture
    def model_configs(self):
        """Standard model configurations"""
        return [
            ModelConfig("fast", "DecisionTreeClassifier", 1.0, 
                       params={"max_depth": 5}),
            ModelConfig("medium", "RandomForestClassifier", 5.0,
                       params={"n_estimators": 50}),
            ModelConfig("slow", "GradientBoostingClassifier", 15.0,
                       params={"n_estimators": 100})
        ]
    
    def test_training_performance(self, large_dataset, model_configs):
        """Test training performance on large dataset"""
        start_time = time.time()
        
        nomad = train_nomad_from_data(
            data=large_dataset,
            model_configs=model_configs,
            role_model_name="slow"
        )
        
        training_time = time.time() - start_time
        
        # Training should complete within reasonable time
        assert training_time < 300  # 5 minutes max
        assert nomad._trained
        
        print(f"Training time: {training_time:.2f} seconds")
    
    def test_classification_speed(self, large_dataset, model_configs):
        """Test classification speed"""
        nomad = train_nomad_from_data(
            data=large_dataset.iloc[:1000],  # Smaller training set
            model_configs=model_configs,
            role_model_name="slow"
        )
        
        test_data = large_dataset.iloc[1000:1100]  # 100 test events
        
        start_time = time.time()
        
        for _, row in test_data.iterrows():
            event_df = pd.DataFrame([row.drop('target')])
            nomad.classify_event(event_df)
        
        classification_time = time.time() - start_time
        events_per_second = len(test_data) / classification_time
        
        # Should achieve reasonable throughput
        assert events_per_second > 10  # At least 10 events/second
        
        print(f"Classification speed: {events_per_second:.1f} events/second")
    
    def test_memory_usage(self, large_dataset, model_configs):
        """Test memory usage during processing"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        nomad = train_nomad_from_data(
            data=large_dataset,
            model_configs=model_configs,
            role_model_name="slow"
        )
        
        post_training_memory = process.memory_info().rss
        
        # Process large batch
        test_data = large_dataset.iloc[:1000]
        for _, row in test_data.iterrows():
            event_df = pd.DataFrame([row.drop('target')])
            nomad.classify_event(event_df)
        
        final_memory = process.memory_info().rss
        
        # Memory growth should be reasonable
        training_growth = (post_training_memory - initial_memory) / 1024 / 1024  # MB
        processing_growth = (final_memory - post_training_memory) / 1024 / 1024  # MB
        
        print(f"Memory growth - Training: {training_growth:.1f} MB, "
              f"Processing: {processing_growth:.1f} MB")
        
        # Should not exceed reasonable limits
        assert training_growth < 500  # 500 MB for training
        assert processing_growth < 100  # 100 MB for processing
    
    def test_scalability(self, model_configs):
        """Test scalability with different dataset sizes"""
        sizes = [100, 500, 1000, 5000]
        results = []
        
        for size in sizes:
            # Generate dataset of specific size
            np.random.seed(42)
            X = np.random.randn(size, 10)
            y = np.random.choice(['A', 'B'], size)
            df = pd.DataFrame(X, columns=[f'f_{i}' for i in range(10)])
            df['target'] = y
            
            start_time = time.time()
            nomad = train_nomad_from_data(df, model_configs, "medium")
            training_time = time.time() - start_time
            
            # Test classification speed
            test_events = 50
            start_time = time.time()
            for i in range(test_events):
                event_df = pd.DataFrame([df.iloc[i % len(df)].drop('target')])
                nomad.classify_event(event_df)
            classification_time = time.time() - start_time
            
            results.append({
                'dataset_size': size,
                'training_time': training_time,
                'events_per_second': test_events / classification_time
            })
        
        # Verify reasonable scalability
        for i in range(1, len(results)):
            size_ratio = results[i]['dataset_size'] / results[i-1]['dataset_size']
            time_ratio = results[i]['training_time'] / results[i-1]['training_time']
            
            # Training time should not increase too dramatically
            assert time_ratio < size_ratio * 2  # At most quadratic scaling
        
        print("Scalability results:")
        for result in results:
            print(f"Size: {result['dataset_size']}, "
                  f"Training: {result['training_time']:.2f}s, "
                  f"Speed: {result['events_per_second']:.1f} events/s")
```

### Documentation Standards

#### Code Documentation

```python
# Example of well-documented NOMAD code
class NOMADEngine:
    """
    NOMAD (Navigating Optimal Model Application to Datastreams) Engine.
    
    The NOMAD engine orchestrates cost-efficient multi-class classification
    by dynamically sequencing machine learning models with varying cost-quality
    tradeoffs. It ensures that the final prediction quality remains within
    ε-tolerance of a designated role model while minimizing computational cost.
    
    Key Features:
        - Dynamic model chaining based on utility maximization
        - Formal quality guarantees through chain safety checks
        - Adaptive distribution tracking with drift detection
        - Support for dependent models (early-exit DNNs, ensembles)
        
    Example:
        >>> from nomad_api import NOMADBuilder, ModelConfig
        >>> models = [
        ...     ModelConfig("fast", "DecisionTreeClassifier", cost=1.0),
        ...     ModelConfig("accurate", "RandomForestClassifier", cost=5.0)
        ... ]
        >>> nomad = (NOMADBuilder()
        ...           .with_models(models)
        ...           .with_role_model("accurate")
        ...           .with_epsilon(0.1)
        ...           .build())
    
    Args:
        models: List of Model instances representing the candidate model portfolio
        config: NOMADConfig instance containing system configuration
        class_names: List of target class names for classification
        initial_class_priors: Optional dictionary of initial class probabilities
        farml_registry: Optional FARML model registry for integration
        
    Attributes:
        models: Dictionary mapping model names to Model instances
        config: System configuration object
        role_model: The designated role model (quality benchmark)
        class_names: List of classification target classes
        adaptive_manager: Handles distribution tracking and adaptation
        
    Raises:
        ValueError: If role_model_name not found in provided models
        RuntimeError: If system is not properly trained before classification
    """
    
    def classify_event(self, event_features: pd.DataFrame) -> Tuple[str, float, List[str]]:
        """
        Classify a single event using the NOMAD strategy.
        
        This method implements the core NOMAD algorithm, which iteratively
        selects models based on utility maximization, checks chain safety
        constraints, and terminates when an exit condition is met.
        
        The algorithm:
        1. Calculates utility for all available models
        2. Selects the model with highest utility (effectiveness/cost ratio)
        3. Checks if adding this model maintains chain safety
        4. Executes the model and checks exit conditions
        5. Updates belief state if continuation is needed
        6. Repeats until exit condition is met or role model is used
        
        Args:
            event_features: DataFrame containing feature values for a single event.
                Must have the same columns as training data (excluding target).
                Shape should be (1, n_features).
                
        Returns:
            A tuple containing:
            - prediction: Predicted class name (str)
            - cost: Total computational cost incurred (float) 
            - chain: List of model names used in the classification chain
            
        Raises:
            RuntimeError: If NOMAD system has not been trained
            ValueError: If event_features has incorrect shape or missing columns
            
        Example:
            >>> event = pd.DataFrame([[1.2, 0.8, 2.1]], columns=['f1', 'f2', 'f3'])
            >>> prediction, cost, chain = nomad.classify_event(event)
            >>> print(f"Predicted {prediction} using {chain} at cost {cost}")
            Predicted 'attack' using ['fast_tree', 'svm'] at cost 4.0
            
        Note:
            The cost is cumulative across all models in the realized chain.
            Chain safety ensures the prediction quality remains within ε-tolerance
            of the role model's quality for the predicted class.
        """
        pass  # Implementation details...
```

#### API Reference Generation

```python
# docs/generate_api_docs.py
"""
Automated API documentation generation for NOMAD
"""

import ast
import inspect
import json
from typing import Dict, Any, List
from nomad_api import *

def extract_class_info(cls) -> Dict[str, Any]:
    """Extract comprehensive information about a class"""
    info = {
        'name': cls.__name__,
        'module': cls.__module__,
        'docstring': inspect.getdoc(cls),
        'methods': [],
        'attributes': [],
        'inheritance': [base.__name__ for base in cls.__bases__ if base != object]
    }
    
    # Extract methods
    for name, method in inspect.getmembers(cls, predicate=inspect.ismethod):
        if not name.startswith('_') or name in ['__init__']:
            method_info = {
                'name': name,
                'signature': str(inspect.signature(method)),
                'docstring': inspect.getdoc(method),
                'parameters': extract_parameters(method),
                'returns': extract_return_info(method)
            }
            info['methods'].append(method_info)
    
    return info

def extract_parameters(func) -> List[Dict[str, Any]]:
    """Extract parameter information from function"""
    signature = inspect.signature(func)
    parameters = []
    
    for name, param in signature.parameters.items():
        param_info = {
            'name': name,
            'type': str(param.annotation) if param.annotation != param.empty else None,
            'default': str(param.default) if param.default != param.empty else None,
            'kind': str(param.kind)
        }
        parameters.append(param_info)
    
    return parameters

def generate_markdown_docs():
    """Generate markdown documentation"""
    classes_to_document = [
        NOMADEngine, NOMADBuilder, Model, ModelConfig, 
        NOMADConfig, AdaptiveConfig
    ]
    
    docs = ["# NOMAD API Reference\n\n"]
    
    for cls in classes_to_document:
        class_info = extract_class_info(cls)
        docs.append(f"## {class_info['name']}\n\n")
        
        if class_info['docstring']:
            docs.append(f"{class_info['docstring']}\n\n")
        
        if class_info['inheritance']:
            docs.append(f"**Inherits from:** {', '.join(class_info['inheritance'])}\n\n")
        
        # Document methods
        if class_info['methods']:
            docs.append("### Methods\n\n")
            for method in class_info['methods']:
                docs.append(f"#### {method['name']}{method['signature']}\n\n")
                if method['docstring']:
                    docs.append(f"{method['docstring']}\n\n")
        
        docs.append("---\n\n")
    
    return ''.join(docs)

if __name__ == "__main__":
    markdown_docs = generate_markdown_docs()
    with open('api_reference.md', 'w') as f:
        f.write(markdown_docs)
    print("API documentation generated: api_reference.md")
```

### Contribution Guidelines

#### Pull Request Process

1. **Fork and Branch**: Create a feature branch from `main`
2. **Code Standards**: Follow PEP 8 and use type hints
3. **Testing**: Add comprehensive tests for new functionality
4. **Documentation**: Update docstrings and documentation
5. **Performance**: Include performance considerations for new features
6. **Backwards Compatibility**: Maintain API compatibility when possible

#### Code Review Checklist

```markdown
# NOMAD Code Review Checklist

## Functionality
- [ ] Code implements the described functionality correctly
- [ ] Edge cases are handled appropriately
- [ ] Error handling is comprehensive and informative
- [ ] Performance implications are considered

## Code Quality
- [ ] Code follows PEP 8 style guidelines
- [ ] Type hints are provided for all functions/methods
- [ ] Docstrings follow Google/NumPy style
- [ ] Variable names are descriptive and clear
- [ ] No code duplication without justification

## Testing
- [ ] Unit tests cover new functionality
- [ ] Integration tests verify end-to-end behavior
- [ ] Performance tests validate scalability
- [ ] All tests pass locally and in CI

## Documentation
- [ ] API documentation is updated
- [ ] Examples are provided for new features
- [ ] Changelog is updated with changes
- [ ] Breaking changes are clearly marked

## NOMAD-Specific
- [ ] Quality guarantees are maintained
- [ ] Cost calculations are accurate
- [ ] Chain safety checks are properly implemented
- [ ] Adaptive behavior is tested with drift scenarios
```

---

## Conclusion

This comprehensive documentation covers every aspect of NOMAD - from theoretical foundations to practical implementation, deployment, and extension. The system provides a robust framework for cost-efficient machine learning deployment while maintaining formal quality guarantees.

Key takeaways:

1. **Theoretical Rigor**: NOMAD is grounded in solid algorithmic foundations with formal safety guarantees
2. **Practical Implementation**: Multiple API levels support different user expertise and use cases
3. **Scalable Architecture**: Designed for integration with broader ML lifecycle management systems
4. **Extensive Tooling**: Comprehensive monitoring, testing, and debugging capabilities
5. **Extensible Design**: Plugin architecture supports custom models, algorithms, and integrations

The NOMAD framework addresses the critical challenge of balancing computational efficiency with prediction quality in real-world ML deployments, making it valuable for production systems where both accuracy and resource constraints matter.

For additional support, examples, and community contributions, visit the project repository and documentation site.
