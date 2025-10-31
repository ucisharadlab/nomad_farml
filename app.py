from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import os
import uuid
import pandas as pd
import numpy as np
import json
import traceback
from werkzeug.utils import secure_filename

# Import the updated NOMAD implementation
from nomad_arima_backend import (
    Model,
    NOMADEngine,
    AdaptivePriorsManager,
    get_classifier_from_config,
    get_quality_metrics
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CUSTOM_MODELS_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'custom_models')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CUSTOM_MODELS_FOLDER'], exist_ok=True)

# Global state
CANDIDATE_MODELS = {
    "Dummy (Uniform)": {
        "name": "Dummy (Uniform)",
        "type": "DummyClassifier",
        "params": {"strategy": "uniform"},
        "cost": 0.1,
        "is_custom": False
    },
    "Decision Tree (Shallow)": {
        "name": "Decision Tree (Shallow)",
        "type": "DecisionTreeClassifier",
        "params": {"max_depth": 3, "random_state": 42},
        "cost": 1.0,
        "is_custom": False
    },
    "Random Forest (Small)": {
        "name": "Random Forest (Small)",
        "type": "RandomForestClassifier",
        "params": {"n_estimators": 50, "max_depth": 10, "random_state": 42},
        "cost": 25.0,
        "is_custom": False
    }
}
current_run_data = {}

# --- Helper Functions ---
def load_and_preprocess_data(csv_file_path):
    """Load and preprocess data from CSV file."""
    try:
        data = pd.read_csv(csv_file_path)
    except Exception as e:
        return None, None, None, None, None, None, None, None, None, f"Error loading CSV: {str(e)}"

    if data.empty:
        return None, None, None, None, None, None, None, None, None, "CSV file is empty."
    if data.shape[1] < 2:
        return None, None, None, None, None, None, None, None, None, "CSV must have at least two columns."

    # Separate features and target
    X_df = data.iloc[:, :-1]
    y_raw_series = data.iloc[:, -1]

    # Encode target variable
    label_encoder = LabelEncoder()
    y_encoded_array = label_encoder.fit_transform(y_raw_series)
    unique_labels_encoded = np.sort(np.unique(y_encoded_array))
    
    # Create class mappings
    class_map_num_to_str = {
        int(enc): str(name)
        for enc, name in zip(unique_labels_encoded, label_encoder.inverse_transform(unique_labels_encoded))
    }
    unique_class_names_ordered_str = [class_map_num_to_str[int(el)] for el in unique_labels_encoded]

    # Create preprocessor
    numerical_cols = X_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    transformers_list = []
    if numerical_cols:
        transformers_list.append(('num', StandardScaler(), numerical_cols))
    if categorical_cols:
        transformers_list.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols))
    
    preprocessor = ColumnTransformer(transformers=transformers_list, remainder='passthrough')
    
    return (data, X_df, y_encoded_array, label_encoder, unique_labels_encoded,
            class_map_num_to_str, unique_class_names_ordered_str, preprocessor, X_df.columns.tolist(), None)

def train_and_evaluate_models(X_full, y_encoded_full, preprocessor, class_map_numeric_to_str,
                             unique_enc_labels, candidate_configs_dict, custom_models_path):
    """Train and evaluate all models."""
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
        print(f"Training model '{name}'...")
        try:
            # Instantiate classifier
            clf_instance = get_classifier_from_config(config, custom_models_path)
            model = Model(name, clf_instance, float(config['cost']), preprocessor=preprocessor)
            
            # Train
            if X_train.shape[0] > 0:
                model.train(X_train, y_train)
            
            # Evaluate
            if X_test.shape[0] > 0:
                model.evaluate(X_test, y_test, labels=unique_enc_labels, class_names_map=class_map_numeric_to_str)
            
            trained_models_dict[name] = model
            model_summaries.append({
                "name": name,
                "accuracy": model.accuracy,
                "f1_score_weighted": model.avg_metrics.get('f1-score', 0.0),
                "cost": model.cost
            })
        except Exception as e:
            print(f"ERROR: Failed to train model '{name}': {e}")
            model_summaries.append({
                "name": f"{name} (FAILED)",
                "accuracy": 0,
                "f1_score_weighted": 0,
                "cost": config['cost']
            })

    return trained_models_dict, model_summaries, X_test, y_test, y_train

# --- API Endpoints ---
@app.route('/api/hello')
def hello():
    return jsonify({"message": "NOMAD Backend is running!"})

@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify(list(CANDIDATE_MODELS.values()))

@app.route('/api/models', methods=['POST'])
def add_model():
    data = request.json
    model_name = data.get('name')
    if not model_name or model_name in CANDIDATE_MODELS:
        return jsonify({"error": "Model name is missing or already exists."}), 400
    
    CANDIDATE_MODELS[model_name] = {
        "name": model_name,
        "type": data.get('type'),
        "params": data.get('params', {}),
        "cost": data.get('cost', 1.0),
        "is_custom": data.get('is_custom', False),
        "module_name": data.get('module_name'),
        "class_name": data.get('class_name')
    }
    return jsonify({"message": f"Model '{model_name}' added successfully."}), 201

@app.route('/api/models/<string:model_name>', methods=['DELETE'])
def remove_model(model_name):
    if model_name not in CANDIDATE_MODELS:
        return jsonify({"error": "Model not found."}), 404
    del CANDIDATE_MODELS[model_name]
    return jsonify({"message": f"Model '{model_name}' removed."})

@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.csv'):
        filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load and preprocess data
        current_run_data.clear()
        data_df, X_df, y_enc_arr, label_enc, unique_lab_enc, cls_map_num_to_str, \
        unique_cls_names_ord_str, preproc, x_col_names, err_msg = load_and_preprocess_data(filepath)

        if err_msg:
            return jsonify({"error": err_msg}), 400

        current_run_data.update({
            'data_df': data_df,
            'X_column_names': x_col_names,
            'X_full': X_df,
            'y_encoded_full': y_enc_arr,
            'class_map_numeric_to_str': cls_map_num_to_str,
            'unique_class_names_ordered_str': unique_cls_names_ord_str,
            'unique_labels_encoded': unique_lab_enc.tolist(),
            'preprocessor_for_models': preproc
        })

        # Train models
        try:
            trained_models, model_summaries, X_test, y_test, y_train = train_and_evaluate_models(
                current_run_data['X_full'],
                current_run_data['y_encoded_full'],
                current_run_data['preprocessor_for_models'],
                current_run_data['class_map_numeric_to_str'],
                current_run_data['unique_labels_encoded'],
                CANDIDATE_MODELS,
                app.config['CUSTOM_MODELS_FOLDER']
            )
        except Exception as e:
            return jsonify({"error": f"Model training error: {str(e)}"}), 500

        current_run_data.update({
            'trained_models_dict': trained_models,
            'X_test_data': X_test,
            'y_test_data': y_test
        })
        
        # Calculate initial priors from training data
        initial_priors = {name: 1.0/len(unique_cls_names_ord_str) for name in unique_cls_names_ord_str}
        if y_train is not None and len(y_train) > 0:
            unique, counts = np.unique(y_train, return_counts=True)
            priors_num = dict(zip(unique, counts/len(y_train)))
            for num, name in cls_map_num_to_str.items():
                initial_priors[name] = float(priors_num.get(num, 0.0))
        
        current_run_data['initial_class_priors_str_keys'] = initial_priors
        
        return jsonify({
            "message": "File uploaded and models trained.",
            "classes_str": unique_cls_names_ord_str,
            "model_summaries": model_summaries,
            "initial_priors_str_keys": initial_priors,
            "candidate_model_names": list(CANDIDATE_MODELS.keys()),
            "num_features": len(x_col_names)
        }), 200
    
    return jsonify({"error": "Invalid file type."}), 400

@app.route('/api/nomad_stream', methods=['GET'])
def nomad_stream_endpoint():
    if 'trained_models_dict' not in current_run_data:
        def err_gen():
            yield f"data: {json.dumps({'type':'error','message':'Prerequisite data missing.'})}\n\n"
        return Response(err_gen(), mimetype='text/event-stream')
    
    try:
        # Parse configuration
        config = {
            "role_model_name": request.args.get('role_model_name'),
            "epsilon": float(request.args.get('epsilon', 0.1)),
            "quality_metric": request.args.get('quality_metric', 'recall'),
            "safety_mode": request.args.get('safety_mode', 'global'),
            "passthrough_mode": request.args.get('passthrough_mode', 'relaxed'),
            "batch_size": int(request.args.get('batch_size', 10))
        }
        
        # Parse adaptive configuration
        adaptive_config = {
            "ph_threshold": float(request.args.get('ph_threshold', 50.0)),
            "ph_delta": float(request.args.get('ph_delta', 0.005)),
            "buffer_size": int(request.args.get('buffer_size', 1000)),
            "min_buffer_for_arima": int(request.args.get('min_buffer_for_arima', 20)),
            "enable_arima": request.args.get('enable_arima', 'true').lower() == 'true'
        }
        
        # Parse ARIMA order
        arima_order_str = request.args.get('arima_order', '1,0,1')
        try:
            arima_order = tuple(map(int, arima_order_str.split(',')))
            if len(arima_order) != 3:
                arima_order = (1, 0, 1)
        except:
            arima_order = (1, 0, 1)
        
        adaptive_config["arima_order"] = arima_order
        
    except (ValueError, json.JSONDecodeError) as e:
        def err_gen():
            yield f"data: {json.dumps({'type':'error','message':f'Invalid params: {e}'})}\n\n"
        return Response(err_gen(), mimetype='text/event-stream')

    if not config["role_model_name"] or config["role_model_name"] not in current_run_data['trained_models_dict']:
        def err_gen():
            yield f"data: {json.dumps({'type':'error','message':'Invalid role model.'})}\n\n"
        return Response(err_gen(), mimetype='text/event-stream')

    def generate_updates():
        try:
            # Initialize NOMAD Engine
            nomad_engine = NOMADEngine(
                models_dict=current_run_data['trained_models_dict'],
                role_model_name=config["role_model_name"],
                epsilon=config["epsilon"],
                initial_class_priors_str_keys=current_run_data['initial_class_priors_str_keys'],
                unique_classes_ordered_str=current_run_data['unique_class_names_ordered_str'],
                class_map_numeric_to_str=current_run_data['class_map_numeric_to_str'],
                quality_metric=config["quality_metric"],
                safety_mode=config["safety_mode"],
                passthrough_mode=config["passthrough_mode"],
                adaptive_config=adaptive_config
            )
            
            # Get role model info
            role_model = nomad_engine.role_model
            
            # Send setup information
            yield f"data: {json.dumps({
                'type': 'setup',
                'role_model_name': config['role_model_name'],
                'role_model_accuracy': float(role_model.accuracy),
                'role_model_cost': float(role_model.cost),
                'exit_classes_info': {
                    name: list(model.exit_classes)
                    for name, model in nomad_engine.models_dict.items()
                },
                'initial_priors': nomad_engine.initial_class_priors_str_keys,
                'adaptive_stats': nomad_engine.adaptive_manager.get_statistics()
            })}\n\n"
            
            # Stream evaluation results
            for update in nomad_engine.stream_evaluate(
                current_run_data['X_test_data'],
                current_run_data['y_test_data'],
                batch_size=config["batch_size"]
            ):
                yield f"data: {json.dumps(update)}\n\n"
                
        except Exception as e:
            error_msg = {
                'type': 'error',
                'message': f'Simulation error: {str(e)}',
                'traceback': traceback.format_exc()
            }
            yield f"data: {json.dumps(error_msg)}\n\n"
    
    return Response(generate_updates(), mimetype='text/event-stream')

@app.route('/api/adaptive_config', methods=['GET'])
def get_adaptive_config():
    """Get default adaptive configuration settings."""
    return jsonify({
        "ph_threshold": 50.0,
        "ph_delta": 0.005,
        "buffer_size": 1000,
        "min_buffer_for_arima": 20,
        "arima_order": [1, 0, 1],
        "enable_arima": True
    })

@app.route('/api/nomad_config', methods=['GET'])
def get_nomad_config():
    """Get available NOMAD configuration options."""
    return jsonify({
        "quality_metrics": ["recall", "precision", "f1-score"],
        "safety_modes": ["global", "class-based"],
        "passthrough_modes": ["relaxed", "conservative"],
        "epsilon_range": {"min": 0.0, "max": 0.5, "default": 0.1}
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001, threaded=True)
