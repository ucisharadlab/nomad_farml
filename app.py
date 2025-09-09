from flask import Flask, request, jsonify, render_template, Response, url_for
from flask_cors import CORS # Import CORS
import os
import uuid
import pandas as pd
import numpy as np
import json
import traceback
import importlib.util
from werkzeug.utils import secure_filename

from nomad_arima_backend import (
    train_evaluate_individual_models_for_flask,
    run_nomad_simulation_for_flask_streamed,
    load_and_preprocess_data_for_flask,
    get_classifier_from_config,
    AdaptivePriorsManager  # Import the new class
)

app = Flask(__name__)
CORS(app)

# Configuration for file uploads
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CUSTOM_MODELS_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'custom_models')
# Configuration for the cascade result images
app.config['RESULT_FOLDER'] = os.path.join('static', 'cascade_result')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CUSTOM_MODELS_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True) # Ensure this folder exists

# --- Global State Management ---
CANDIDATE_MODELS = {
    "Dummy (Uniform)": {"name": "Dummy (Uniform)", "type": "DummyClassifier", "params": {"strategy": "uniform"}, "cost": 0.1, "is_custom": False},
    "Decision Tree (Shallow)": {"name": "Decision Tree (Shallow)", "type": "DecisionTreeClassifier", "params": {"max_depth": 3, "random_state": 42}, "cost": 1.0, "is_custom": False},
    "Random Forest (Small)": {"name": "Random Forest (Small)", "type": "RandomForestClassifier", "params": {"n_estimators": 50, "max_depth": 10, "random_state": 42}, "cost": 25.0, "is_custom": False}
}
current_run_data = {}

# --- API Endpoints ---
@app.route('/api/hello')
def hello():
    return jsonify({"message": "Backend is running!"})

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
        "name": model_name, "type": data.get('type'), "params": data.get('params', {}),
        "cost": data.get('cost', 1.0), "is_custom": data.get('is_custom', False),
        "module_name": data.get('module_name'), "class_name": data.get('class_name')
    }
    return jsonify({"message": f"Model '{model_name}' added successfully."}), 201

@app.route('/api/models/<string:model_name>', methods=['DELETE'])
def remove_model(model_name):
    if model_name not in CANDIDATE_MODELS:
        return jsonify({"error": "Model not found."}), 404
    del CANDIDATE_MODELS[model_name]
    return jsonify({"message": f"Model '{model_name}' removed."})

@app.route('/api/upload_model_file', methods=['POST'])
def upload_model_file():
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    class_name = request.form.get('className')
    if file.filename == '' or not class_name:
        return jsonify({"error": "No selected file or class name provided"}), 400

    if file and file.filename.endswith('.py'):
        module_name = f"custom_model_{uuid.uuid4().hex}"
        filepath = os.path.join(app.config['CUSTOM_MODELS_FOLDER'], f"{module_name}.py")
        try:
            file.save(filepath)
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            custom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_module)
            if not hasattr(custom_module, class_name):
                os.remove(filepath)
                raise AttributeError(f"Class '{class_name}' not found.")
            return jsonify({"message": "Custom model uploaded.", "module_name": module_name, "class_name": class_name}), 201
        except Exception as e:
            if os.path.exists(filepath): os.remove(filepath)
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Invalid file type."}), 400

@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.csv'):
        filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        current_run_data.clear()
        data_df, X_df, y_enc_arr, _, unique_lab_enc, cls_map_num_to_str, \
        unique_cls_names_ord_str, preproc, x_col_names, X_by_cls_str, \
        y_by_cls_str, err_msg = load_and_preprocess_data_for_flask(filepath)

        if err_msg:
            return jsonify({"error": err_msg}), 400

        current_run_data.update({
            'X_column_names': x_col_names, 'X_full': X_df, 'y_encoded_full': y_enc_arr,
            'class_map_numeric_to_str': cls_map_num_to_str,
            'unique_class_names_ordered_str': unique_cls_names_ord_str,
            'unique_labels_encoded': unique_lab_enc.tolist(),
            'preprocessor_for_models': preproc, 'X_by_class_str_keys': X_by_cls_str,
            'y_by_class_str_keys': y_by_cls_str
        })

        try:
            trained_mdls, mdl_summaries, X_tst, y_tst, y_trn = train_evaluate_individual_models_for_flask(
                current_run_data['X_full'], current_run_data['y_encoded_full'], current_run_data['preprocessor_for_models'],
                current_run_data['class_map_numeric_to_str'], current_run_data['unique_labels_encoded'],
                CANDIDATE_MODELS, app.config['CUSTOM_MODELS_FOLDER']
            )
        except Exception as e:
            return jsonify({"error": f"Model training error: {str(e)}"}), 500

        current_run_data.update({'trained_models_dict': trained_mdls, 'X_test_data_overall': X_tst, 'y_test_encoded_data_overall': y_tst})
        
        initial_priors = {name: 1.0/len(unique_cls_names_ord_str) for name in unique_cls_names_ord_str} if unique_cls_names_ord_str else {}
        if y_trn is not None and len(y_trn) > 0:
            unique, counts = np.unique(y_trn, return_counts=True)
            priors_num = dict(zip(unique, counts/len(y_trn)))
            for num, name in cls_map_num_to_str.items():
                initial_priors[name] = float(priors_num.get(num, 0.0))
        current_run_data['initial_class_priors_str_keys'] = initial_priors
        
        # Add data summary to response
        data_head_json = data_df.head().to_json(orient='split') if not data_df.empty else None

        return jsonify({
            "message": "File uploaded and models trained.", "classes_str": unique_cls_names_ord_str,
            "model_summaries": mdl_summaries, "initial_priors_str_keys": initial_priors,
            "candidate_model_names": list(CANDIDATE_MODELS.keys()),
            "num_features": len(x_col_names), "data_head": data_head_json
        }), 200
    return jsonify({"error": "Invalid file type."}), 400

@app.route('/api/visualizations', methods=['GET'])
def get_all_visuals():
    """
    Returns a list of image URLs based on the selected option.
    Mirrors the logic from the user-provided `/get_all` endpoint.
    """
    selected_value = request.args.get('selected')
    if not selected_value:
        return jsonify({"error": "No selection provided."}), 400

    # This mapping replaces the long if/elif chain
    VISUALIZATION_MAP = {
        'model_option1': ['decision_tree.png'],
        'model_option2': ['decision_tree_0.png'],
        'model_option3': ['decision_tree_2.png'],
        'model_option4': ['decision_tree_3.png'],
        'model_option5': ['decision_tree_4.png'],
        'test_method_option1': ['acc_f1.png', 'nodes.png'],
        'test_method_option2': ['pred_path_anim.gif'],
        'attack_option1': ['feature_corr.png'],
        'attack_option2': ['atk_acc_f1.png'],
        'adtrain_option1': ['adtrain_acc_f1.png', 'adtrain_atk_acc_f1.png'],
        'adtrain_option2': ['boundary_anim.gif']
    }

    image_files = VISUALIZATION_MAP.get(selected_value, [])
    
    # Generate full URLs for the frontend
    # Assumes images are in `static/cascade_result/`
    image_urls = [url_for('static', filename=f'cascade_result/{img}') for img in image_files]

    return jsonify({"plots": image_urls})


@app.route('/api/nomad_event_stream', methods=['GET'])
def nomad_event_stream_endpoint():
    if 'trained_models_dict' not in current_run_data:
        def err_gen(): yield f"data: {json.dumps({'type':'error','message':'Prerequisite data missing.'})}\n\n"
        return Response(err_gen(), mimetype='text/event-stream')
    
    try:
        # Parse basic configuration
        config = {
            "role_model_name": request.args.get('role_model_name'),
            "epsilon": float(request.args.get('epsilon', 0.05)),
            "quality_metric_for_ec": request.args.get('quality_metric_for_ec', 'f1-score'),
            "safety_check_type": request.args.get('safety_check_type', 'conservative'),
            "batch_size": int(request.args.get('batch_size', 10)),
            "workload_phases": json.loads(request.args.get('workload_phases', '[]'))
        }
        
        # Parse adaptive configuration parameters with defaults
        adaptive_config = {
            "ph_threshold": float(request.args.get('ph_threshold', 50.0)),
            "ph_delta": float(request.args.get('ph_delta', 0.005)),
            "buffer_size": int(request.args.get('buffer_size', 100)),
            "min_buffer_for_arima": int(request.args.get('min_buffer_for_arima', 20)),
            "enable_arima": request.args.get('enable_arima', 'true').lower() == 'true'
        }
        
        # Parse ARIMA order (p,d,q)
        arima_order_str = request.args.get('arima_order', '1,0,1')
        try:
            arima_order = tuple(map(int, arima_order_str.split(',')))
            if len(arima_order) != 3:
                raise ValueError("ARIMA order must have exactly 3 components")
        except:
            arima_order = (1, 0, 1)  # Default fallback
            
        adaptive_config["arima_order"] = arima_order
        
        # Additional parameters
        config.update({
            "adaptive_config": adaptive_config,
            "enable_dependent_models": request.args.get('enable_dependent_models', 'false').lower() == 'true'
        })
        
    except (ValueError, json.JSONDecodeError) as e:
        def err_gen(): yield f"data: {json.dumps({'type':'error','message':f'Invalid params: {e}'})}\n\n"
        return Response(err_gen(), mimetype='text/event-stream')

    if not config["role_model_name"] or config["role_model_name"] not in current_run_data['trained_models_dict']:
        def err_gen(): yield f"data: {json.dumps({'type':'error','message':'Invalid role model.'})}\n\n"
        return Response(err_gen(), mimetype='text/event-stream')

    def generate_updates():
        try:
            for update in run_nomad_simulation_for_flask_streamed(
                current_run_data['trained_models_dict'], config["role_model_name"], config["epsilon"],
                current_run_data['initial_class_priors_str_keys'], current_run_data['unique_class_names_ordered_str'],
                current_run_data['class_map_numeric_to_str'], config["quality_metric_for_ec"], config["safety_check_type"],
                current_run_data['X_test_data_overall'], current_run_data['y_test_encoded_data_overall'],
                CANDIDATE_MODELS, current_run_data['X_column_names'],
                batch_size=config["batch_size"], 
                workload_phases=config["workload_phases"],
                X_by_class_str_keys=current_run_data['X_by_class_str_keys'], 
                y_by_class_str_keys=current_run_data['y_by_class_str_keys'],
                adaptive_config=config["adaptive_config"],
                enable_dependent_models=config["enable_dependent_models"]
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
    """Get current adaptive configuration settings"""
    default_config = {
        "ph_threshold": 50.0,
        "ph_delta": 0.005,
        "buffer_size": 100,
        "min_buffer_for_arima": 20,
        "arima_order": [1, 0, 1],
        "enable_arima": True
    }
    return jsonify(default_config)

@app.route('/api/adaptive_config', methods=['POST'])
def set_adaptive_config():
    """Set adaptive configuration settings"""
    try:
        config = request.json
        
        # Validate configuration
        validated_config = {
            "ph_threshold": float(config.get('ph_threshold', 50.0)),
            "ph_delta": float(config.get('ph_delta', 0.005)),
            "buffer_size": int(config.get('buffer_size', 100)),
            "min_buffer_for_arima": int(config.get('min_buffer_for_arima', 20)),
            "enable_arima": bool(config.get('enable_arima', True))
        }
        
        # Handle ARIMA order
        arima_order = config.get('arima_order', [1, 0, 1])
        if isinstance(arima_order, list) and len(arima_order) == 3:
            validated_config["arima_order"] = tuple(map(int, arima_order))
        else:
            validated_config["arima_order"] = (1, 0, 1)
        
        # Store in current run data for future simulations
        current_run_data['adaptive_config'] = validated_config
        
        return jsonify({
            "message": "Adaptive configuration updated successfully",
            "config": {
                **validated_config,
                "arima_order": list(validated_config["arima_order"])  # Convert back to list for JSON
            }
        }), 200
        
    except (ValueError, KeyError, TypeError) as e:
        return jsonify({"error": f"Invalid adaptive configuration: {str(e)}"}), 400

@app.route('/api/system_status', methods=['GET'])
def get_system_status():
    """Get current system status and health information"""
    status = {
        "models_loaded": len(CANDIDATE_MODELS),
        "data_loaded": 'X_full' in current_run_data,
        "models_trained": 'trained_models_dict' in current_run_data,
        "available_models": list(CANDIDATE_MODELS.keys()),
        "system_ready": all([
            'trained_models_dict' in current_run_data,
            'X_test_data_overall' in current_run_data,
            current_run_data.get('trained_models_dict')
        ])
    }
    
    if status["models_trained"]:
        trained_models = current_run_data.get('trained_models_dict', {})
        status["trained_model_count"] = len(trained_models)
        status["trained_model_names"] = list(trained_models.keys())
        
        # Add model performance summaries
        model_performance = {}
        for name, model in trained_models.items():
            if hasattr(model, 'accuracy') and hasattr(model, 'avg_metrics'):
                model_performance[name] = {
                    "accuracy": float(model.accuracy),
                    "avg_f1": float(model.avg_metrics.get('f1-score', 0.0)),
                    "cost": float(model.cost)
                }
        status["model_performance"] = model_performance
    
    return jsonify(status)

@app.route('/api/test_adaptive', methods=['POST'])
def test_adaptive_functionality():
    """Test endpoint for adaptive functionality"""
    try:
        if 'unique_class_names_ordered_str' not in current_run_data:
            return jsonify({"error": "No data loaded"}), 400
        
        # Create a test adaptive manager
        class_names = current_run_data['unique_class_names_ordered_str']
        initial_priors = current_run_data.get('initial_class_priors_str_keys', {})
        
        test_config = request.json.get('adaptive_config', {})
        
        adaptive_manager = AdaptivePriorsManager(
            class_names=class_names,
            initial_priors=initial_priors,
            ph_threshold=test_config.get('ph_threshold', 50.0),
            ph_delta=test_config.get('ph_delta', 0.005),
            buffer_size=test_config.get('buffer_size', 100),
            min_buffer_for_arima=test_config.get('min_buffer_for_arima', 20),
            arima_order=tuple(test_config.get('arima_order', [1, 0, 1])),
            enable_arima=test_config.get('enable_arima', True)
        )
        
        # Simulate some events
        test_results = []
        for i, class_name in enumerate(class_names[:10]):  # Test first 10 or fewer classes
            updated_priors, drift_detected = adaptive_manager.update(class_name)
            test_results.append({
                "event": i + 1,
                "observed_class": class_name,
                "updated_priors": updated_priors,
                "drift_detected": drift_detected
            })
        
        final_stats = adaptive_manager.get_statistics()
        
        return jsonify({
            "message": "Adaptive functionality test completed",
            "test_results": test_results,
            "final_statistics": final_stats,
            "arima_available": hasattr(adaptive_manager, 'enable_arima') and adaptive_manager.enable_arima
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": f"Adaptive test failed: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001, threaded=True)