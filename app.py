from flask import Flask, request, jsonify, render_template, Response
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
    get_classifier_from_config # New function to instantiate models
)

app = Flask(__name__)
# Configuration for file uploads
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CUSTOM_MODELS_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'custom_models')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CUSTOM_MODELS_FOLDER'], exist_ok=True)

# --- Global State Management ---
# In a real multi-user app, this would be in a database or user session.
# For this visualizer, a global dict is sufficient.

# Initial default models
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
# This dictionary will store data for the current CSV run
current_run_data = {}


@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

# --- API for Model Management ---

@app.route('/api/models', methods=['GET'])
def get_models():
    """Returns the current list of candidate models."""
    return jsonify(list(CANDIDATE_MODELS.values()))

@app.route('/api/models', methods=['POST'])
def add_model():
    """Adds a new model configuration to the candidate list."""
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
        "module_name": data.get('module_name'), # For custom models
        "class_name": data.get('class_name') # For custom models
    }
    app.logger.info(f"Added model: {model_name}")
    return jsonify({"message": f"Model '{model_name}' added successfully."}), 201

@app.route('/api/models/<string:model_name>', methods=['DELETE'])
def remove_model(model_name):
    """Removes a model from the candidate list."""
    if model_name not in CANDIDATE_MODELS:
        return jsonify({"error": "Model not found."}), 404
    
    # Check if the model has an associated file and remove it
    model_config = CANDIDATE_MODELS[model_name]
    if model_config.get('is_custom') and model_config.get('module_name'):
        module_path = os.path.join(app.config['CUSTOM_MODELS_FOLDER'], f"{model_config['module_name']}.py")
        if os.path.exists(module_path):
            try:
                os.remove(module_path)
                app.logger.info(f"Removed custom model file: {module_path}")
            except Exception as e:
                app.logger.error(f"Error removing custom model file {module_path}: {e}")
                # Don't block removal of config even if file delete fails
    
    del CANDIDATE_MODELS[model_name]
    app.logger.info(f"Removed model: {model_name}")
    return jsonify({"message": f"Model '{model_name}' removed."})

@app.route('/api/upload_model_file', methods=['POST'])
def upload_model_file():
    """Handles the upload of a custom Python model file."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    class_name = request.form.get('className')

    if file.filename == '' or not class_name:
        return jsonify({"error": "No selected file or class name provided"}), 400

    if file and file.filename.endswith('.py'):
        # Create a unique name for the module to avoid conflicts
        module_name = f"custom_model_{uuid.uuid4().hex}"
        filename = f"{module_name}.py"
        filepath = os.path.join(app.config['CUSTOM_MODELS_FOLDER'], filename)
        
        try:
            file.save(filepath)
            # Test if the module can be imported and the class exists
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            custom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_module)
            if not hasattr(custom_module, class_name):
                os.remove(filepath) # Clean up invalid file
                raise AttributeError(f"Class '{class_name}' not found in '{file.filename}'.")
            
            app.logger.info(f"Custom model file '{file.filename}' uploaded and validated as '{filename}'.")
            return jsonify({
                "message": "Custom model uploaded successfully.",
                "module_name": module_name,
                "class_name": class_name
            }), 201
        except Exception as e:
            app.logger.error(f"Failed to upload or validate custom model: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Invalid file type, please upload a .py file."}), 400


# --- API for Main Application Logic ---

@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    """Handles CSV upload, data preprocessing, and initial model training."""
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.csv'):
        # Create a unique filename for the uploaded data
        safe_original_filename = secure_filename(file.filename)
        filename = f"{uuid.uuid4()}_{safe_original_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        current_run_data.clear()

        # Preprocess the data
        data_df, X_df, y_enc_arr, _, unique_lab_enc, \
        cls_map_num_to_str, unique_cls_names_ord_str, preproc, x_col_names, \
        X_by_cls_str, y_by_cls_str, err_msg = load_and_preprocess_data_for_flask(filepath)

        if err_msg:
            if os.path.exists(filepath): os.remove(filepath)
            return jsonify({"error": err_msg}), 400

        current_run_data.update({
            'X_column_names': x_col_names, 'X_full': X_df, 'y_encoded_full': y_enc_arr,
            'class_map_numeric_to_str': cls_map_num_to_str,
            'unique_class_names_ordered_str': unique_cls_names_ord_str,
            'unique_labels_encoded': unique_lab_enc.tolist(),
            'preprocessor_for_models': preproc,
            'X_by_class_str_keys': X_by_cls_str, 'y_by_class_str_keys': y_by_cls_str
        })

        # Train and evaluate the models from the dynamic configuration
        try:
            # Pass the candidate model configs and the custom models path
            trained_mdls, mdl_summaries, X_tst, y_tst, y_trn = train_evaluate_individual_models_for_flask(
                X_full=current_run_data['X_full'],
                y_encoded_full=current_run_data['y_encoded_full'],
                preprocessor=current_run_data['preprocessor_for_models'],
                class_map_numeric_to_str=current_run_data['class_map_numeric_to_str'],
                unique_enc_labels=current_run_data['unique_labels_encoded'],
                candidate_configs_dict=CANDIDATE_MODELS, # Pass the dynamic dict
                custom_models_path=app.config['CUSTOM_MODELS_FOLDER'] # Pass path
            )
        except Exception as e_train_eval:
            if os.path.exists(filepath): os.remove(filepath)
            app.logger.error(f"Error during model training: {e_train_eval}", exc_info=True)
            return jsonify({"error": f"An unexpected error occurred during model training: {str(e_train_eval)}"}), 500

        current_run_data.update({
            'trained_models_dict': trained_mdls,
            'X_test_data_overall': X_tst, 'y_test_encoded_data_overall': y_tst
        })
        
        # Calculate initial class priors from the training set
        initial_priors_dict_str_keys = {}
        if y_trn is not None and len(y_trn) > 0:
            unique_tr, counts_tr = np.unique(y_trn, return_counts=True)
            temp_priors_num = {cls_num: count/len(y_trn) for cls_num, count in zip(unique_tr, counts_tr)}
            for num_lab_global in unique_lab_enc:
                cls_name_str = cls_map_num_to_str.get(int(num_lab_global))
                if cls_name_str: initial_priors_dict_str_keys[cls_name_str] = float(temp_priors_num.get(num_lab_global, 0.0))
        else: # Fallback
            num_classes = len(unique_cls_names_ord_str)
            initial_priors_dict_str_keys = {name: 1.0/num_classes for name in unique_cls_names_ord_str} if num_classes > 0 else {}
        current_run_data['initial_class_priors_str_keys'] = initial_priors_dict_str_keys

        return jsonify({
            "message": "File uploaded and base models trained.", "filename": os.path.basename(filepath),
            "classes_str": unique_cls_names_ord_str, "model_summaries": mdl_summaries,
            "initial_priors_str_keys": initial_priors_dict_str_keys,
            "candidate_model_names": list(CANDIDATE_MODELS.keys())
        }), 200

    return jsonify({"error": "Invalid file type, please upload a CSV."}), 400

@app.route('/api/nomad_event_stream', methods=['GET'])
def nomad_event_stream_endpoint():
    """Endpoint to stream NOMAD simulation results."""
    # Check if prerequisite data from a CSV upload exists
    if 'trained_models_dict' not in current_run_data:
        def err_gen(): yield f"data: {json.dumps({'type':'error','message':'Prerequisite data missing. Please upload a CSV file and train models first.'})}\n\n"
        return Response(err_gen(), mimetype='text/event-stream')

    # Extract query parameters for the simulation
    try:
        role_model_name = request.args.get('role_model_name')
        workload_phases_json = request.args.get('workload_phases', '[]')
        workload_phases = json.loads(workload_phases_json)
        epsilon = float(request.args.get('epsilon', 0.05))
        batch_size = int(request.args.get('batch_size', 10))
        adaptive_window = int(request.args.get('adaptive_update_window', 100))
        adaptive_beta = float(request.args.get('adaptive_beta', 0.3))
        quality_metric_ec = request.args.get('quality_metric_for_ec', 'f1-score')
        safety_check = request.args.get('safety_check_type', 'conservative')
    except (ValueError, json.JSONDecodeError) as e:
         def err_gen(): yield f"data: {json.dumps({'type':'error','message':f'Invalid NOMAD parameters: {e}'})}\n\n"
         return Response(err_gen(), mimetype='text/event-stream')

    if not role_model_name or role_model_name not in current_run_data.get('trained_models_dict', {}):
        def err_gen(): yield f"data: {json.dumps({'type':'error','message':f'Role model {role_model_name} is invalid or not trained.'})}\n\n"
        return Response(err_gen(), mimetype='text/event-stream')

    def generate_updates():
        """Generator function to run the simulation and yield updates."""
        try:
            # The CANDIDATE_MODELS dict is now the source of truth for model configs
            for update in run_nomad_simulation_for_flask_streamed(
                current_run_data['trained_models_dict'], role_model_name, epsilon,
                current_run_data['initial_class_priors_str_keys'],
                current_run_data['unique_class_names_ordered_str'],
                current_run_data['class_map_numeric_to_str'], quality_metric_ec,
                safety_check, current_run_data['X_test_data_overall'],
                current_run_data['y_test_encoded_data_overall'],
                CANDIDATE_MODELS, current_run_data['X_column_names'],
                batch_size=batch_size, adaptive_update_window=adaptive_window,
                adaptive_beta=adaptive_beta,
                workload_phases=workload_phases,
                X_by_class_str_keys=current_run_data['X_by_class_str_keys'],
                y_by_class_str_keys=current_run_data['y_by_class_str_keys']
            ):
                yield f"data: {json.dumps(update)}\n\n"
        except Exception as e_stream:
            tb_str = traceback.format_exc()
            app.logger.error(f"STREAMING ERROR: {str(e_stream)}\n{tb_str}")
            yield f"data: {json.dumps({'type':'error','message':f'Simulation error: {str(e_stream)}'})}\n\n"
            
    return Response(generate_updates(), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run(debug=True, port=5001, threaded=True)

