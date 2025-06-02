from flask import Flask, request, jsonify, render_template, Response
import os
import uuid
import pandas as pd
import numpy as np
import json

# Assuming nomad_backend.py is in the same directory or accessible via PYTHONPATH
from nomad_arima_backend import (
    CANDIDATE_MODELS_CONFIG,
    load_and_preprocess_data_for_flask,
    train_evaluate_individual_models_for_flask,
    run_nomad_simulation_for_flask_streamed
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Simple global storage; not suitable for production with concurrent users.
# Consider using Flask sessions or a more robust per-request/per-user storage.
current_run_data = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/candidate_models', methods=['GET'])
def get_candidate_models_endpoint():
    return jsonify([{"name": name, "cost": float(cost)} for name, _, cost in CANDIDATE_MODELS_CONFIG])

@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.csv'):
        # Sanitize filename to prevent directory traversal or other issues
        # Though uuid makes it unique, sanitizing original name part is good practice
        safe_original_filename = "".join(c if c.isalnum() or c in ['.', '_'] else '_' for c in file.filename)
        filename = str(uuid.uuid4()) + "_" + safe_original_filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
        except Exception as e:
            # Log the full error server-side for debugging
            app.logger.error(f"Failed to save uploaded file '{filename}': {e}")
            return jsonify({"error": f"Failed to save file. Please try again."}), 500
        
        current_run_data.clear() # Clear data from any previous run

        data_df, X_df, y_enc, _, unique_lab_enc, \
        cls_map_num_to_str, unique_cls_names_ord_str, preproc, x_col_names, err_msg = load_and_preprocess_data_for_flask(filepath)

        if err_msg:
            if os.path.exists(filepath): os.remove(filepath) # Clean up failed upload
            return jsonify({"error": err_msg}), 400
        
        # Check for empty data after preprocessing
        if X_df is None or X_df.empty or y_enc is None or len(y_enc) == 0:
            if os.path.exists(filepath): os.remove(filepath)
            return jsonify({"error": "No valid data (features or target) found in the CSV after preprocessing."}), 400

        current_run_data['X_column_names'] = x_col_names
        current_run_data['X_full'] = X_df # This should be a DataFrame
        current_run_data['y_encoded_full'] = y_enc
        current_run_data['class_map_numeric_to_str'] = cls_map_num_to_str
        current_run_data['unique_class_names_ordered_str'] = unique_cls_names_ord_str
        current_run_data['unique_labels_encoded'] = unique_lab_enc.tolist() # Ensure it's a list for JSON
        current_run_data['preprocessor_for_models'] = preproc

        try:
            trained_mdls, mdl_summaries, X_tst, y_tst, y_trn = train_evaluate_individual_models_for_flask(
                current_run_data['X_full'], current_run_data['y_encoded_full'],
                current_run_data['preprocessor_for_models'],
                current_run_data['class_map_numeric_to_str'],
                current_run_data['unique_labels_encoded'], # Pass numeric labels
                CANDIDATE_MODELS_CONFIG
            )
        except ValueError as ve: # Catch specific errors from training like empty data
            if os.path.exists(filepath): os.remove(filepath)
            return jsonify({"error": f"Error during model training: {str(ve)}"}), 400
        except Exception as e: # Generic catch for other training errors
            if os.path.exists(filepath): os.remove(filepath)
            app.logger.error(f"Unhandled error during model training: {e}", exc_info=True)
            return jsonify({"error": f"An unexpected error occurred during model training. Check server logs."}), 500


        current_run_data['trained_models_dict'] = trained_mdls
        # Ensure X_test_data and y_test_encoded_data are serializable if they are DataFrames/Series
        # For run_nomad_simulation, they are used as is (Pandas or NumPy)
        current_run_data['X_test_data'] = X_tst
        current_run_data['y_test_encoded_data'] = y_tst

        # Calculate initial class priors from the training set (y_trn)
        initial_priors_dict_str_keys = {} # string_name -> probability
        if y_trn is not None and len(y_trn) > 0:
            unique_classes_in_train, class_counts_in_train = np.unique(y_trn, return_counts=True)
            # Map these numeric counts back to string class names using cls_map_num_to_str
            # Ensure correct mapping even if some classes from full dataset are not in y_trn (though stratify should help)
            total_train_samples = len(y_trn)
            temp_priors_numeric = {cls_num: count/total_train_samples for cls_num, count in zip(unique_classes_in_train, class_counts_in_train)}
            
            for numeric_label_global in unique_lab_enc: # Iterate over all known classes
                class_name_str = cls_map_num_to_str.get(int(numeric_label_global))
                if class_name_str:
                    initial_priors_dict_str_keys[class_name_str] = float(temp_priors_numeric.get(numeric_label_global, 0.0))
        else: # Fallback if y_trn is empty (e.g., very small dataset)
            num_total_classes = len(unique_cls_names_ord_str)
            uniform_prior = 1.0 / num_total_classes if num_total_classes > 0 else 0.0
            for class_name_str in unique_cls_names_ord_str:
                initial_priors_dict_str_keys[class_name_str] = uniform_prior

        current_run_data['initial_class_priors_str_keys'] = initial_priors_dict_str_keys

        return jsonify({
            "message": "File uploaded and base models trained.", "filename": os.path.basename(filepath), # Just basename for display
            "num_samples": int(len(data_df)) if data_df is not None else 0,
            "num_features": int(X_df.shape[1]) if X_df is not None else 0,
            "classes_str": unique_cls_names_ord_str, # List of string names
            "model_summaries": mdl_summaries,
            "initial_priors_str_keys": initial_priors_dict_str_keys,
            "candidate_model_names": [name for name,_,_ in CANDIDATE_MODELS_CONFIG]
        }), 200
    else:
        return jsonify({"error": "Invalid file type, please upload a CSV."}), 400


@app.route('/api/nomad_event_stream', methods=['GET'])
def nomad_event_stream_endpoint():
    required_setup_keys = ['trained_models_dict', 'X_test_data', 'y_test_encoded_data',
                           'initial_class_priors_str_keys', 'unique_class_names_ordered_str',
                           'class_map_numeric_to_str', 'X_column_names']
    for key in required_setup_keys:
        if key not in current_run_data or current_run_data[key] is None:
            # This function needs to yield, so wrap the error in a generator
            def error_stream_nodata_gen(missing_key):
                yield f"data: {json.dumps({'type': 'error', 'message': f'Prerequisite data missing: {missing_key}. Please upload file first.'})}\n\n"
            return Response(error_stream_nodata_gen(key), mimetype='text/event-stream')

    role_model_name = request.args.get('role_model_name')
    try:
        epsilon = float(request.args.get('epsilon', 0.05))
        batch_size = int(request.args.get('batch_size', 10))
        adaptive_update_window = int(request.args.get('adaptive_update_window', 100)) # New
        adaptive_beta = float(request.args.get('adaptive_beta', 0.3)) # New

        if batch_size <= 0: batch_size = 10
        if adaptive_update_window < 0: adaptive_update_window = 0 # 0 disables it
        if not (0.0 <= adaptive_beta <= 1.0): adaptive_beta = 0.3

    except ValueError:
        # Error in parsing parameters
        def error_stream_params_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Invalid NOMAD parameters provided.'})}\n\n"
        return Response(error_stream_params_gen(), mimetype='text/event-stream')

    quality_metric_for_ec = request.args.get('quality_metric_for_ec', 'f1-score')
    safety_check_type = request.args.get('safety_check_type', 'conservative')

    if not role_model_name or role_model_name not in current_run_data.get('trained_models_dict', {}):
        def error_stream_role_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': f'Role model {role_model_name} not valid or not trained.'})}\n\n"
        return Response(error_stream_role_gen(), mimetype='text/event-stream')

    # Ensure test data is available and not empty
    if current_run_data['X_test_data'] is None or current_run_data['X_test_data'].shape[0] == 0 or \
       current_run_data['y_test_encoded_data'] is None or len(current_run_data['y_test_encoded_data']) == 0:
        def error_stream_test_data_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Test data is missing or empty. Cannot run simulation.'})}\n\n"
        return Response(error_stream_test_data_gen(), mimetype='text/event-stream')


    def generate_nomad_updates():
        try:
            # Prepare data for the streaming function
            # Ensure X_test_data is in the format expected by the engine (e.g. numpy array or DataFrame)
            # The current engine expects DataFrame for predict calls if preprocessor is involved.
            # train_evaluate_individual_models_for_flask already returns X_test as DataFrame/ndarray.

            for update_package in run_nomad_simulation_for_flask_streamed(
                current_run_data['trained_models_dict'], role_model_name, epsilon,
                current_run_data['initial_class_priors_str_keys'],
                current_run_data['unique_class_names_ordered_str'],
                current_run_data['class_map_numeric_to_str'], quality_metric_for_ec,
                safety_check_type, current_run_data['X_test_data'],
                current_run_data['y_test_encoded_data'], CANDIDATE_MODELS_CONFIG,
                current_run_data['X_column_names'],
                batch_size=batch_size,
                adaptive_update_window=adaptive_update_window, # Pass new param
                adaptive_beta=adaptive_beta                   # Pass new param
            ):
                yield f"data: {json.dumps(update_package)}\n\n"
        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            app.logger.error(f"STREAMING ERROR in generate_nomad_updates: {str(e)}\n{tb_str}")
            error_message = {"type": "error", "message": f"Error during NOMAD simulation: {str(e)}"}
            yield f"data: {json.dumps(error_message)}\n\n"

    return Response(generate_nomad_updates(), mimetype='text/event-stream')


if __name__ == '__main__':
    # For development, threaded=True can be helpful.
    # For production, use a proper WSGI server like Gunicorn or Waitress.
    app.run(debug=True, port=5001, threaded=True)