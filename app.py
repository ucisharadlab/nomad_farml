from flask import Flask, request, jsonify, render_template, Response
import os
import uuid
import pandas as pd
import numpy as np
import json

from nomad_backend import (
    CANDIDATE_MODELS_CONFIG,
    load_and_preprocess_data_for_flask,
    train_evaluate_individual_models_for_flask,
    run_nomad_simulation_for_flask_streamed
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

current_run_data = {} # Simple global storage; not for production with multiple users

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
        filename = str(uuid.uuid4()) + "_" + file.filename.replace(" ", "_")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
        except Exception as e:
            return jsonify({"error": f"Failed to save file: {str(e)}"}), 500
        
        current_run_data.clear()

        data_df, X_df, y_enc, _, unique_lab_enc, \
        cls_map_num_to_str, unique_cls_names_ord_str, preproc, x_col_names, err_msg = load_and_preprocess_data_for_flask(filepath)

        if err_msg:
            if os.path.exists(filepath): os.remove(filepath)
            return jsonify({"error": err_msg}), 400
        
        current_run_data['X_column_names'] = x_col_names # Store original column names of X
        current_run_data['X_full'] = X_df
        current_run_data['y_encoded_full'] = y_enc # Full y_encoded
        current_run_data['class_map_numeric_to_str'] = cls_map_num_to_str # numeric_label -> string_name
        current_run_data['unique_class_names_ordered_str'] = unique_cls_names_ord_str # list of string_name
        current_run_data['unique_labels_encoded'] = unique_lab_enc # list of numeric_label
        current_run_data['preprocessor_for_models'] = preproc

        # Train base models and get test split
        trained_mdls, mdl_summaries, X_tst, y_tst, y_trn = train_evaluate_individual_models_for_flask(
            current_run_data['X_full'], current_run_data['y_encoded_full'],
            current_run_data['preprocessor_for_models'], 
            current_run_data['class_map_numeric_to_str'], # Pass numeric_to_string map
            current_run_data['unique_labels_encoded'], # Pass numeric labels
            CANDIDATE_MODELS_CONFIG
        )
        current_run_data['trained_models_dict'] = trained_mdls
        current_run_data['X_test_data'] = X_tst # This is now a DataFrame or ndarray from split
        current_run_data['y_test_encoded_data'] = y_tst # Numeric labels for test set

        initial_priors_dict = {} # string_name -> prob
        if y_trn is not None and len(y_trn) > 0:
            _, cls_counts = np.unique(y_trn, return_counts=True) # cls_counts corresponds to sorted unique numeric labels in y_trn
            sorted_unique_numeric_labels_in_y_trn = np.sort(np.unique(y_trn))

            temp_priors_list = cls_counts / len(y_trn)
            for i, numeric_label in enumerate(sorted_unique_numeric_labels_in_y_trn):
                class_name_str = cls_map_num_to_str.get(int(numeric_label))
                if class_name_str:
                    initial_priors_dict[class_name_str] = float(temp_priors_list[i])
        current_run_data['initial_class_priors_str_keys'] = initial_priors_dict


        return jsonify({
            "message": "File uploaded and base models trained.", "filename": filename,
            "num_samples": int(len(data_df)) if data_df is not None else 0,
            "num_features": int(X_df.shape[1]) if X_df is not None else 0,
            "classes_str": unique_cls_names_ord_str, # List of string names
            "model_summaries": mdl_summaries, # Contains floats and strings
            "initial_priors_str_keys": initial_priors_dict, # string_name -> prob
            "candidate_model_names": [name for name,_,_ in CANDIDATE_MODELS_CONFIG] # For chart labels
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
            def error_stream_nodata():
                yield f"data: {json.dumps({'type': 'error', 'message': f'Prerequisite data missing: {key}. Please upload file first.'})}\n\n"
            return Response(error_stream_nodata(), mimetype='text/event-stream')

    role_model_name = request.args.get('role_model_name')
    try:
        epsilon = float(request.args.get('epsilon', 0.05))
        batch_size = int(request.args.get('batch_size', 10)) # Get batch_size from query params, default 10
        if batch_size <= 0: batch_size = 10 # Ensure positive batch size
    except ValueError: 
        epsilon = 0.05
        batch_size = 10
    quality_metric_for_ec = request.args.get('quality_metric_for_ec', 'f1-score')
    safety_check_type = request.args.get('safety_check_type', 'conservative')

    if not role_model_name or role_model_name not in current_run_data.get('trained_models_dict', {}):
        def error_stream_role():
            yield f"data: {json.dumps({'type': 'error', 'message': f'Role model {role_model_name} not valid.'})}\n\n"
        return Response(error_stream_role(), mimetype='text/event-stream')

    def generate_nomad_updates():
        try:
            for update_package in run_nomad_simulation_for_flask_streamed(
                current_run_data['trained_models_dict'], role_model_name, epsilon,
                current_run_data['initial_class_priors_str_keys'],
                current_run_data['unique_class_names_ordered_str'],
                current_run_data['class_map_numeric_to_str'], quality_metric_for_ec,
                safety_check_type, current_run_data['X_test_data'],
                current_run_data['y_test_encoded_data'], CANDIDATE_MODELS_CONFIG,
                current_run_data['X_column_names'],
                batch_size=batch_size # Pass batch_size
            ):
                yield f"data: {json.dumps(update_package)}\n\n"
        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            print(f"STREAMING ERROR: {str(e)}\n{tb_str}")
            error_message = {"type": "error", "message": f"Streaming error: {str(e)}"}
            yield f"data: {json.dumps(error_message)}\n\n"

    return Response(generate_nomad_updates(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, port=5001, threaded=True) # threaded=True might help with multiple connections if EventSource is not closed properly by client