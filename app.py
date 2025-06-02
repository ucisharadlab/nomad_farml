from flask import Flask, request, jsonify, render_template, Response
import os
import uuid
import pandas as pd
import numpy as np
import json
import traceback # For detailed error logging

from nomad_arima_backend import (
    CANDIDATE_MODELS_CONFIG,
    load_and_preprocess_data_for_flask,
    train_evaluate_individual_models_for_flask,
    run_nomad_simulation_for_flask_streamed
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
current_run_data = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/candidate_models', methods=['GET'])
def get_candidate_models_endpoint():
    return jsonify([{"name": name, "cost": float(cost)} for name, _, cost in CANDIDATE_MODELS_CONFIG])

@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.csv'):
        safe_original_filename = "".join(c if c.isalnum() or c in ['.', '_'] else '_' for c in file.filename)
        filename = str(uuid.uuid4()) + "_" + safe_original_filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try: file.save(filepath)
        except Exception as e: app.logger.error(f"Failed to save file: {e}"); return jsonify({"error": "Failed to save file."}), 500
        
        print(f"DEBUG upload_csv: File saved to {filepath}")
        current_run_data.clear()

        data_df, X_df, y_enc_arr, _, unique_lab_enc, \
        cls_map_num_to_str, unique_cls_names_ord_str, preproc, x_col_names, \
        X_by_cls_str, y_by_cls_str, err_msg = load_and_preprocess_data_for_flask(filepath)

        if err_msg:
            if os.path.exists(filepath): os.remove(filepath)
            print(f"ERROR upload_csv: load_and_preprocess_data_for_flask failed: {err_msg}")
            return jsonify({"error": err_msg}), 400
        
        if X_df is None or y_enc_arr is None or len(y_enc_arr) == 0:
            if os.path.exists(filepath): os.remove(filepath)
            print("ERROR upload_csv: No valid data (X_df or y_enc_arr is None or y_enc_arr empty) after preprocessing.")
            return jsonify({"error": "No valid data (features or target) found after preprocessing."}), 400
        
        print(f"DEBUG upload_csv: Data loaded. X_df shape: {X_df.shape}, y_enc_arr length: {len(y_enc_arr)}")
        print(f"DEBUG upload_csv: X_by_class_str_keys sample: { {k: v.shape for k,v in X_by_cls_str.items()} if X_by_cls_str else 'None' }")


        current_run_data['X_column_names'] = x_col_names
        current_run_data['X_full'] = X_df
        current_run_data['y_encoded_full'] = y_enc_arr
        current_run_data['class_map_numeric_to_str'] = cls_map_num_to_str
        current_run_data['unique_class_names_ordered_str'] = unique_cls_names_ord_str
        current_run_data['unique_labels_encoded'] = unique_lab_enc.tolist()
        current_run_data['preprocessor_for_models'] = preproc
        current_run_data['X_by_class_str_keys'] = X_by_cls_str
        current_run_data['y_by_class_str_keys'] = y_by_cls_str

        try:
            trained_mdls, mdl_summaries, X_tst, y_tst, y_trn = train_evaluate_individual_models_for_flask(
                current_run_data['X_full'], current_run_data['y_encoded_full'],
                current_run_data['preprocessor_for_models'],
                current_run_data['class_map_numeric_to_str'],
                current_run_data['unique_labels_encoded'],
                CANDIDATE_MODELS_CONFIG
            )
        except ValueError as ve: # Catch specific errors like empty data for training
            if os.path.exists(filepath): os.remove(filepath)
            print(f"ERROR upload_csv: ValueError during model training: {str(ve)}")
            return jsonify({"error": f"Error during model training: {str(ve)}"}), 400
        except Exception as e_train_eval: # Generic catch for other training errors
            if os.path.exists(filepath): os.remove(filepath)
            app.logger.error(f"Unhandled error during model training: {e_train_eval}", exc_info=True)
            print(f"ERROR upload_csv: Exception during model training: {e_train_eval}\n{traceback.format_exc()}")
            return jsonify({"error": "An unexpected error occurred during model training."}), 500

        # Logging X_tst and y_tst shapes
        print(f"DEBUG upload_csv: X_test_data_overall (X_tst) shape: {X_tst.shape if X_tst is not None else 'None'}")
        print(f"DEBUG upload_csv: y_test_encoded_data_overall (y_tst) length: {len(y_tst) if y_tst is not None else 'None'}")
        if X_tst is not None and not X_tst.empty: # Check if DataFrame and not empty
            print(f"DEBUG upload_csv: X_tst head:\n{X_tst.head() if isinstance(X_tst, pd.DataFrame) else X_tst[:5]}")
        if y_tst is not None and len(y_tst) > 0:
            print(f"DEBUG upload_csv: y_tst sample: {y_tst[:5]}")

        # Ensure X_tst and y_tst are not None before assigning, though train_evaluate should return valid structures
        if X_tst is None or y_tst is None:
            print("ERROR upload_csv: X_tst or y_tst is None after model training/evaluation. This is unexpected.")
            # Handle this error case, e.g., by returning an error to the user
            if os.path.exists(filepath): os.remove(filepath)
            return jsonify({"error": "Failed to prepare test data split after model training."}), 500


        current_run_data['trained_models_dict'] = trained_mdls
        current_run_data['X_test_data_overall'] = X_tst
        current_run_data['y_test_encoded_data_overall'] = y_tst

        initial_priors_dict_str_keys = {}
        if y_trn is not None and len(y_trn) > 0:
            unique_tr, counts_tr = np.unique(y_trn, return_counts=True)
            temp_priors_num = {cls_num: count/len(y_trn) for cls_num, count in zip(unique_tr, counts_tr)}
            for num_lab_global in unique_lab_enc:
                cls_name_str = cls_map_num_to_str.get(int(num_lab_global))
                if cls_name_str: initial_priors_dict_str_keys[cls_name_str] = float(temp_priors_num.get(num_lab_global, 0.0))
        else: # Fallback if y_trn is empty
            num_total_classes = len(unique_cls_names_ord_str)
            uniform_p = 1.0 / max(1, num_total_classes) # Avoid division by zero
            for class_name_str in unique_cls_names_ord_str:
                initial_priors_dict_str_keys[class_name_str] = uniform_p
        current_run_data['initial_class_priors_str_keys'] = initial_priors_dict_str_keys
        print(f"DEBUG upload_csv: Initial priors calculated: {initial_priors_dict_str_keys}")


        return jsonify({
            "message": "File uploaded and base models trained.", "filename": os.path.basename(filepath),
            "num_samples": int(len(data_df)), "num_features": int(X_df.shape[1]),
            "classes_str": unique_cls_names_ord_str, "model_summaries": mdl_summaries,
            "initial_priors_str_keys": initial_priors_dict_str_keys,
            "candidate_model_names": [name for name,_,_ in CANDIDATE_MODELS_CONFIG]
        }), 200
    return jsonify({"error": "Invalid file type, please upload a CSV."}), 400

@app.route('/api/nomad_event_stream', methods=['GET'])
def nomad_event_stream_endpoint():
    print("DEBUG nomad_event_stream: Endpoint hit.")
    required_keys = ['trained_models_dict', 'X_test_data_overall', 'y_test_encoded_data_overall',
                     'initial_class_priors_str_keys', 'unique_class_names_ordered_str',
                     'class_map_numeric_to_str', 'X_column_names',
                     'X_by_class_str_keys', 'y_by_class_str_keys']
    for key in required_keys:
        if key not in current_run_data or current_run_data[key] is None:
            print(f"ERROR nomad_event_stream: Prerequisite data missing: {key}")
            def err_gen(k): yield f"data: {json.dumps({'type':'error','message':f'Prereq data missing: {k}. Upload file.'})}\n\n"
            return Response(err_gen(key), mimetype='text/event-stream')

    role_model_name = request.args.get('role_model_name')
    workload_phases_json = request.args.get('workload_phases', '[]')
    workload_phases = []
    try:
        parsed_phases = json.loads(workload_phases_json)
        if isinstance(parsed_phases, list): workload_phases = parsed_phases
        else: raise ValueError("Workload phases must be a list.")
        print(f"DEBUG nomad_event_stream: Parsed workload_phases: {workload_phases}")
    except json.JSONDecodeError:
        def err_gen_phase_json(): yield f"data: {json.dumps({'type':'error','message':'Invalid JSON for workload phases.'})}\n\n"
        return Response(err_gen_phase_json(), mimetype='text/event-stream')
    except ValueError as ve:
        def err_gen_phase_val(msg): yield f"data: {json.dumps({'type':'error','message':msg})}\n\n"
        return Response(err_gen_phase_val(str(ve)), mimetype='text/event-stream')

    try:
        epsilon = float(request.args.get('epsilon', 0.05))
        batch_size = int(request.args.get('batch_size', 10))
        adaptive_window = int(request.args.get('adaptive_update_window', 100))
        adaptive_beta_val = float(request.args.get('adaptive_beta', 0.3))
        if not (0 <= batch_size): batch_size = 10
        if not (0 <= adaptive_window): adaptive_window = 0
        if not (0.0 <= adaptive_beta_val <= 1.0): adaptive_beta_val = 0.3
    except ValueError:
        def err_gen_params(): yield f"data: {json.dumps({'type':'error','message':'Invalid NOMAD params.'})}\n\n"
        return Response(err_gen_params(), mimetype='text/event-stream')

    quality_metric_ec = request.args.get('quality_metric_for_ec', 'f1-score')
    safety_check = request.args.get('safety_check_type', 'conservative')

    if not role_model_name or role_model_name not in current_run_data.get('trained_models_dict', {}):
        print(f"ERROR nomad_event_stream: Role model '{role_model_name}' not valid or not trained.")
        def err_gen_role(): yield f"data: {json.dumps({'type':'error','message':f'Role model {role_model_name} invalid.'})}\n\n"
        return Response(err_gen_role(), mimetype='text/event-stream')

    # Ensure X_test_data_overall is not None if workload_phases is empty
    if not workload_phases and (current_run_data['X_test_data_overall'] is None or \
                               (isinstance(current_run_data['X_test_data_overall'], pd.DataFrame) and current_run_data['X_test_data_overall'].empty) or \
                               (not isinstance(current_run_data['X_test_data_overall'], pd.DataFrame) and len(current_run_data['X_test_data_overall']) == 0) ):
        print("ERROR nomad_event_stream: X_test_data_overall is missing or empty for sequential mode.")
        def err_gen_test_data(): yield f"data: {json.dumps({'type':'error','message':'Test data missing/empty for sequential mode.'})}\n\n"
        return Response(err_gen_test_data(), mimetype='text/event-stream')


    def generate_updates():
        try:
            print("DEBUG nomad_event_stream: Starting generate_updates stream.")
            for update in run_nomad_simulation_for_flask_streamed(
                current_run_data['trained_models_dict'], role_model_name, epsilon,
                current_run_data['initial_class_priors_str_keys'],
                current_run_data['unique_class_names_ordered_str'],
                current_run_data['class_map_numeric_to_str'], quality_metric_ec,
                safety_check, current_run_data['X_test_data_overall'],
                current_run_data['y_test_encoded_data_overall'],
                CANDIDATE_MODELS_CONFIG, current_run_data['X_column_names'],
                batch_size=batch_size, adaptive_update_window=adaptive_window,
                adaptive_beta=adaptive_beta_val,
                workload_phases=workload_phases,
                X_by_class_str_keys=current_run_data['X_by_class_str_keys'],
                y_by_class_str_keys=current_run_data['y_by_class_str_keys']
            ):
                yield f"data: {json.dumps(update)}\n\n"
            print("DEBUG nomad_event_stream: Finished generate_updates stream successfully.")
        except Exception as e_stream:
            tb_str = traceback.format_exc()
            app.logger.error(f"STREAMING ERROR in generate_nomad_updates: {str(e_stream)}\n{tb_str}")
            print(f"ERROR nomad_event_stream generate_updates: {str(e_stream)}\n{tb_str}")
            yield f"data: {json.dumps({'type':'error','message':f'Sim error: {str(e_stream)}'})}\n\n"
    return Response(generate_updates(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, port=5001, threaded=True)