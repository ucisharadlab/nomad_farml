from flask import Flask, Response, request, jsonify, render_template
import matplotlib.pyplot as plt
import pandas as pd
import io
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

RESULT_FOLDER = os.path.join('static', 'cascade_result')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = RESULT_FOLDER

@app.route('/main')
def main():
    return render_template('cascade_main.html')

@app.route('/get_orig_data')
def get_orig_data():
    orig_data_file = os.path.join(app.config['UPLOAD_FOLDER'], 'demo_data_orig.csv') 
    data_orig = pd.read_csv(orig_data_file)
    df_orig = pd.DataFrame(data_orig)
    html_table_orig = df_orig.to_html(classes='table table-bordered')
    return render_template('cascade_view_data.html', tables=[html_table_orig], titles=['IoT Flow-based Data'])

@app.route('/get_proc_data')
def get_proc_data():
    proc_data_file = os.path.join(app.config['UPLOAD_FOLDER'], 'demo_data_proc.csv') 
    data_proc = pd.read_csv(proc_data_file)
    df_proc = pd.DataFrame(data_proc)
    html_table_proc = df_proc.to_html(classes='table table-bordered')
    return render_template('cascade_view_data.html', tables=[html_table_proc], titles=['IoT Preprocessed'])

@app.route('/get_all', methods=['POST', 'GET'])
def model_respond():
    selected_value = request.args.get('selected')

    if selected_value == 'model_option1':
        model_filename =  os.path.join(app.config['UPLOAD_FOLDER'],'decision_tree.png')
        plot_list = [model_filename]
        return render_template("cascade_view_img.html", plots = plot_list)
    elif selected_value == 'model_option2':
        submodel_filename =  os.path.join(app.config['UPLOAD_FOLDER'],'decision_tree_0.png')
        plot_list = [submodel_filename]
        return render_template("cascade_view_img.html", plots = plot_list)
    elif selected_value == 'model_option3':
        submodel_filename =  os.path.join(app.config['UPLOAD_FOLDER'],'decision_tree_2.png')
        plot_list = [submodel_filename]
        return render_template("cascade_view_img.html", plots = plot_list)
    elif selected_value == 'model_option4':
        submodel_filename =  os.path.join(app.config['UPLOAD_FOLDER'],'decision_tree_3.png')
        plot_list = [submodel_filename]
        return render_template("cascade_view_img.html", plots = plot_list)
    elif selected_value == 'model_option5':
        submodel_filename =  os.path.join(app.config['UPLOAD_FOLDER'],'decision_tree_4.png')
        plot_list = [submodel_filename]
        return render_template("cascade_view_img.html", plots = plot_list)
    elif selected_value == 'test_method_option1':
        accf1_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'acc_f1.png')
        nodes_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'nodes.png')
        plot_list = [accf1_filename, nodes_filename]
        return render_template("cascade_view_img.html", plots = plot_list)
    elif selected_value == 'test_method_option2':
        path_filename = os.path.join(app.config['UPLOAD_FOLDER'],'pred_path_anim.gif')
        plot_list = [path_filename]
        return render_template("cascade_view_img.html", plots = plot_list)
    elif selected_value == 'attack_option1':
        corr_filename = os.path.join(app.config['UPLOAD_FOLDER'],'feature_corr.png')
        plot_list = [corr_filename]
        return render_template("cascade_view_img.html", plots = plot_list)
    elif selected_value == 'attack_option2':
        atk_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'atk_acc_f1.png')
        plot_list = [atk_filename]
        return render_template("cascade_view_img.html", plots = plot_list)
    elif selected_value == 'adtrain_option1':
        adtrain_accf1_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'adtrain_acc_f1.png')
        adtrain_atk_accf1_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'adtrain_atk_acc_f1.png')
        plot_list = [adtrain_accf1_filename, adtrain_atk_accf1_filename]
        return render_template("cascade_view_img.html", plots = plot_list)
    elif selected_value == 'adtrain_option2':
        boundary_filename = os.path.join(app.config['UPLOAD_FOLDER'],'boundary_anim.gif')
        plot_list = [boundary_filename]
        return render_template("cascade_view_img.html", plots = plot_list)
    else:
        return render_template("cascade_warning.html")


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)



