NOMAD: Adaptive Model Chaining Visualizer
Introduction
The NOMAD (Non-Monotonic Adaptive Deployment) Visualizer is an interactive web application designed to demonstrate and analyze a powerful strategy for optimizing machine learning model deployments. It provides a "what-if" engine to explore the trade-offs between a model's computational cost and its predictive accuracy.

At its core, this tool simulates a smart system that dynamically selects the most efficient ML model from a pool of candidates for each individual task, aiming to achieve the highest possible accuracy for the lowest possible cost.

This visualizer helps answer the fundamental question: How can we achieve the performance of our most powerful models while only paying the cost of our cheapest ones?

The Problem It Solves
In modern machine learning, there is an inherent conflict:

High-Performance Models (e.g., large ensembles, deep neural networks) are highly accurate but are computationally expensive, leading to high operational costs, energy consumption, and slower response times.

Low-Cost Models (e.g., shallow decision trees, linear models) are fast and cheap to run but often lack the accuracy required for critical tasks.

A one-size-fits-all approach, where a single model is used for every task, is inefficient. It's like using a sledgehammer to crack a nut. The NOMAD strategy proposes a more intelligent alternative: an adaptive chain of models that can dynamically route tasks, ensuring that easy predictions are handled by cheap models and only the most difficult predictions are escalated to expensive ones.

Features
Dynamic Model Management:

Add and remove scikit-learn based models directly from the UI.

Define custom computational costs for each model.

Configure model hyperparameters.

Upload custom model classes from external .py files.

Data-Driven Training & Evaluation:

Upload your own CSV dataset to train and evaluate all candidate models.

View baseline performance metrics (Accuracy, F1-Score, Cost) for each individual model.

Interactive NOMAD Configuration:

Select a "Role Model" to act as the gold standard for performance.

Set an "Epsilon" (ε) tolerance to define the acceptable trade-off between accuracy and cost-savings.

Choose between different chain safety algorithms to manage risk.

Advanced Workload Simulation:

Simulate a simple workload based on the test set split of your uploaded data.

Design complex, multi-stage workloads with varying class distributions to stress-test the system's adaptability.

Live Results Visualization:

Watch the simulation unfold with real-time charts powered by Chart.js.

Model Execution Frequency: See which models NOMAD chooses to use.

Cumulative Cost: Compare the operational cost of NOMAD against the Role Model.

Accuracy Trend: Track NOMAD's live accuracy to ensure performance goals are met.

Class Prior Distribution: Observe how the system adapts its internal beliefs about the incoming data.

Comprehensive Final Summary:

Get a detailed breakdown of the final accuracy, average cost, and confusion matrices for both the NOMAD strategy and the baseline Role Model.

Tech Stack
Backend: Python, Flask

Frontend: HTML5, CSS3, JavaScript (ES6+)

Machine Learning: Scikit-learn, Pandas, NumPy

Visualization: Chart.js

Getting Started
Follow these instructions to set up and run the NOMAD Visualizer on your local machine.

Prerequisites
Python 3.7+

pip (Python package installer)

Installation
Clone the repository:

git clone <your-repository-url>
cd <repository-directory>

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required Python packages:
The application relies on several libraries. Install them using the provided requirements.txt file.

pip install -r requirements.txt

(If a requirements.txt is not available, install them manually: pip install Flask pandas numpy scikit-learn)

Running the Application
Start the Flask server:

python app.py

Access the visualizer:
Open your web browser and navigate to:
http://127.0.0.1:5001

Walkthrough
Step 1: Manage Candidate Models
This is your "garage" of available models.

View Models: The table shows the default models, their type, parameters, and cost.

Add a Model: Use the "Add New Model" form.

Give it a Unique Name.

Set its Cost. This is an abstract value representing its computational expense.

Select a Model Type from the dropdown. Hyperparameter fields will appear dynamically.

Fill in any desired parameters.

Click Add Model.

Add a Custom Model:

Select -- Custom Model (from file) -- as the type.

Enter the exact Python Class Name from your file.

Choose the .py file containing your model. The model must have a scikit-learn-compatible API (i.e., it must have .fit(), .predict(), and preferably .predict_proba() methods).

Remove a Model: Click the "Remove" button in the table.

Step 2: Upload Data & Train
Click "Choose File" and select a CSV file. The target variable (the class label) should be the second-to-last column.

Click "Upload and Train".

The system will train all models in your list and display the "Individual Model Performance" table, showing the baseline results.

Step 3: Configure & Run the Simulation
Once the models are trained, this section appears.

Role Model: Select your best-performing (and likely most expensive) model from the dropdown. This is the benchmark NOMAD will be measured against.

Epsilon (ε): Set your tolerance. A value of 0.05 means you're willing to accept an accuracy that is at most 5% worse than the Role Model's if it results in a cost saving.

Workload Phases (Optional): To simulate specific scenarios, define one or more phases. For each, set the Duration (number of events) and the Target Class Distribution (the percentage of each class). If you leave this empty, the simulation will run on the test data split.

Click "Run NOMAD Simulation".

Step 4: Interpret the Results
The live charts will update as the simulation runs. The key insights are:

Is the Cumulative Cost of NOMAD significantly lower than the Role Model?

Is the Accuracy Trend of NOMAD staying close to the Role Model's static accuracy line?

How does the Model Execution Frequency change when the workload shifts between phases?

When the simulation completes, the Final Summary provides a quantitative comparison of the two strategies.

File Structure
.
├── app.py                  # Main Flask application, API endpoints, and state management.
├── nomad_arima_backend.py  # Core NOMAD logic, model classes, and simulation engine.
├── requirements.txt        # List of Python dependencies.
├── static/
│   ├── script.js           # Frontend JavaScript for interactivity, API calls, and charting.
│   └── style.css           # CSS for styling the web interface.
└── templates/
    └── index.html          # The main HTML file for the user interface.
