# NOMAD: Adaptive Model Chaining Visualizer (React Edition)

## Introduction

The NOMAD Visualizer is an interactive web application designed to demonstrate and analyze a powerful strategy for optimizing machine learning model deployments. It provides a "what-if" engine to explore the trade-offs between a model's computational **cost** and its predictive **accuracy**.

At its core, this tool simulates a smart system that dynamically selects the most efficient ML model from a pool of candidates for each individual task, aiming to achieve the highest possible accuracy for the lowest possible cost.

**This visualizer helps answer the fundamental question: How can we achieve the performance of our most powerful models while only paying the cost of our cheapest ones?**

---

## Features

* **Dynamic Model Management:** Add/remove `scikit-learn` models, define costs, configure hyperparameters, and upload custom model classes from `.py` files.
* **Data-Driven Training & Evaluation:** Upload a CSV dataset to train and evaluate all candidate models and view their baseline performance.
* **Interactive NOMAD Configuration:** Select a "Role Model" as a performance benchmark and set an "Epsilon" (ε) tolerance for cost-saving.
* **Advanced Workload Simulation:** Design complex, multi-stage workloads with varying data distributions to stress-test the system's adaptability.
* **Live Results Visualization:** Watch the simulation unfold with real-time charts for model execution frequency, cumulative cost, and accuracy.
* **Comprehensive Final Summary:** Get a detailed breakdown of the final results comparing the NOMAD strategy to the baseline Role Model.

---

## Tech Stack

* **Backend:** Python, Flask, Flask-CORS
* **Frontend:** React, Vite, Tailwind CSS v4
* **Machine Learning:** Scikit-learn, Pandas, NumPy
* **Visualization:** Chart.js

---

## Getting Started

This project has a separate backend and frontend. You will need to run both simultaneously.

### Prerequisites

* Python 3.7+ and `pip`
* Node.js v16+ and `npm`

### 1. Backend Setup

1.  **Navigate to your project folder.**

2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Python dependencies:**
    *(Create a `requirements.txt` file with `Flask`, `Flask-Cors`, `pandas`, `numpy`, `scikit-learn`)*
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Flask backend server:**
    ```bash
    python app.py
    ```
    The backend will be running at `http://127.0.0.1:5001`.

### 2. Frontend Setup (with Vite and Tailwind CSS v4)

This guide assumes you are starting a new React project with Vite, the recommended modern build tool.

1.  **Open a new terminal window** and create a new Vite + React project:
    ```bash
    npm create vite@latest nomad-frontend -- --template react
    cd nomad-frontend
    ```

2.  **Install all frontend dependencies in one command:**
    ```bash
    npm install && npm install -D tailwindcss @tailwindcss/vite react-chartjs-2 chart.js @heroicons/react
    ```

3.  **Create `tailwind.config.js`:** This file is still needed to tell Tailwind where to look for your classes. In the `nomad-frontend` root, create the file with this content:
    ```javascript
    /** @type {import('tailwindcss').Config} */
    export default {
      content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
      ],
      theme: {
        extend: {},
      },
      plugins: [],
    }
    ```

4.  **Configure the Vite Plugin:** Add the `@tailwindcss/vite` plugin to your `vite.config.js` file.
    ```javascript
    import { defineConfig } from 'vite'
    import react from '@vitejs/plugin-react'
    import tailwindcss from '@tailwindcss/vite'

    // [https://vitejs.dev/config/](https://vitejs.dev/config/)
    export default defineConfig({
      plugins: [react(), tailwindcss()],
    })
    ```

5.  **Import Tailwind's CSS:** Open your `./src/index.css` file, remove all existing content, and add the following line. Note that the old `@tailwind` directives are no longer used.
    ```css
    @import 'tailwindcss';
    ```

6.  **Replace the content of `src/App.jsx`** with the React component code provided for this project.

7.  **Run the React development server:**
    ```bash
    npm run dev
    ```
    The frontend will open in your browser (usually at `http://localhost:5173`) and will automatically connect to the backend API running on port 5001.

---

## Walkthrough

### Step 1: Manage Candidate Models

This is your "garage" of available models.

* **View Models:** The table shows the default models, their type, parameters, and cost.
* **Add a Model:** Use the "Add New Model" form.
    1.  Give it a **Unique Name**.
    2.  Set its **Cost**. This is an abstract value representing its computational expense.
    3.  Select a **Model Type** from the dropdown. Hyperparameter fields will appear dynamically.
    4.  Fill in any desired parameters.
    5.  Click **Add Model**.
* **Add a Custom Model:**
    1.  Select `-- Custom Model (from file) --` as the type.
    2.  Enter the exact **Python Class Name** from your file.
    3.  Choose the `.py` file containing your model. The model must have a `scikit-learn`-compatible API (i.e., it must have `.fit()`, `.predict()`, and preferably `.predict_proba()` methods).
* **Remove a Model:** Click the "Remove" button in the table.

### Step 2: Upload Data & Train

1.  Click **"Choose File"** and select a CSV file. The target variable (the class label) should be the second-to-last column. The smallest dataset(smaller models) is available as part of this repository - UNSW_NB15_training-set.csv.
2.  Click **"Upload and Train"**.
3.  The system will train all models in your list and display the "Individual Model Performance" table, showing the baseline results.

### Step 3: Configure & Run the Simulation

Once the models are trained, this section appears. The values are pre-selected to demonstrate NOMAD effectively on a laptop with basic compute power.

1.  **Role Model:** Select your best-performing (and likely most expensive) model from the dropdown. This is the benchmark NOMAD will be measured against.
2.  **Epsilon (ε):** Set your tolerance. A value of `0.2` means you're willing to accept an accuracy that is at most 20% worse than the Role Model's *if* it results in a cost saving.
3.  **Workload Phases (Optional):** To simulate specific scenarios, define one or more phases. For each, set the **Duration** (number of events) and the **Target Class Distribution** (the percentage of each class). If you leave this empty, the simulation will run on the test data split.
4.  Click **"Run NOMAD Simulation"**.

### Step 4: Interpret the Results

The live charts will update as the simulation runs. The key insights are:

* Is the **Cumulative Cost** of NOMAD significantly lower than the Role Model?
* Is the **Accuracy Trend** of NOMAD staying close to the Role Model's static accuracy line?
* How does the **Model Execution Frequency** change when the workload shifts between phases?

When the simulation completes, the **Final Summary** provides a quantitative comparison of the two strategies.

---

## File Structure

```
    ├── backend/
    │   ├── app.py                  # Main Flask application, API endpoints, and state management.
    │   ├── nomad_arima_backend.py  # Core NOMAD logic, model classes, and simulation engine.
    │   └── requirements.txt        # List of Python dependencies.
    └── nomad-frontend/
        ├── index.html              # Main HTML file for the Vite.js frontend.
        ├── vite.config.js          # Vite configuration file.
        ├── tailwind.config.js      # Tailwind CSS configuration file.
        ├── package.json            # Node.js dependencies and scripts.
        └── src/
            ├── App.jsx             # Main React component.
            ├── index.css           # Tailwind CSS imports.
            └── components/         # React components directory.
    ```