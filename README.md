# NOMAD: Dynamic Model Chaining for Efficient Classification

This project implements the NOMAD algorithm, a strategy for dynamic, cost-efficient classification of event streams. It selects from a set of pre-trained machine learning models with varying computational costs and quality characteristics, creating a chain of execution for each incoming event. The goal is to match the classification quality of a high-quality (but potentially expensive) "role model" within a defined tolerance (`epsilon`), while significantly minimizing the average computational cost per event.

This implementation is based on the concepts described in the provided paper sections, focusing on dynamic model selection using utility scores, chain safety checks, and probability updates.

**Current Date Context:** Generated on Tuesday, April 29, 2025 (IST).

## Core Concepts

* **Model Set (`mathcal{M}`):** A collection of classification models (e.g., Decision Tree, Random Forest, SVC) trained for the same task, each with an associated prediction `cost` (e.g., average inference time) and quality metrics (e.g., recall, precision per class).
* **Role Model (`M_r`):** A designated model within the set, typically the one with the highest overall quality (e.g., highest F1-score or recall), often also having a high cost. The performance of the NOMAD strategy is benchmarked against this model.
* **Epsilon (`epsilon`):** A tolerance parameter ($0 \le \epsilon < 1$) defining the maximum acceptable *relative* drop in quality for the overall strategy compared to the role model. For example, `epsilon = 0.05` means the strategy's recall (or other chosen metric) for each class must be at least 95% of the role model's recall for that same class.
* **Exit Class (`EC(M_i)`):** For a given model `M_i`, its Exit Classes are the set of target classes `C_j` for which `M_i`'s quality is considered "good enough" compared to the role model, specifically: `Quality(M_i, C_j) >= (1 - epsilon) * Quality(M_r, C_j)`. If `M_i` predicts an event belongs to one of its exit classes, the chain terminates, and that prediction is used.
* **Model Chain (`S_e`):** The ordered sequence of models executed for a specific event `e`. The chain stops when a model predicts one of its own exit classes.
* **Chain Safety:** A mechanism to ensure that executing a sequence of models doesn't degrade the classification quality below the `(1 - epsilon)` threshold compared to the role model, especially for classes that might pass through multiple models before reaching their designated exit model. This implementation uses the "Conservative Chain Safety Estimation" based on recall products.
* **Utility Score (`U(M_i)`):** A heuristic used to select the next model to run in the chain. It typically balances the probability of the current event belonging to one of the model's exit classes against the model's computational cost: `Utility = Sum(P(Cj) for Cj in EC(Mi)) / Cost(Mi)`.
* **Probability Updates:** After a model runs but *doesn't* exit, the system updates its internal belief (probability distribution) about the event's true class based on the model's output (e.g., softmax scores), influencing the utility calculation for subsequent model selection.

## Project Structure
nomad_project/
├── nomad/                  # Main Python package
│   ├── init.py         # Makes 'nomad' a package
│   ├── config.py           # Configuration variables (paths, epsilon, etc.)
│   ├── data_utils.py       # Data loading, preprocessing, splitting
│   ├── model_training.py   # Functions to train or load models
│   ├── evaluation.py       # ModelWrapper class, metric calculations, cost timing
│   └── nomad_strategy.py   # Core NOMAD logic (EC, safety, utility, select_and_classify)
├── data/                   # Default directory for datasets
│   └── your_dataset.csv    # Placeholder for your data file
├── models/                 # Default directory to save/load trained models (optional)
├── results/                # Default directory to save simulation results (optional)
├── main.py                 # Main executable script to run simulation & evaluation
└── requirements.txt        # Project dependencies

* **`nomad/config.py`**: Central place to set file paths, epsilon, role model name, etc.
* **`nomad/data_utils.py`**: Handles loading your specific dataset (or generates dummy data), preprocessing, and splitting into train/validation/test sets. Returns integer labels for training/evaluation and string class names.
* **`nomad/model_training.py`**: Contains functions to train the different models (M1, M2, M3). Currently uses Decision Tree, Random Forest, and SVC. Modify this to add/change models or load pre-trained ones. Uses integer labels for `.fit()`.
* **`nomad/evaluation.py`**: Defines the `ModelWrapper` class to hold models and their metadata. Includes `calculate_quality_metrics` to evaluate models on the validation set (using integer labels), measure cost (using `timeit`), and determine `predict_proba` capability. Handles conversion between integer-based metrics and string-based attributes in the wrapper.
* **`nomad/nomad_strategy.py`**: Implements the core NOMAD algorithm logic:
    * `determine_exit_classes`: Calculates Exit Classes based on epsilon and role model comparison (uses string class names).
    * `check_chain_safety_conservative`: Implements the recall-based chain safety check (uses string class names).
    * `update_probabilities`: Updates class belief based on softmax scores (uses string class names).
    * `select_and_classify`: Orchestrates the model selection, execution, safety check, and exit logic for a single event (uses string class names).
* **`main.py`**: The main script that:
    * Loads configuration.
    * Calls `data_utils` to load and split data.
    * Calls `model_training` to train models.
    * Calls `evaluation` to wrap models and get metrics/cost.
    * Calls `nomad_strategy` functions to set up exit classes.
    * Calculates initial class probabilities.
    * Runs the event-by-event simulation using `select_and_classify`.
    * Evaluates the overall NOMAD strategy performance against the test set and compares it to the role model baseline.
    * Checks if the quality guarantee (epsilon-comparable recall) was met.
* **`requirements.txt`**: Lists necessary Python packages.

## Setup

1.  **Clone the Repository (if applicable):**
    ```bash
    # git clone <repository_url>
    cd nomad_project
    ```
2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows use `venv\Scripts\activate`
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Prepare Data:**
    * Place your dataset (e.g., a CSV file) in the `data/` directory (or update the path in `config.py`).
    * **Modify `nomad/data_utils.py`**: Adapt the `load_and_preprocess_data` function to correctly load, preprocess, and extract features (X) and integer labels (y\_int) from *your specific dataset format*. Ensure it also correctly identifies the list of unique string class names (`all_classes`). The current version includes a dummy data generator as a fallback.

## Configuration

Adjust parameters in `nomad/config.py` before running:

* **`DATA_FILEPATH`**: **(Required)** Set this to the correct path of your dataset relative to the `nomad_project` directory.
* **`ROLE_MODEL_NAME`**: Set this to the key (e.g., 'M1', 'M2', 'M3') corresponding to the model you want to use as the high-quality benchmark in the `model_training.py` output dictionary.
* **`EPSILON`**: Set the desired quality tolerance (e.g., `0.05` for 5%, `0.1` for 10%). Adjust this based on debugging or desired trade-off (you likely needed to increase this from the initial 0.05).
* **`MODEL_SAVE_DIR`, `RESULTS_DIR`**: Change if you want to store models or results elsewhere.
* **Model Hyperparameters**: Modify model parameters directly within `nomad/model_training.py`.
* **Artificial Cost**: If needed for testing (e.g., if measured costs don't reflect expected hierarchy), you can enable and configure the artificial cost adjustment block within `main.py` (search for `# --- 4a. Artificially Adjust Role Model Cost ---`).

## Running the Simulation

Execute the main script from the `nomad_project` directory:

```bash
python main.py
```

The script will perform all steps: data loading/splitting, training, evaluation, exit class determination, simulation, and final comparison.

## Understanding the Output

The script will print detailed logs for each step. Key sections at the end include:

### NOMAD Strategy Performance Evaluation:

* Overall Accuracy.
* Average Cost per Event (lower is better).
* Average Chain Length per Event.
* Per-Class Metrics (Recall, Precision, F1-Score).
* Confusion Matrix for the NOMAD strategy's predictions.
* Most Common Model Chains executed.

### Role Model Baseline Performance:

* Accuracy and Per-Class Recalls for the Role Model if run on every event.
* Static Cost per Event for the Role Model.

### Quality Guarantee Check:

* Compares NOMAD's per-class recall against the `(1 - epsilon) * Role Model Recall` bound. Indicates if the strategy met the requirement for each class.

### Cost Comparison:

* Shows the average cost of NOMAD vs. the Role Model cost.
* Estimates percentage cost savings.

## Customization and Extension

* **Adding/Changing Models:** Modify `nomad/model_training.py` to include different scikit-learn classifiers or load your own pre-trained models. Ensure they have `.fit()`, `.predict()`, and ideally `.predict_proba()` methods. Update the model dictionary keys and potentially `config.ROLE_MODEL_NAME`.
* **Changing Cost Metric:** Modify the `cost` calculation within `nomad/evaluation.py`. You could use FLOPs, memory usage, or other relevant metrics instead of inference time.
* **Relaxed Chain Safety:** Implement the "Relaxed Chain Safety Estimation" logic (using confusion matrices) from the paper in `nomad_strategy.py` as an alternative function (e.g., `check_chain_safety_relaxed`) and point to it in `main.py`.
* **Different Quality Metric for EC:** Change `config.QUALITY_METRIC_FOR_EC` and update `determine_exit_classes` if needed to use precision, F1, etc., for defining Exit Classes.
* **Data Source:** Modify `nomad/data_utils.py` to handle different data formats or sources.

## Dependencies

* Python 3.8+
* NumPy
* Pandas
* Scikit-learn

See `requirements.txt` for specific versions used during development.

## License

(Specify your license here, e.g., MIT, Apache 2.0, or leave blank if undecided)

## Citation

(If this code is based on a specific published paper, please add the citation details here.)