# California Housing Prediction with MLFlow Integration

## Overview

This project demonstrates a foundational, end-to-end MLOps workflow using MLflow. It covers the full lifecycle of a model:

1.  **Experiment Tracking:** Training a model and logging its parameters, metrics, and the model file itself.
2.  **Model Registration:** Registering the trained model in the MLflow Model Registry.
3.  **Model Promotion:** Automatically validating the model against a performance threshold (a "quality gate") and promoting it to "Production" using aliases.

The project uses the California Housing Prices dataset to train a `RandomForestRegressor`.

---

## Workflow Explained

This lab is split into two main scripts:

### 1. `src/train.py`: Training & Registration
This script simulates the work of a data scientist. It:
* Sets the MLflow experiment to `california-housing`.
* Starts a new MLflow run.
* Logs the hyperparameters (`n_estimators`, `max_depth`).
* Trains a `RandomForestRegressor` on the housing data.
* Logs the resulting performance metrics (`mse`, `rmse`).
* Logs the trained model and **registers it** under the name `california-housing-rf` in the MLflow Model Registry.

### 2. `src/promote.py`: Validation & Promotion
This script simulates an automated CI/CD or "GitOps" process. It acts as a quality gate for production. It:
* Finds the **latest completed run** from the `california-housing` experiment.
* Checks its `mse` metric.
* **Validation Gate:** If the `mse` is less than or equal to our threshold (`0.4`), the script proceeds.
* If the model passes, it finds the corresponding model version (e.g., `Version 1`) and **sets its alias to "Production"**.
* It also removes the "Production" alias from any old versions, ensuring only one model is live at a time.

---

## Getting Started

Here is the step-by-step process to run the full workflow.

### 1. Environment Setup
You can set up a virtual environment and install the necessary dependencies using the following commands:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start the MLflow UI

Run following command in a new terminal. This will create a new mlruns directory in your folder to store all the experiment data.

```bash
source .venv/bin/activate
mlflow ui
```

Leave this terminal running.

### 3. Run Training script

In your original terminal, run the training script. This will train your first model and register it.

```bash
python src/train.py
```

### 4. Run the Validation and Promotion Script

Now, run the promotion script. This script will find the latest model you just trained, check its mse, and (if it passes) promote it to "Production".

```bash
python src/promote.py
```

### 4. Observe the Results in MLflow

Go to the MLflow UI in your browser (http://127.0.0.1:5000) and refresh the page.

#### 1. In the "Experiments" Tab

- Click on the california-housing experiment.

- You will see your run (or multiple runs if you ran train.py more than once).

- Click on the latest run to see the Parameters (e.g., max_depth) and Metrics (e.g., mse) that were logged.

#### 2. In the "Models" Tab

- Click on the california-housing-rf model.

- You will see all the versions you've registered (e.g., Version 1, Version 2).

- Look at the "Aliases" column for your latest version. After running promote.py, you will see a "Production" tag there, indicating the model was successfully validated and promoted.

### 5. Experimentation

You can experiment with multiple hyperparameters by modifying the training script and running the train script and promote script.