"""
End-to-End MLflow Lab: Training and Registration

This script demonstrates a complete MLOps workflow using MLflow for the
California Housing dataset:

1.  Sets up an MLflow Experiment.
2.  Loads and preprocesses the Scikit-learn California Housing dataset.
3.  Starts an MLflow run.
4.  Logs model hyperparameters (n_estimators, max_depth).
5.  Trains a RandomForestRegressor model.
6.  Logs model performance metrics (MSE, RMSE).
7.  Logs the trained model and simultaneously registers it in the
    MLflow Model Registry under the name 'california-housing-rf'.
"""

import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# --- 1. Set MLflow Experiment ---
# We set the experiment name. If it doesn't exist, MLflow creates it.
# All our runs will be grouped under this experiment.
EXPERIMENT_NAME = "california-housing"
try:
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"Using experiment: '{EXPERIMENT_NAME}'")
except mlflow.exceptions.MlflowException as e:
    print(f"Could not set experiment: {e}")
    print("Using default '0' experiment.")


# --- 2. Load Data ---
print("Loading California Housing data...")
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

# --- 3. Start MLflow Run & Train Model ---
print("Starting MLflow run...")

# mlflow.start_run() creates a new run and makes it "active".
# Using 'with' ensures the run is automatically closed (ended)
# even if errors occur.
with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")

    # --- 4. Log Parameters ---
    # Define hyperparameters for this run
    n_estimators = 100
    max_depth = 8
    
    # Log parameters to make runs reproducible and comparable
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    print(f"Logged parameters: n_estimators={n_estimators}, max_depth={max_depth}")

    # --- 5. Train Model ---
    print("Training RandomForestRegressor...")
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)

    # --- 6. Log Metrics ---
    # Evaluate the model on the test set
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    # Log metrics to track model performance
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    print(f"Logged metrics: MSE={mse:.4f}, RMSE={rmse:.4f}")

    # --- 7. Log and Register Model ---
    # This is the key step that connects tracking with the registry.
    print("Logging and registering the model...")
    
    # 'registered_model_name' is the name in the Model Registry. If this
    # model name doesn't exist, MLflow creates it. If it does, MLflow
    # creates a new version.
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="random-forest-model",  # Name in the "artifacts" section
        input_example=X_train[:5], # Helps with schema inference
        registered_model_name="california-housing-rf" # Name in the Model Registry
    )
    
print(f"\n--- Run {run_id} finished. ---")
print(f"Model '{model_info.registered_model_name}' version {model_info.version} is registered.")