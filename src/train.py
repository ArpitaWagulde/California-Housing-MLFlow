import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# --- 1. Set MLflow Experiment ---
# This will create a new experiment if it doesn't exist
try:
    mlflow.set_experiment("california-housing")
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
with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")

    # --- 4. Log Parameters ---
    n_estimators = 100
    max_depth = 8
    
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    print("Logged parameters (n_estimators, max_depth)")

    # --- 5. Train Model ---
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)

    # --- 6. Log Metrics ---
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    print(f"Logged metrics (RMSE: {rmse:.4f})")

    # --- 7. Log and Register Model ---
    # This is the key step: we log the model and register it in one command.
    print("Logging and registering the model...")
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="random-forest-model",  # Name in the "artifacts" section
        input_example=X_train[:5], # Helps with schema inference
        registered_model_name="california-housing-rf" # Name in the Model Registry
    )
    
print(f"Run {run_id} finished.")
print(f"Model 'california-housing-rf' is registered.")