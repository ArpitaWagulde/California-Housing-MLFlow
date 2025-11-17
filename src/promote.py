"""
MLflow Model Validation and Promotion Script

This script automates the CI/CD process for a registered MLflow model.
It performs the following steps:

1.  Connects to the MLflow tracking server.
2.  Finds the most recently completed run within a specified experiment
    (EXPERIMENT_NAME).
3.  Retrieves the 'mse' (Mean Squared Error) metric from that run.
4.  Validates the metric: checks if the 'mse' is below a defined
    VALIDATION_THRESHOLD.
5.  If validation passes:
    a.  Finds the model version in the MLflow Model Registry (MODEL_NAME)
        that was created by that specific run.
    b.  Promotes this model version by setting its alias to "Production".
    c.  Atomically removes the "Production" alias from any previous version
        to ensure only one model is marked for production.
6.  If validation fails, it prints a failure message and exits.
"""

import mlflow
from mlflow.tracking import MlflowClient

# --- Configuration ---
# These constants define the model and experiment this script will target.
EXPERIMENT_NAME = "california-housing"
MODEL_NAME = "california-housing-rf"
# Define the business rule for promotion: lower MSE is better.
VALIDATION_THRESHOLD = 0.4 # Promote if MSE is <= 0.4
# ---------------------

print(f"Starting model validation for experiment: '{EXPERIMENT_NAME}'")

# Initialize MLflow client
# The MlflowClient provides a low-level API to interact with the
# tracking server, experiments, runs, and model registry.
client = MlflowClient()

# Set the experiment to get its ID
try:
    # We must find the experiment_id to search for runs within it.
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        print(f"Error: Experiment '{EXPERIMENT_NAME}' not found.")
        exit()
    experiment_id = exp.experiment_id
except Exception as e:
    print(f"Error setting experiment: {e}")
    exit()

# --- 1. Find the latest run ---
print("Searching for the latest completed run...")
try:
    # Search for runs within our experiment
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment_id],
        # Filter for only successfully completed runs
        filter_string="attributes.status = 'FINISHED'",
        # Order by time to get the most recent run first
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if runs_df.empty:
        print("No completed runs found for this experiment.")
        exit()
        
    # Get the first row (which is the latest run)
    latest_run = runs_df.iloc[0]
    run_id = latest_run.run_id
    mse = latest_run["metrics.mse"] # Extract the MSE metric
    
    print(f"Found latest run: {run_id}")
    print(f"  - Run Metric (MSE): {mse:.4f}")
    
except Exception as e:
    print(f"Error searching for runs: {e}")
    exit()

# --- 2. Run Validation ---
print(f"Running validation: (MSE {mse:.4f} <= {VALIDATION_THRESHOLD}?)")

# Check if the model's performance meets our criteria
if mse <= VALIDATION_THRESHOLD:
    print("  - Validation PASSED.")
    
    # --- 3. Find and Promote Model using Aliases ---
    print(f"Searching for model version linked to run {run_id}...")
    
    model_version_found = None
    # Get all versions of our registered model
    all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    
    # Loop through all versions to find the one created by our 'run_id'
    for mv in all_versions:
        if mv.run_id == run_id:
            model_version_found = mv
            break # Found it
            
    if model_version_found:
        version_number = model_version_found.version
        print(f"  - Found: Version {version_number} of model '{MODEL_NAME}'")
        
        # --- New Alias Logic ---
        # This section ensures only our new model is "Production"
        print(f"  - Setting 'Production' alias for version {version_number}...")
        
        # 1. Remove the "Production" alias if it exists on another version.
        #    This makes the promotion atomic.
        try:
            client.delete_registered_model_alias(name=MODEL_NAME, alias="Production")
            print("    - Cleared 'Production' alias from any old versions.")
        except mlflow.exceptions.RestException as e:
            # This error is expected if no model is currently "Production"
            if e.error_code == "RESOURCE_DOES_NOT_EXIST":
                print("    - No old version had 'Production' alias. Good to proceed.")
            else:
                raise # Re-raise any other unexpected errors
            
        # 2. Set the "Production" alias on our new, validated version
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="Production",
            version=version_number
        )
        
        print(f"\nSUCCESS: Set 'Production' alias for model version {version_number}.")
        
    else:
        # This can happen if train.py logged a run but failed to register the model
        print(f"Error: No model version found in registry for run {run_id}.")

else:
    # This block executes if the MSE was worse than the threshold
    print("  - Validation FAILED.")
    print("Model will not be promoted.")