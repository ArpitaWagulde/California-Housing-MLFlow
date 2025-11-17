import mlflow
from mlflow.tracking import MlflowClient

# --- Configuration ---
EXPERIMENT_NAME = "california-housing"
MODEL_NAME = "california-housing-rf"
VALIDATION_THRESHOLD = 0.4 # Promote if MSE is <= 0.4
# ---------------------

print(f"Starting model validation for experiment: '{EXPERIMENT_NAME}'")

# Initialize MLflow client
client = MlflowClient()

# Set the experiment
try:
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
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if runs_df.empty:
        print("No completed runs found for this experiment.")
        exit()
        
    latest_run = runs_df.iloc[0]
    run_id = latest_run.run_id
    mse = latest_run["metrics.mse"]
    
    print(f"Found latest run: {run_id}")
    print(f"  - Run Metric (MSE): {mse:.4f}")
    
except Exception as e:
    print(f"Error searching for runs: {e}")
    exit()

# --- 2. Run Validation ---
print(f"Running validation: (MSE {mse:.4f} <= {VALIDATION_THRESHOLD}?)")

if mse <= VALIDATION_THRESHOLD:
    print("  - Validation PASSED.")
    
    # --- 3. Find and Promote Model using Aliases ---
    print(f"Searching for model version linked to run {run_id}...")
    
    model_version_found = None
    all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    
    for mv in all_versions:
        if mv.run_id == run_id:
            model_version_found = mv
            break
            
    if model_version_found:
        version_number = model_version_found.version
        print(f"  - Found: Version {version_number} of model '{MODEL_NAME}'")
        
        # --- New Alias Logic ---
        print(f"  - Setting 'Production' alias for version {version_number}...")
        
        # 1. Remove the "Production" alias if it exists on another version
        try:
            client.delete_registered_model_alias(name=MODEL_NAME, alias="Production")
            print("    - Cleared 'Production' alias from any old versions.")
        except mlflow.exceptions.RestException as e:
            if e.error_code == "RESOURCE_DOES_NOT_EXIST":
                print("    - No old version had 'Production' alias. Good to proceed.")
            else:
                raise # Re-raise other errors
            
        # 2. Set the "Production" alias on our new version
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="Production",
            version=version_number
        )
        
        print(f"\nSUCCESS: Set 'Production' alias for model version {version_number}.")
        
    else:
        print(f"Error: No model version found in registry for run {run_id}.")

else:
    print("  - Validation FAILED.")
    print("Model will not be promoted.")