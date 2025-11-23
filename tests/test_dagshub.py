# test_dagshub.py
import dagshub
import mlflow

# Initialize DagsHub
dagshub.init(repo_owner='BhautikVekariya21', repo_name='ci', mlflow=True)

# Test MLflow logging
with mlflow.start_run(run_name="test_dagshub"):
    mlflow.log_param('test_parameter', 'test_value')
    mlflow.log_metric('test_metric', 123.45)
    print("✅ Successfully logged to DagsHub!")
    print("🌐 View at: https://dagshub.com/BhautikVekariya21/ci.mlflow")