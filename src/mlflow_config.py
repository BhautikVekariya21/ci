# mlflow_config.py
import mlflow
import os

class MLflowConfig:
    """MLflow configuration manager"""
    
    def __init__(self, experiment_name="sentiment_classification"):
        self.experiment_name = experiment_name
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        
    def setup(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        print(f"✅ MLflow tracking URI: {self.tracking_uri}")
        print(f"✅ MLflow experiment: {self.experiment_name}")
        
    @staticmethod
    def get_latest_run_id(experiment_name="sentiment_classification"):
        """Get the latest run ID from an experiment"""
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            if not runs.empty:
                return runs.iloc[0]['run_id']
        return None