# src/train_and_evaluate.py
import mlflow
from model_building import main as train_model
from model_evaluation import main as evaluate_model
from utils import logger

def main():
    """Combined training and evaluation with single MLflow run"""
    
    mlflow.set_experiment("sentiment_classification")
    
    with mlflow.start_run(run_name="full_pipeline") as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        
        # Training phase
        logger.info("Starting model training...")
        train_model()
        
        # Evaluation phase
        logger.info("Starting model evaluation...")
        evaluate_model()
        
        logger.info(f"Pipeline complete. View results at: http://localhost:5000/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")

if __name__ == "__main__":
    main()