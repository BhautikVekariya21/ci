# src/utils.py
import yaml
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def load_params() -> dict:
    try:
        with open("params.yaml", "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
        logger.info("Parameters loaded successfully from params.yaml")
        return params or {}
    except Exception as e:
        logger.error(f"Failed to load params.yaml: {e}")
        raise

def setup_mlflow():
    """Setup MLflow tracking URI and other configurations"""
    # Set tracking URI (local by default)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    
    logger.info(f"MLflow tracking URI set to: {tracking_uri}")
    
    return tracking_uri