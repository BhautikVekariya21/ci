# src/model_building.py
import pandas as pd
import pickle
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from utils import load_params, logger

def setup_mlflow():
    """Setup MLflow with token-based authentication for DagsHub"""
    
    is_ci = os.getenv('CI') == 'true'
    dagshub_token = os.getenv('DAGSHUB_TOKEN')
    
    if is_ci and dagshub_token:
        # Use DagsHub with token authentication
        dagshub_url = "https://dagshub.com/BhautikVekariya21/ci.mlflow"
        
        os.environ['MLFLOW_TRACKING_URI'] = dagshub_url
        os.environ['MLFLOW_TRACKING_USERNAME'] = 'BhautikVekariya21'
        os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
        
        mlflow.set_tracking_uri(dagshub_url)
        
        logger.info("✅ Using DagsHub MLflow with token authentication")
        return "dagshub"
    else:
        mlflow.set_tracking_uri("file:./mlruns")
        logger.info("✅ Using local MLflow tracking")
        return "local"

def main():
    # Setup MLflow
    mlflow_backend = setup_mlflow()
    
    # Load all params
    all_params = load_params()
    params = all_params["model_building"]
    
    # Set MLflow experiment
    mlflow.set_experiment("sentiment_classification")
    
    # Start MLflow run
    with mlflow.start_run(run_name="random_forest_training"):
        
        # Log backend and tags
        mlflow.set_tag("mlflow_backend", mlflow_backend)
        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.set_tag("framework", "sklearn")
        mlflow.set_tag("task", "sentiment_classification")
        
        # Log all parameters
        mlflow.log_params({
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "random_state": params["random_state"],
            "n_jobs": params.get("n_jobs", -1)
        })
        
        # Log data ingestion params
        mlflow.log_params({
            "test_size": all_params["data_ingestion"]["test_size"],
            "data_random_state": all_params["data_ingestion"]["random_state"]
        })
        
        # Log feature engineering params
        mlflow.log_params({
            "max_features": all_params["feature_engineering"]["max_features"],
            "ngram_range": str(all_params["feature_engineering"]["ngram_range"])
        })
        
        # Log preprocessing params
        preprocessing_params = all_params["data_preprocessing"]
        mlflow.log_params({
            "lowercase": preprocessing_params.get("lowercase", True),
            "remove_urls": preprocessing_params.get("remove_urls", True),
            "remove_punctuations": preprocessing_params.get("remove_punctuations", True),
            "remove_numbers": preprocessing_params.get("remove_numbers", True),
            "remove_stopwords": preprocessing_params.get("remove_stopwords", True),
            "apply_lemmatization": preprocessing_params.get("apply_lemmatization", True),
            "min_words_per_tweet": preprocessing_params["min_words_per_tweet"]
        })
        
        # Load training data
        train_df = pd.read_csv("data/features/train_bow.csv")
        X_train = train_df.iloc[:, :-1].values
        y_train = train_df.iloc[:, -1].values
        
        # Log dataset info
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("n_features", X_train.shape[1])
        
        # Train model
        logger.info("Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=params["random_state"],
            n_jobs=params.get("n_jobs", -1)
        )
        model.fit(X_train, y_train)
        
        # Training score
        train_score = model.score(X_train, y_train)
        mlflow.log_metric("train_accuracy", train_score)
        logger.info(f"Training accuracy: {train_score:.4f}")
        
        # Save model locally
        os.makedirs("models", exist_ok=True)
        model_path = "models/model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        # Log model as artifact (compatible with DagsHub)
        try:
            # Try to log model with MLflow sklearn
            if mlflow_backend == "local":
                # Full model logging for local MLflow
                mlflow.sklearn.log_model(model, "model")
            else:
                # For DagsHub, just log as artifact to avoid unsupported endpoints
                mlflow.log_artifact(model_path, "model")
                logger.info("Model logged as artifact (DagsHub compatible)")
        except Exception as e:
            # Fallback to artifact logging
            logger.warning(f"MLflow model logging failed: {e}")
            logger.info("Falling back to artifact logging...")
            mlflow.log_artifact(model_path, "model")
        
        # Log feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature_index': range(len(model.feature_importances_)),
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            importance_path = "models/feature_importance.csv"
            feature_importance.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)
            
            # Log top 5 feature importances as metrics
            for idx, row in feature_importance.head(5).iterrows():
                mlflow.log_metric(f"feature_{int(row['feature_index'])}_importance", row['importance'])
        
        # Log model parameters
        mlflow.log_metric("model_n_estimators", model.n_estimators)
        mlflow.log_metric("model_max_depth", model.max_depth if model.max_depth else 0)
        
        if mlflow_backend == "dagshub":
            logger.info("✅ Model trained and logged to DagsHub")
            logger.info("🌐 View: https://dagshub.com/BhautikVekariya21/ci.mlflow")
        else:
            logger.info("✅ Model trained and logged locally")
            logger.info("🌐 View: Run 'mlflow ui' and visit http://localhost:5000")

if __name__ == "__main__":
    main()