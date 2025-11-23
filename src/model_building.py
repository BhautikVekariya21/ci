# src/model_building.py
import pandas as pd
import pickle
import os
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from utils import load_params, logger

# Initialize DagsHub
dagshub.init(repo_owner='BhautikVekariya21', repo_name='ci', mlflow=True)

def main():
    # Load all params
    all_params = load_params()
    params = all_params["model_building"]
    
    # Set MLflow experiment
    mlflow.set_experiment("sentiment_classification")
    
    # Start MLflow run
    with mlflow.start_run(run_name="random_forest_training"):
        
        # Log all parameters from different stages
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
        with open("models/model.pkl", "wb") as f:
            pickle.dump(model, f)
        
        # Log model to MLflow (DagsHub) - REMOVED registered_model_name
        mlflow.sklearn.log_model(
            model, 
            "model"
            # REMOVED: registered_model_name="RandomForestSentimentClassifier"
        )
        
        # Log model artifact
        mlflow.log_artifact("models/model.pkl")
        
        # Log feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature_index': range(len(model.feature_importances_)),
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            feature_importance.to_csv("models/feature_importance.csv", index=False)
            mlflow.log_artifact("models/feature_importance.csv")
            
            # Log top 5 feature importances as metrics
            for idx, row in feature_importance.head(5).iterrows():
                mlflow.log_metric(f"feature_{int(row['feature_index'])}_importance", row['importance'])
        
        # Log model parameters summary
        mlflow.log_metric("model_n_estimators", model.n_estimators)
        mlflow.log_metric("model_max_depth", model.max_depth if model.max_depth else 0)
        
        # Tag the model
        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.set_tag("framework", "sklearn")
        mlflow.set_tag("task", "sentiment_classification")
        
        logger.info("✅ Model trained and logged to DagsHub successfully")
        logger.info(f"🌐 View on DagsHub: https://dagshub.com/BhautikVekariya21/ci.mlflow")

if __name__ == "__main__":
    main()