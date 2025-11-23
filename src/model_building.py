# src/model_building.py
import pandas as pd
import pickle
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from utils import load_params, logger

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
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=params["random_state"],
            n_jobs=params.get("n_jobs", -1)
        )
        model.fit(X_train, y_train)
        
        # Save model locally
        os.makedirs("models", exist_ok=True)
        with open("models/model.pkl", "wb") as f:
            pickle.dump(model, f)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="RandomForestSentimentClassifier"
        )
        
        # Log model artifact
        mlflow.log_artifact("models/model.pkl")
        
        # Log feature importance if available
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature_index': range(len(model.feature_importances_)),
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            feature_importance.to_csv("models/feature_importance.csv", index=False)
            mlflow.log_artifact("models/feature_importance.csv")
        
        logger.info("Random-Forest model trained and logged to MLflow successfully")

if __name__ == "__main__":
    main()