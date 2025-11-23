# src/model_evaluation.py
import pandas as pd
import pickle
import json
import mlflow
import mlflow.sklearn
import os
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    roc_auc_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_confusion_matrix(y_test, y_pred, save_path="evaluation/confusion_matrix.png"):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Sadness', 'Happiness'],
                yticklabels=['Sadness', 'Happiness'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    return cm

def main():
    # Setup MLflow
    mlflow_backend = setup_mlflow()
    
    params = load_params()['model_evaluation']['metrics']
    
    # Set experiment
    mlflow.set_experiment("sentiment_classification")
    
    # Start MLflow run
    with mlflow.start_run(run_name="model_evaluation"):
        
        mlflow.set_tag("mlflow_backend", mlflow_backend)
        mlflow.set_tag("stage", "evaluation")
        mlflow.set_tag("model_type", "RandomForestClassifier")
        
        # Load model
        logger.info("Loading model for evaluation...")
        model = pickle.load(open("models/model.pkl", "rb"))
        test_df = pd.read_csv("data/features/test_bow.csv")

        X_test = test_df.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1].values
        
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", X_test.shape[1])

        # Predictions
        logger.info("Making predictions...")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics_dict = {}
        
        if "accuracy" in params:
            metrics_dict["accuracy"] = round(accuracy_score(y_test, y_pred), 4)
            mlflow.log_metric("accuracy", metrics_dict["accuracy"])
            
        if "precision" in params:
            metrics_dict["precision"] = round(precision_score(y_test, y_pred), 4)
            mlflow.log_metric("precision", metrics_dict["precision"])
            
        if "recall" in params:
            metrics_dict["recall"] = round(recall_score(y_test, y_pred), 4)
            mlflow.log_metric("recall", metrics_dict["recall"])
            
        if "auc" in params:
            metrics_dict["auc"] = round(roc_auc_score(y_test, y_proba), 4)
            mlflow.log_metric("auc", metrics_dict["auc"])
        
        f1 = round(f1_score(y_test, y_pred), 4)
        metrics_dict["f1_score"] = f1
        mlflow.log_metric("f1_score", f1)

        logger.info("📊 Model Metrics:")
        for metric_name, value in metrics_dict.items():
            logger.info(f"  {metric_name}: {value:.4f}")

        # Save metrics
        os.makedirs("evaluation", exist_ok=True)
        with open("evaluation/metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=4)
        
        mlflow.log_artifact("evaluation/metrics.json")
        
        # Confusion matrix
        logger.info("Creating confusion matrix...")
        cm = plot_confusion_matrix(y_test, y_pred)
        mlflow.log_artifact("evaluation/confusion_matrix.png")
        
        mlflow.log_metric("true_negatives", int(cm[0][0]))
        mlflow.log_metric("false_positives", int(cm[0][1]))
        mlflow.log_metric("false_negatives", int(cm[1][0]))
        mlflow.log_metric("true_positives", int(cm[1][1]))
        
        total_samples = cm.sum()
        correctly_classified = cm.trace()
        misclassified = total_samples - correctly_classified
        
        mlflow.log_metric("total_samples", int(total_samples))
        mlflow.log_metric("correctly_classified", int(correctly_classified))
        mlflow.log_metric("misclassified", int(misclassified))
        mlflow.log_metric("error_rate", float(misclassified / total_samples))
        
        # Classification report
        report = classification_report(y_test, y_pred, 
                                      target_names=['Sadness', 'Happiness'],
                                      output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv("evaluation/classification_report.csv")
        mlflow.log_artifact("evaluation/classification_report.csv")
        
        mlflow.log_metric("sadness_precision", report['Sadness']['precision'])
        mlflow.log_metric("sadness_recall", report['Sadness']['recall'])
        mlflow.log_metric("sadness_f1", report['Sadness']['f1-score'])
        mlflow.log_metric("happiness_precision", report['Happiness']['precision'])
        mlflow.log_metric("happiness_recall", report['Happiness']['recall'])
        mlflow.log_metric("happiness_f1", report['Happiness']['f1-score'])
        
        mlflow.sklearn.log_model(model, "evaluated_model")
        
        if mlflow_backend == "dagshub":
            logger.info("✅ Evaluation logged to DagsHub")
            logger.info("🌐 View: https://dagshub.com/BhautikVekariya21/ci.mlflow")
        else:
            logger.info("✅ Evaluation logged locally")

if __name__ == "__main__":
    main()