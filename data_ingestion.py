# =============================================
# Random Forest on Iris with GridSearchCV + Proper MLflow Logging
# =============================================

import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# --------------------- Data ---------------------
iris = datasets.load_iris()
X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df["species"] = pd.Categorical.from_codes(y, target_names)

print("Dataset shape:", X.shape)
print(df.head())
print("\nClass distribution:\n", df["species"].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# --------------------- Hyperparameter Grid ---------------------
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 5, 6, 7, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

rf_base = RandomForestClassifier(random_state=47)

grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=1
)

# --------------------- MLflow Setup ---------------------
mlflow.set_experiment("Iris Random Forest GridSearch + Best Model")
mlflow.autolog(log_models=False)  # We log the final model manually for full control

with mlflow.start_run(run_name="Parent_Run_GridSearch",description="best hyperparameter tunug") as parent_run:

    mlflow.set_tag("stage", "hyperparameter_tuning")
    mlflow.set_tag("dataset", "Iris")

    print("\nStarting GridSearchCV...")
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

    # --------------------- Log each CV trial as nested run ---------------------
    cv_results = grid_search.cv_results_
    for i in range(len(cv_results['params'])):
        with mlflow.start_run(run_name=f"Trial_{i+1}", nested=True):
            mlflow.log_params(cv_results['params'][i])
            mlflow.log_metric("cv_accuracy", cv_results['mean_test_score'][i])
            mlflow.log_metric("cv_std", cv_results['std_test_score'][i])
            mlflow.set_tag("gridsearch_trial", "true")

    # --------------------- Train FINAL best model ---------------------
    best_rf = grid_search.best_estimator_   # This already has the best params!

    # Or explicitly create it (same thing):
    # best_rf = RandomForestClassifier(**grid_search.best_params_, random_state=47)
    # best_rf.fit(X_train, y_train)

    y_pred = best_rf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy with best model: {test_accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

    # --------------------- Confusion Matrix ---------------------
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix - Best Model")
    plt.savefig("confusion_matrix_best.png")
    plt.close()

    # --------------------- Final Logging (in parent run) ---------------------
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("best_cv_accuracy", grid_search.best_score_)

    mlflow.log_artifact("confusion_matrix_best.png")
    mlflow.log_artifact(__file__)

    # Train/test datasets
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df["species"] = pd.Categorical.from_codes(y_train, target_names)
    test_df  = pd.DataFrame(X_test,  columns=feature_names)
    test_df["species"]  = pd.Categorical.from_codes(y_test,  target_names)

    mlflow.log_input(mlflow.data.from_pandas(train_df, targets="species"), context="training")
    mlflow.log_input(mlflow.data.from_pandas(test_df,  targets="species"), context="test")

    train_df.to_csv("train_set.csv", index=False)
    test_df.to_csv("test_set.csv", index=False)
    mlflow.log_artifact("train_set.csv")
    mlflow.log_artifact("test_set.csv")

    # --------------------- Log final model with signature ---------------------
    from mlflow.models import infer_signature
    signature = infer_signature(X_train, best_rf.predict(X_train))

    mlflow.sklearn.log_model(
        sk_model=best_rf,
        artifact_path="best_random_forest_model",
        signature=signature,
        input_example=X_train[:5],
        registered_model_name="IrisBestRandomForest"  # optional: registers in Model Registry
    )

    mlflow.set_tag("model_status", "best_from_gridsearch")

print("\nMLflow run completed! Check the UI — you will see 1 parent run + many nested child runs + final best model.")