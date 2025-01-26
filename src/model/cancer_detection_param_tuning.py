import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score

# Configuration
CONFIG = {
    "random_seed": 42,
    "test_size": 0.2,
    "param_grids": {
        "RandomForest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "SVM": {
            "kernel": ["linear", "rbf"],
            "C": [0.1, 1.0, 10.0],
            "gamma": ["scale", "auto"],
        },
        "LogisticRegression": {
            "solver": ["liblinear", "lbfgs"],
            "C": [0.1, 1.0, 10.0],
        },
    },
}
print("MLflow tracking URI:", mlflow.get_tracking_uri())


# Step 1: Load dataset
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    print("Dataset shape:", df.shape)
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y


# Step 2: Preprocess the data
def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_seed"]
    )
    return X_train, X_test, y_train, y_test


# Step 3: Train and evaluate models with GridSearchCV
def train_and_evaluate(model_name, param_grid, X_train, X_test, y_train, y_test):
    # Define model
    if model_name == "RandomForest":
        model = RandomForestClassifier(random_state=CONFIG["random_seed"])
    elif model_name == "SVM":
        model = SVC(random_state=CONFIG["random_seed"])
    elif model_name == "LogisticRegression":
        model = LogisticRegression(random_state=CONFIG["random_seed"])
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Perform GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=3,
        verbose=2,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    # Best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best parameters for {model_name}: {best_params}")

    # Evaluate on the test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return best_model, best_params, accuracy, precision, f1


# Step 4: Main pipeline with MLflow integration
def main():
    # Initialize MLflow experiment
    mlflow.set_experiment("Cancer Detection v4")

    # Load and preprocess data
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    for model_name, param_grid in CONFIG["param_grids"].items():
        with mlflow.start_run(run_name=model_name):
            print(f"Training and evaluating {model_name} with hyperparameter tuning...")

            # Train and evaluate
            best_model, best_params, accuracy, precision, f1 = train_and_evaluate(
                model_name, param_grid, X_train, X_test, y_train, y_test
            )

            # Log parameters, metrics, and model to MLflow
            mlflow.log_params(best_params)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("f1_score", f1)
            mlflow.sklearn.log_model(best_model, artifact_path="model")

            print(
                f"{model_name} metrics - "
                f"Accuracy: {accuracy:.2f}, "
                f"Precision: {precision:.2f}, "
                f"F1-score: {f1:.2f}"
            )


if __name__ == "__main__":
    main()
