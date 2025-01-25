import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score

# Configuration
CONFIG = {
    "random_seed": 42,
    "test_size": 0.2,
    "models": {
        "RandomForest": {"n_estimators": 100, "max_depth": 5},
        "SVM": {"kernel": "linear", "C": 1.0},
        "LogisticRegression": {"solver": "liblinear"},
    },
}
print("MLflow tracking URI:", mlflow.get_tracking_uri())

# Step 1: Load dataset


def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)  # Check the shape
    # (number_of_samples, number_of_features)
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


# Step 3: Train and evaluate models


def train_and_evaluate(model_name, model_config, X_train, X_test, y_train, y_test):
    if model_name == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=model_config["n_estimators"],
            max_depth=model_config["max_depth"],
            random_state=CONFIG["random_seed"],
        )
    elif model_name == "SVM":
        model = SVC(
            kernel=model_config["kernel"],
            C=model_config["C"],
            random_state=CONFIG["random_seed"],
        )
    elif model_name == "LogisticRegression":
        model = LogisticRegression(
            solver=model_config["solver"], random_state=CONFIG["random_seed"]
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return model, accuracy, precision, f1


# Step 4: Main pipeline with MLflow integration


def main():
    # Initialize MLflow experiment
    mlflow.set_experiment("Cancer Detection v4")

    # Load and preprocess data
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    for model_name, model_config in CONFIG["models"].items():
        with mlflow.start_run(run_name=model_name):
            print(f"Training and evaluating {model_name}...")

            # Train and evaluate
            model, accuracy, precision, f1 = train_and_evaluate(
                model_name, model_config, X_train, X_test, y_train, y_test
            )
            # Log parameters, metrics, and model to MLflow
            mlflow.log_params(model_config)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("f1_score", f1)
            mlflow.sklearn.log_model(model, artifact_path="model")

            print(
                f"{model_name} metrics - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, F1-score: {f1:.2f}"
            )


if __name__ == "__main__":
    main()
