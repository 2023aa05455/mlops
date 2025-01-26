import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier

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
        X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


# Step 3: Train and evaluate models


def train_and_evaluate(X_train, X_test, y_train, y_test):
    # model = LogisticRegression(solver="liblinear", random_state=42)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return model, accuracy, precision, f1


def main():
    # Initialize MLflow experiment
    mlflow.set_experiment("Cancer Detection v2")

    # Load and preprocess data
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    print("Training and evaluating RandomForest Model ..!")
    # Start MLFLOW run
    mlflow.start_run(run_name="RandomForestClassifier")

    # Train and evaluate
    model, accuracy, precision, f1 = train_and_evaluate(
        X_train, X_test, y_train, y_test
    )
    # Log parameters, metrics, and model to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, artifact_path="model")

    # Log parameters and metrics
    print(
        "------RandomForestClassifier Metrics ----- \n",
        f"Accuracy: {accuracy:.2f}, "
        f"Precision: {precision:.2f}, "
        f"F1-score: {f1:.2f}\n",
        "---------------------------------------"
    )


if __name__ == "__main__":
    main()
