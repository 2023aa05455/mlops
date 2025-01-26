import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score


def load_data():
    """
    Load the breast cancer dataset and prepare features and labels.
    Convert all integer columns to float64.
    """
    data = load_breast_cancer()
    print("Dataset shape:", data.data.shape)
    X = pd.DataFrame(data.data, columns=data.feature_names).astype("float64")
    y = pd.Series(data.target, name="target").astype("float64")  # Convert target to float64
    return X, y


def preprocess_data(X, y):
    """
    Standardize the features and split the data into training and testing sets.
    """
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), columns=X.columns
    )  # Retain feature names
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Train a Logistic Regression model and evaluate its performance.
    """
    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return model, accuracy, precision, f1


def main():
    """
    Main function to train the model, log metrics, and save the model with MLflow.
    """
    # Set MLflow experiment
    mlflow.set_experiment("Cancer Detection v1")

    # Load and preprocess data
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    print("Training and evaluating Logistic Regression Model...")

    # Start MLflow run
    with mlflow.start_run(run_name="LogisticRegression"):
        # Train and evaluate the model
        model, accuracy, precision, f1 = train_and_evaluate(
            X_train, X_test, y_train, y_test
        )

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1_score", f1)

        # Create an input example (convert to float64 for MLflow schema)
        input_example = X.iloc[:1, :].astype("float64")

        # Log the model with the input example
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=input_example
        )

        # Print model metrics
        print(
            "------Logistic Regression Metrics ------\n"
            f"Accuracy: {accuracy:.2f}\n"
            f"Precision: {precision:.2f}\n"
            f"F1-score: {f1:.2f}\n"
            "----------------------------------------"
        )


if __name__ == "__main__":
    main()
