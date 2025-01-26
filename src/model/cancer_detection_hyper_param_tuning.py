import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score

# Configuration
CONFIG = {
    "random_seed": 42,
    "test_size": 0.2,
    "param_grids": {
            "solver": ["liblinear", "lbfgs"],
            "C": [0.1, 1.0, 10.0],
        },
}


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
        X_scaled, y, test_size=CONFIG["test_size"],
        random_state=CONFIG["random_seed"]
    )
    return X_train, X_test, y_train, y_test


# Step 3: Train and evaluate models with GridSearchCV
def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Define model
    model = LogisticRegression(random_state=CONFIG["random_seed"])
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }

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
    print(f"Best parameters for LogisticRegression: {best_params}")

    # Evaluate on the test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return best_model, best_params, accuracy, precision, f1


# Step 4: Main pipeline with MLflow integration
def main():
    # Load and preprocess data
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    print("Training and evaluating Model with hyperparameter tuning...")

    # Train and evaluate
    best_model, best_params, accuracy, precision, f1 = train_and_evaluate(
        X_train, X_test, y_train, y_test
    )
    print(
        " ------Model Performance Metrics----- \n"
        f"Accuracy: {accuracy:.2f}, "
        f"Precision: {precision:.2f}, "
        f"F1-score: {f1:.2f}"
    )


if __name__ == "__main__":
    main()
