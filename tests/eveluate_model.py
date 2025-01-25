import mlflow
import os
from sklearn.metrics import accuracy_score
import cancer_detection_model  # Import the above workflow


def test_mlflow_experiment():
    # Run the MLflow experiment
    run_id, accuracy = train_and_log_model()

    # Fetch the MLflow run
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    # Assert the run exists
    assert run is not None, "MLflow run does not exist"

    # Assert parameters
    params = run.data.params
    assert params["n_estimators"] == "100", "n_estimators not logged correctly"
    assert params["max_depth"] == "5", "max_depth not logged correctly"

    # Assert metrics
    metrics = run.data.metrics
    assert "accuracy" in metrics, "Accuracy metric not logged"
    assert metrics["accuracy"] == accuracy, "Logged accuracy does not match expected accuracy"

    # Assert model logging
    artifacts = client.list_artifacts(run_id)
    artifact_paths = [artifact.path for artifact in artifacts]
    assert "model" in artifact_paths, "Model artifact not logged"


def test_model_loading():
    # Run the MLflow experiment
    run_id, _ = train_and_log_model()

    # Load the logged model
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    # Assert model loads successfully
    assert model is not None, "Failed to load the logged model"

    # Test model predictions
    X, y = [[0] * 10], [0]  # Dummy data for testing
    prediction = model.predict(X)
    assert len(prediction) == 1, "Model prediction failed"


def test_model_accuracy(generate_data):
    X, y = generate_data
    model = LogisticRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    assert accuracy > 0.9, f"Model accuracy should be >0.9, got {accuracy}"
