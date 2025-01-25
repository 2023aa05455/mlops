from unittest import TestCase
from mlops.src.model.cancer_detaction_model import (
    train_and_evaluate,
    preprocess_data,
    load_data,
    CONFIG,
)


class Train_and_Evaluate_Model(TestCase):
    def test_train_and_evaluate(self):
        # Load the data
        X, y = load_data()

        # Preprocess data
        X_train, X_test, y_train, y_test = preprocess_data(X, y)

        for model_name, model_config in CONFIG["models"].items():
            model, accuracy, precision, f1 = train_and_evaluate(
                model_name, model_config, X_train, X_test, y_train, y_test
            )
            assert accuracy > 0.9, f"Model accuracy should be >0.9, got {accuracy}"
