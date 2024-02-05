import pytest
import pandas as pd
import numpy as np 
from ml.data import process_data
from ml.model import compute_model_metrics, train_model
from sklearn.ensemble import RandomForestClassifier



def test_type():
    """
    Test if any ML functions return the expected type of result.
    """
    # Test dataset
    test_data = pd.DataFrame(
        {
            "column_1": [3, 6, 8, 2, 9],
            "column_2": [5, 1, 7, 1, 9],
            "label": [1, 1, 0, 1, 1]
        }
    )

    # Process data
    X_train, y_train, _, _ = process_data(
        test_data,
        categorical_features= ["column_2"],
        label= "label",
        training= True
    )

    # Train model
    rf = train_model(X_train, y_train)

    # Check type of returned model
    returned_model = RandomForestClassifier
    assert isinstance(rf, returned_model)


def test_algorithm():
    """
    Test if the ML model uses the expected algorithm.
    """
    # Test dataset
    test_data = pd.DataFrame(
        {
            "column_1": [3, 6, 8, 2, 9],
            "column_2": [5, 1, 7, 1, 9],
            "label": [1, 1, 0, 1, 1]
        }
    )

    # Process data
    X_train, y_train, _, _ = process_data(
        test_data,
        categorical_features= ["column_2"],
        label= "label",
        training= True
    )

    # Train model
    model = train_model(X_train, y_train)

    # Check if the ML model uses the expected algorithm
    assert isinstance(model, RandomForestClassifier)


def test_value():
    """
    Test if the computing metrics functions return the expected value.
    """
    # Test dataset with controlled values
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 1])

    # Compute the metrics
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    # Check if the computing metrics functions return the expected value
    expected_precision = 0.6667
    expected_recall = 1.0
    expected_fbeta = 0.8

    assert round(precision, 4) == expected_precision
    assert round(recall, 4) == expected_recall
    assert round(fbeta, 4) == expected_fbeta
