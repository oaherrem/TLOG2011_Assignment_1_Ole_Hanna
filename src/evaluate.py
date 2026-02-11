from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def evaluate_classification(model, X_test, y_test):
    """
    Evaluates a binary classification model for late delivery prediction.

    Parameters:
    - model: Trained classification model
    - X_test: Feature matrix for testing
    - y_test: True binary labels (0 = on time, 1 = late)

    Returns:
    - Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
    }

