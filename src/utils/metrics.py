from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    max_error,
    mean_absolute_percentage_error,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
)
from sklearn.utils.multiclass import type_of_target


def calculate_regression_metrics(y_true, y_pred):

    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "explained_variance": explained_variance_score(y_true, y_pred),
        "max_error": max_error(y_true, y_pred),
    }
    return metrics


def calculate_classification_metrics(y_true, y_pred, y_proba=None, multi_class=None):
    """
    multi_class='ovr' (one-vs-rest): method fits one classifier per class.
    Each classifier is trained to distinguish a given class from all the other classes combined.
    This method is typically used when classes are imbalanced or when you have a large number of classes.
    It directly compares one class against all others.

    multi_class='ovo' (One-vs-One): OVO splits a multiclass problem into multiple binary classification problems, then takes the average.
    Preferred when classes are balanced
    """

    target_type = type_of_target(y_true)

    if target_type == "binary":
        average_method = "binary"
    else:
        average_method = "weighted"

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average=average_method),
        "precision": precision_score(y_true, y_pred, average=average_method),
        "recall": recall_score(y_true, y_pred, average=average_method),
    }
    if y_proba is not None:
        if multi_class:
            metrics.update(
                {
                    "roc_auc": roc_auc_score(y_true, y_proba, multi_class=multi_class),
                    "log_loss": log_loss(y_true, y_proba),
                }
            )
        else:
            metrics.update(
                {
                    "roc_auc": roc_auc_score(y_true, y_proba),
                    "log_loss": log_loss(y_true, y_proba),
                }
            )
    return metrics


def log_metrics(metrics, logger):

    summary_metrics = [
        "accuracy",
        "f1_score",
        "precision",
        "recall",
        "roc_auc",
        "log_loss",
    ]
    for name, value in metrics.items():
        if name in summary_metrics or "macro avg" in name or "weighted avg" in name:
            logger.info(f"{name}: {value}")
        else:
            logger.debug(f"{name}: {value}")


def confusion_matrix_report(y_true, y_pred):

    return confusion_matrix(y_true, y_pred)
