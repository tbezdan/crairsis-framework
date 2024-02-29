from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    max_error,
    mean_absolute_percentage_error,
)


def calculate_metrics(y_true, y_pred):
    """Calculate various evaluation metrics."""
    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "Explained Variance": explained_variance_score(y_true, y_pred),
        "Max Error": max_error(y_true, y_pred),
    }
    return metrics


def log_metrics(metrics, logger):
    """Log the calculated metrics using the provided logger."""
    for name, value in metrics.items():
        logger.info(f"{name}: {value}")
