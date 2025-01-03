import pandas as pd
import os


def format_best_models(output_path, task_type):
    if task_type == "regression":
        metric = "r2"
    else:
        metric = "f1_score"

    optimized_df = pd.read_csv(
        os.path.join(output_path, "results", "optimized_models_results_cv.csv")
    )

    if (
        optimized_df["filter_value"].isnull().any()
        or optimized_df["filter_column"].isnull().any()
    ):
        optimized_df["filter_value"].fillna("None", inplace=True)
        optimized_df["filter_column"].fillna("None", inplace=True)

    best_models_df = optimized_df.loc[
        optimized_df.groupby(["target", "filter_value", "filename"])[metric].idxmax()
    ]

    selected_columns = [
        "ml_model",
        metric,
        "filename",
        "metaheuristic",
        "target",
        "filter_value",
        "filter_column",
    ]

    best_models_structured_df = best_models_df.loc[:, selected_columns].copy()
    # Rename columns to match the desired output
    best_models_structured_df.rename(
        columns={
            metric: "value",
        },
        inplace=True,
    )

    best_models_structured_df["name"] = metric

    structured_df = best_models_structured_df[
        [
            "name",
            "value",
            "filename",
            "filter_value",
            "filter_column",
            "target",
            "ml_model",
            "metaheuristic",
        ]
    ]

    # Save the structured best models to CSV
    structured_df.to_csv(os.path.join(output_path, "best_models.csv"), index=False)


def format_detailed_metrics(output_path, task_type):
    optimized_df = pd.read_csv(
        os.path.join(output_path, "results", "optimized_models_results.csv")
    )

    detailed_metrics = []
    if task_type == "regression":
        metrics_list = [
            "mae",
            "mse",
            "rmse",
            "mape",
            "explained_variance",
            "max_error",
            "r2",
        ]
    else:
        metrics_list = [
            "accuracy",
            "f1_score",
            "precision",
            "recall",
            "roc_auc",
            "log_loss",
        ]

    for index, row in optimized_df.iterrows():
        for metric_name in metrics_list:
            detailed_metrics.append(
                {
                    "name": metric_name,
                    "value": row[metric_name],
                    "filename": row["filename"],
                    "filter_column": row["filter_column"],
                    "filter_value": row["filter_value"],
                    "target": row["target"],
                    "ml_model": row["ml_model"],
                    "metaheuristic": row["metaheuristic"],
                }
            )

    detailed_metrics_df = pd.DataFrame(detailed_metrics)

    detailed_metrics_df.to_csv(
        os.path.join(output_path, "best_models_and_metrics.csv"),
        index=False,
    )
