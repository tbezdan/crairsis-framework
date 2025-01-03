from utils.config import sage_folder, datasets_path, models_path
import pandas as pd
import numpy as np
import joblib
import sage
import os


from utils.logger import setup_logger

logger = setup_logger(__name__)


def process_data(
    target, file, filename, filter_column, filter_value, data_usage, datetime_col
):
    model_path = os.path.join(models_path, file + ".joblib")
    data_path = os.path.join(
        datasets_path,
        f"filename_{filename}_filter_col_{filter_column}_filter_val_{filter_value}_target_{target}.csv",
    )

    X = pd.read_csv(data_path)
    model = joblib.load(model_path)

    if data_usage == "train_and_test":
        selected_data = X
    else:
        selected_data = X[X["usage"] == "test"]

    columns_to_drop = ["usage", "id"]
    if datetime_col:
        columns_to_drop.append(datetime_col)

    selected_data = selected_data.drop(columns_to_drop, axis=1)

    y = selected_data[target].values
    x = selected_data.drop(target, axis=1)

    if hasattr(model, "feature_names_in_"):
        model_features = model.feature_names_in_

    elif hasattr(model, "feature_name_"):
        model_features = model.feature_name_

    else:
        model_features = x.columns

    if list(x.columns) != list(model_features):
        x = x[model_features]

    feature_names = x.columns.tolist()

    return model, x, y, feature_names


def calculate_and_save_sage(
    model, x, y, feature_names, file_name, task_type, threshold
):

    x = x.values
    imputer = sage.MarginalImputer(model, x[: int(x.shape[0] * 0.05)])

    if task_type == "regression":
        estimator = sage.PermutationEstimator(imputer, "mse", n_jobs=-1)
    elif task_type == "classification":
        estimator = sage.PermutationEstimator(imputer, "cross entropy", n_jobs=-1)

    sage_values = estimator(x, y)
    sensitivity = estimator(x)

    sage_global_values_df = pd.DataFrame(
        {
            "feature": feature_names,
            "global_impact": sage_values.values,
            "global_impact_std": sage_values.std,
            "sensitivity": sensitivity.values,
            "sensitivity_std": sensitivity.std,
            "absolute_global_impact": np.abs(sage_values.values),
            "absolute_global_sensitivity": np.abs(sensitivity.values),
            "relative_global_impact": (
                np.abs(sage_values.values) / sum(np.abs(sage_values.values))
            )
            * 100,
            "relative_global_sensitivity": (
                np.abs(sensitivity.values) / sum(np.abs(sensitivity.values))
            )
            * 100,
        }
    )

    sage_global_values_df.sort_values(
        by="relative_global_impact", ascending=False, inplace=True
    )
    sage_global_values_df["cumsum_global_impact"] = sage_global_values_df[
        "relative_global_impact"
    ].cumsum()

    sage_global_values_df["important"] = sage_global_values_df[
        "cumsum_global_impact"
    ].apply(lambda x: 1 if x <= threshold else 0)

    if sage_global_values_df["important"].sum() == 0:
        sage_global_values_df.loc[sage_global_values_df.index[0], "important"] = 1

    output_path = os.path.join(sage_folder, f"sage_global_impact_{file_name}.csv")
    sage_global_values_df.to_csv(output_path, index=False)


def perform_sage_analysis(
    best_models, filter_column, task_type, data_usage, datetime_col, threshold
):
    logger.info("Starting SAGE analysis...")

    for i in range(best_models.shape[0]):
        target = best_models.loc[i, "target"]
        filename = best_models.loc[i, "filename"]
        filter_value = best_models.loc[i, "filter_value"]
        ml_model = best_models.loc[i, "ml_model"]
        mh_algo = best_models.loc[i, "metaheuristic"]

        logger.info(
            f"Analyzing model: {ml_model} with mh_algo: {mh_algo} for target: {target}, filename: {filename}, filter value: {filter_value}"
        )

        file = f"filename_{filename}_filter_col_{filter_column}_filter_val_{filter_value}_target_{target}_ml_model_{ml_model}_mh_algo_{mh_algo}"
        model, x, y, feature_names = process_data(
            target,
            file,
            filename,
            filter_column,
            filter_value,
            data_usage,
            datetime_col,
        )

        calculate_and_save_sage(model, x, y, feature_names, file, task_type, threshold)
        logger.info("SAGE analysis completed.")
