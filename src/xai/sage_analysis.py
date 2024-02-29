from utils.config import sage_folder, datasets_path, models_path
import pandas as pd
import numpy as np
import joblib
import sage
import os
import warnings

warnings.filterwarnings("ignore")
from utils.logger import setup_logger

logger = setup_logger(__name__)


def process_data(target, file, site, covid):

    model_path = os.path.join(models_path, file + ".joblib")

    data_path = os.path.join(
        datasets_path, f"site_{site}_covid_{covid}_target_{target}.csv"
    )

    X = pd.read_csv(data_path)
    model = joblib.load(model_path)

    test_data = X[X["Usage"] == "Test"]

    test_data = test_data.drop(["Usage", "Datetime", "id"], axis=1)

    y = test_data[target].values
    x = test_data.drop(target, axis=1).values
    feature_names = test_data.drop(target, axis=1).columns.tolist()

    return model, x, y, feature_names


def calculate_and_save_sage(model, x, y, feature_names, file_name):

    imputer = sage.MarginalImputer(model, x[: int(x.shape[0] * 0.05)])
    estimator = sage.PermutationEstimator(imputer, "mse")
    sage_values = estimator(x, y)
    sensitivity = estimator(x)

    sage_global_values_df = pd.DataFrame(
        {
            "feature": feature_names,
            "global_impact": sage_values.values,
            "global_impact_std": sage_values.std,
            "Sensitivity": sensitivity.values,
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
    threshold = 90

    sage_global_values_df["important"] = sage_global_values_df[
        "cumsum_global_impact"
    ].apply(lambda x: 1 if x <= threshold else 0)
    output_path = os.path.join(sage_folder, f"sage_global_impact{file_name}.csv")
    sage_global_values_df.to_csv(output_path, index=False)


def perform_sage_analysis(best_models):
    logger.info("Starting SAGE analysis...")

    for i in range(best_models.shape[0]):
        target = best_models.loc[i, "target"]
        site = best_models.loc[i, "site"]
        covid = best_models.loc[i, "covid"]
        ml_model = best_models.loc[i, "ml_model"]
        mh_algo = best_models.loc[i, "mh_algo"]

        logger.info(
            f"Analyzing model: {ml_model} with mh_algo: {mh_algo} for target: {target}, site: {site}, covid: {covid}"
        )

        file = f"site_{site}_covid_{covid}_target_{target}_ml_model_{ml_model}_mh_algo_{mh_algo}"
        model, x, y, feature_names = process_data(target, file, site, covid)

        calculate_and_save_sage(model, x, y, feature_names, file)
        logger.info("SAGE analysis completed.")
