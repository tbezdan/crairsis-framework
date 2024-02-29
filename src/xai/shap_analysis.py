from utils.config import (
    shap_folder,
    interactions_folder,
    models_path,
    datasets_path,
    interactions_feature_folder,
)
import joblib
import shap
import pandas as pd
import numpy as np
import os
from utils.logger import setup_logger

logger = setup_logger(__name__)


def get_data_model(file, site, covid, target):

    model_path = os.path.join(models_path, file + ".joblib")

    data_path = os.path.join(
        datasets_path, f"site_{site}_covid_{covid}_target_{target}.csv"
    )

    X = pd.read_csv(data_path)
    model = joblib.load(model_path)
    test_data = X[X["Usage"] == "Test"]

    idx = test_data["id"].reset_index(drop=True)
    test_data = test_data.drop(["Usage", "Datetime", "id"], axis=1)

    x = test_data.drop(target, axis=1)

    return model, x, idx


def shap_loc_rel_norm(impacts, expected_value, test_data, idx, file, folder_output):

    impactsDf = pd.DataFrame(impacts, columns=test_data.columns)
    impactsDf["id"] = idx

    abs_sum = impactsDf.drop("id", axis=1).abs().sum(axis=1)
    relative = impactsDf.drop("id", axis=1).div(abs_sum, axis=0).multiply(100)
    normalized = impactsDf.drop("id", axis=1) / expected_value

    relative["id"] = impactsDf["id"]
    normalized["id"] = impactsDf["id"]

    impactsDf.to_csv(
        os.path.join(folder_output, f"{file} - Impacts - Local.csv"), index=False
    )
    relative.to_csv(
        os.path.join(folder_output, f"{file} - Impacts - Local - Relative.csv"),
        index=False,
    )
    normalized.to_csv(
        os.path.join(folder_output, f"{file} - Impacts - Local - Normalized.csv"),
        index=False,
    )

    pd.DataFrame({"Expected value": [expected_value]}).to_csv(
        os.path.join(folder_output, f"{file} - Expected value.csv"), index=False
    )


def main_effect(
    impacts,
    expected_value,
    interaction_values,
    df,
    idx,
    file,
    folder_output,
):
    interaction_details = True

    # Create DataFrame for impacts
    impactsDf = pd.DataFrame(impacts, columns=df.columns)

    # Handle interactions
    if interaction_details:
        for i in range(interaction_values.shape[2]):
            interactions = pd.DataFrame(
                interaction_values[:, :, i],
                columns=impactsDf.columns.values.tolist(),
            )
            interactions["id"] = idx  # Append 'id' column to avoid fragmentation
            interactions.to_csv(
                os.path.join(
                    interactions_feature_folder,
                    f"{file} - Interactions - Feature - {df.columns[i]}.csv",
                ),
                index=False,
            )

    # Calculate main effects without fragmentation
    main_effects_list = []
    for e, column in enumerate(impactsDf.columns):
        main_effect_series = pd.Series(interaction_values[:, e, e], name=column)
        main_effects_list.append(main_effect_series)
    main_effects = pd.concat(main_effects_list, axis=1)
    main_effects["id"] = idx  # Append 'id' column here

    # Interaction matrices calculation
    tmp = np.abs(interaction_values).sum(0)[:-1, :][:, :-1]
    np.fill_diagonal(tmp, 0)  # Zero diagonal without causing fragmentation

    # Calculate and save interaction matrices
    df_interaction_matrix = pd.DataFrame(
        tmp, columns=df.columns[:-1], index=df.columns[:-1]
    )
    df_interaction_matrix.to_csv(
        os.path.join(folder_output, f"{file} - Interactions - Matrix - Sum.csv"),
        index=True,
    )

    # Normalization and relative calculations
    normalized_main_effect = main_effects.drop("id", axis=1) / expected_value
    normalized_main_effect["id"] = idx  # Re-append 'id' column

    abs_sum = main_effects.drop("id", axis=1).abs().sum(axis=1)
    main_effect_relative = (
        main_effects.drop("id", axis=1).div(abs_sum, axis=0).multiply(100)
    )
    main_effect_relative["id"] = idx  # Re-append 'id' column

    # Save results to CSV
    main_effects.to_csv(
        os.path.join(folder_output, f"{file} - Interactions - Main effects.csv"),
        index=False,
    )
    main_effect_relative.to_csv(
        os.path.join(
            folder_output, f"{file} - Interactions - Main effects Relative.csv"
        ),
        index=False,
    )
    normalized_main_effect.to_csv(
        os.path.join(
            folder_output, f"{file} - Interactions - Normalized Main effects.csv"
        ),
        index=False,
    )


def perform_shap_analysis(best_models):
    for i in range(best_models.shape[0]):
        try:
            logger.info(f"Processing model {i+1}/{best_models.shape[0]}: Start")
            target = best_models.loc[i, "target"]
            site = best_models.loc[i, "site"]
            covid = best_models.loc[i, "covid"]
            ml_model = best_models.loc[i, "ml_model"]
            mh_algo = best_models.loc[i, "mh_algo"]
            file = f"site_{site}_covid_{covid}_target_{target}_ml_model_{ml_model}_mh_algo_{mh_algo}"

            logger.info(f"Loading data and model for {file}")
            model, test_data, idx = get_data_model(file, site, covid, target)

            logger.info("Initializing SHAP explainer")
            explainer = shap.TreeExplainer(model)
            expected_value = explainer.expected_value
            if isinstance(expected_value, np.ndarray) and expected_value.size == 1:
                expected_value = expected_value.item()

            logger.info("Calculating SHAP values")
            impacts = explainer.shap_values(test_data)

            logger.info("Performing local SHAP analysis")
            shap_loc_rel_norm(
                impacts,
                expected_value,
                test_data,
                idx,
                file,
                folder_output=shap_folder,
            )

            logger.info("Calculating SHAP interaction values")
            interaction_values = explainer.shap_interaction_values(test_data)

            logger.info("Analyzing main effects and interactions")
            main_effect(
                impacts,
                expected_value,
                interaction_values,
                test_data,
                idx,
                file,
                interactions_folder,
            )

            logger.info(f"Processing model {i+1}/{best_models.shape[0]}: Completed")
        except Exception as e:
            logger.error(f"Error processing model {i+1}: {e}", exc_info=True)
