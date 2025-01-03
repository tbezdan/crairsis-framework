from utils.config import (
    shap_folder,
    interactions_folder,
    models_path,
    datasets_path,
    interactions_feature_folder,
    actual_predicted_folder,
)
import joblib
import shap
import pandas as pd
import numpy as np
import os
from utils.logger import setup_logger

logger = setup_logger(__name__)

# File System Restrictions: Windows has a maximum path length limit of 260 characters


def get_data_model(
    file, filename, filter_column, filter_value, target, data_usage, datetime_col
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

    idx = selected_data["id"].reset_index(drop=True)

    selected_data = selected_data.drop(columns_to_drop, axis=1)

    x = selected_data.drop(target, axis=1)

    if hasattr(model, "feature_names_in_"):
        model_features = list(model.feature_names_in_)

        try:
            x = x[model.feature_names_in_]
        except KeyError as e:
            raise KeyError(f"Mismatch in test data columns while reordering: {e}")

        if set(x.columns) != set(model_features):
            raise ValueError(
                f"Mismatch between model features and test data columns.\n"
                f"Model features: {model_features}\n"
                f"Data columns: {set(x.columns)}"
            )

    else:

        model_features = list(x.columns)

    return model, x, idx


def shap_loc_rel_norm(
    impacts,
    expected_value,
    data,
    idx,
    file,
    folder_output,
    class_label=None,
    model=None,
    task_type=None,
):

    if task_type == "classification":

        if impacts.ndim == 1:
            impacts = impacts[:, np.newaxis]
        if impacts.shape[0] != len(data):
            raise ValueError(
                f"SHAP impacts length ({impacts.shape[0]}) does not match the test data length ({len(data)})."
            )

        impactsDf = pd.DataFrame(impacts, columns=data.columns)
        impactsDf["id"] = idx

        abs_sum = impactsDf.drop("id", axis=1).abs().sum(axis=1)
        relative = impactsDf.drop("id", axis=1).div(abs_sum, axis=0).multiply(100)

        if np.isscalar(expected_value):

            normalized = impactsDf.drop("id", axis=1) / expected_value
        elif expected_value.shape == (1,):
            normalized = impactsDf.drop("id", axis=1) / expected_value[0]
        else:

            normalized = impactsDf.drop("id", axis=1).divide(
                expected_value.reshape(-1, 1)
            )

        relative["id"] = impactsDf["id"]
        normalized["id"] = impactsDf["id"]

        # Get the model predictions
        predicted_proba = model.predict_proba(data)
        predicted_class = np.argmax(predicted_proba, axis=1)
        max_probability = np.max(predicted_proba, axis=1)

        impactsDf["predicted_class"] = predicted_class
        impactsDf["actual_class"] = class_label
        impactsDf["max_probability"] = max_probability
        impactsDf["membership"] = (predicted_class == class_label).astype(
            int
        )  # Membership: 1 if the id belongs to the class this SHAP file belongs to, else 0

        relative["predicted_class"] = predicted_class
        relative["actual_class"] = class_label
        relative["max_probability"] = max_probability
        relative["membership"] = (predicted_class == class_label).astype(
            int
        )  # Membership: 1 if the id belongs to the class this SHAP file belongs to, else 0

        normalized["predicted_class"] = predicted_class
        normalized["actual_class"] = class_label
        normalized["max_probability"] = max_probability
        normalized["membership"] = (predicted_class == class_label).astype(
            int
        )  # Membership: 1 if the id belongs to the class this SHAP file belongs to, else 0

        suffix = f"_class_{class_label}" if class_label is not None else ""

        impactsDf.to_csv(
            os.path.join(folder_output, f"{file}_impacts_local{suffix}.csv"),
            index=False,
        )
        relative.to_csv(
            os.path.join(folder_output, f"{file}_impacts_local_relative{suffix}.csv"),
            index=False,
        )
        normalized.to_csv(
            os.path.join(folder_output, f"{file}_impacts_local_normalized{suffix}.csv"),
            index=False,
        )

        pd.DataFrame({"Expected value": [expected_value]}).to_csv(
            os.path.join(folder_output, f"{file}_expected_value{suffix}.csv"),
            index=False,
        )

    else:

        impactsDf = pd.DataFrame(impacts, columns=data.columns)
        impactsDf["id"] = idx

        abs_sum = impactsDf.drop("id", axis=1).abs().sum(axis=1)
        relative = impactsDf.drop("id", axis=1).div(abs_sum, axis=0).multiply(100)
        normalized = impactsDf.drop("id", axis=1) / expected_value

        relative["id"] = impactsDf["id"]
        normalized["id"] = impactsDf["id"]

        impactsDf.to_csv(
            os.path.join(folder_output, f"{file} _impacts_local.csv"), index=False
        )
        relative.to_csv(
            os.path.join(folder_output, f"{file}_impacts_local_relative.csv"),
            index=False,
        )
        normalized.to_csv(
            os.path.join(folder_output, f"{file}_impacts_local_normalized.csv"),
            index=False,
        )

        pd.DataFrame({"Expected value": [expected_value]}).to_csv(
            os.path.join(folder_output, f"{file} - Expected value.csv"), index=False
        )


def perform_shap_analysis(
    best_models, filter_column, task_type, data_usage, datetime_col
):
    for i in range(best_models.shape[0]):
        try:
            logger.info(f"Processing model {i+1}/{best_models.shape[0]}: Start")
            target = best_models.loc[i, "target"]
            filename = best_models.loc[i, "filename"]
            filter_value = best_models.loc[i, "filter_value"]
            ml_model = best_models.loc[i, "ml_model"]
            mh_algo = best_models.loc[i, "metaheuristic"]

            file = f"filename_{filename}_filter_col_{filter_column}_filter_val_{filter_value}_target_{target}_ml_model_{ml_model}_mh_algo_{mh_algo}"

            logger.info(f"Loading data and model for {file}")
            model, data, idx = get_data_model(
                file,
                filename,
                filter_column,
                filter_value,
                target,
                data_usage,
                datetime_col,
            )

            logger.info("Initializing SHAP explainer")

            background_sample_size = min(1000, len(data))
            background_data = data.sample(background_sample_size, random_state=42)

            explainer = shap.TreeExplainer(
                model, background_data, feature_perturbation="interventional"
            )
            expected_value = explainer.expected_value

            shap_values = explainer.shap_values(data, check_additivity=False)
            interaction_values = explainer.shap_interaction_values(data)

            if task_type == "classification":

                for class_idx, class_impacts in enumerate(shap_values):

                    logger.info(f"Performing local SHAP analysis for class {class_idx}")

                    if (
                        isinstance(expected_value, np.ndarray)
                        and len(expected_value) > 1
                    ):

                        expected_val = expected_value[class_idx]
                    elif (
                        isinstance(expected_value, (list, np.ndarray))
                        and len(expected_value) == 1
                    ):

                        expected_val = expected_value[0]
                    else:
                        expected_val = expected_value

                    shap_loc_rel_norm(
                        class_impacts,
                        expected_val,
                        data,
                        idx,
                        file,
                        folder_output=shap_folder,
                        class_label=class_idx,
                        model=model,
                        task_type=task_type,
                    )

                    logger.info(
                        f"Analyzing main effects and interactions for class {class_idx}"
                    )
                    if isinstance(interaction_values, list):
                        interaction_val = interaction_values[class_idx]
                    else:
                        interaction_val = interaction_values

                    main_effect(
                        class_impacts,
                        expected_value=expected_val,
                        interaction_values=interaction_val,
                        df=data,
                        idx=idx,
                        file=file,
                        folder_output=interactions_folder,
                        class_label=class_idx,
                    )
                process_all_shap_files(
                    filename, filter_column, filter_value, target, ml_model, mh_algo
                )

            else:
                logger.info("Performing local SHAP analysis for regression")
                shap_loc_rel_norm(
                    shap_values,
                    expected_value,
                    data,
                    idx,
                    file,
                    folder_output=shap_folder,
                    task_type=task_type,
                )

                logger.info("Calculating SHAP interaction values")
                interaction_values = explainer.shap_interaction_values(data)

                logger.info("Analyzing main effects and interactions")
                main_effect(
                    shap_values,
                    expected_value,
                    interaction_values,
                    data,
                    idx,
                    file,
                    interactions_folder,
                )

            logger.info(f"Processing model {i+1}/{best_models.shape[0]}: Completed\n")
        except Exception as e:
            logger.error(f"Error processing model {i+1}: {e}", exc_info=True)


def main_effect(
    impacts,
    expected_value,
    interaction_values,
    df,
    idx,
    file,
    folder_output,
    class_label=None,
):
    interaction_details = True

    impactsDf = pd.DataFrame(impacts, columns=df.columns)
    suffix = f"_class_{class_label}" if class_label is not None else ""

    if interaction_details:

        for i in range(interaction_values.shape[2]):

            interactions = pd.DataFrame(
                interaction_values[:, :, i],
                columns=impactsDf.columns.values.tolist(),
            )

            interactions["id"] = idx

            interaction_file_path = os.path.join(
                interactions_feature_folder,
                f"{file}_interactions_feature_{df.columns[i]}{suffix}.csv",
            )

            interaction_dir = os.path.dirname(interaction_file_path)
            if not os.path.exists(interaction_dir):
                os.makedirs(interaction_dir)

            interactions.to_csv(interaction_file_path, index=False)

        main_effects_list = []
        for e, column in enumerate(impactsDf.columns):
            main_effect_series = pd.Series(interaction_values[:, e, e], name=column)
            main_effects_list.append(main_effect_series)
        main_effects = pd.concat(main_effects_list, axis=1)
        main_effects["id"] = idx
    else:
        main_effects = pd.DataFrame(interaction_values, columns=impactsDf.columns)
        main_effects["id"] = idx

    tmp = np.abs(interaction_values).sum(0)
    if tmp.ndim == 1:
        tmp = tmp.reshape(1, -1)

    tmp_shape = min(len(df.columns), tmp.shape[0])
    np.fill_diagonal(tmp[:tmp_shape, :tmp_shape], 0)

    df_interaction_matrix = pd.DataFrame(
        tmp[:tmp_shape, :tmp_shape],
        columns=df.columns[:tmp_shape],
        index=df.columns[:tmp_shape],
    )

    if not os.path.exists(folder_output):
        os.makedirs(folder_output)

    df_interaction_matrix.to_csv(
        os.path.join(folder_output, f"{file}_interactions_matrix_sum_{suffix}.csv"),
        index=True,
    )

    if np.isscalar(expected_value):
        normalized_main_effect = main_effects.drop("id", axis=1) / expected_value
    elif expected_value.shape == (1,):
        normalized_main_effect = main_effects.drop("id", axis=1) / expected_value[0]
    else:
        normalized_main_effect = main_effects.drop("id", axis=1).divide(
            expected_value.reshape(-1, 1)
        )

    normalized_main_effect["id"] = idx

    abs_sum = main_effects.drop("id", axis=1).abs().sum(axis=1)
    main_effect_relative = (
        main_effects.drop("id", axis=1).div(abs_sum, axis=0).multiply(100)
    )
    main_effect_relative["id"] = idx

    main_effects.to_csv(
        os.path.join(folder_output, f"{file}_interactions_main_effects_{suffix}.csv"),
        index=False,
    )
    main_effect_relative.to_csv(
        os.path.join(
            folder_output, f"{file}_interactions_main_effects_relative_{suffix}.csv"
        ),
        index=False,
    )
    normalized_main_effect.to_csv(
        os.path.join(
            folder_output,
            f"{file}_interactions_normalized_main_effects_{suffix}.csv",
        ),
        index=False,
    )


def get_impact_files(
    filename, filter_col, filter_val, target, ml_model, mh_algo, impact_type
):

    pattern = f"filename_{filename}_filter_col_{filter_col}_filter_val_{filter_val}_target_{target}_ml_model_{ml_model}_mh_algo_{mh_algo}_impacts_{impact_type}_class_"
    files = [f for f in os.listdir(shap_folder) if f.startswith(pattern)]
    return files


def merge_impact_files(files):

    data_frames = []

    for file in files:
        file_path = os.path.join(shap_folder, file)
        df = pd.read_csv(file_path)
        data_frames.append(df)

    merged_df = pd.concat(data_frames, ignore_index=True)
    filtered_df = merged_df[merged_df["membership"] == 1]

    return filtered_df


def process_shap_files(
    filename, filter_col, filter_val, target, ml_model, mh_algo, impact_type
):

    files = get_impact_files(
        filename, filter_col, filter_val, target, ml_model, mh_algo, impact_type
    )

    if not files:
        print(
            f"No files found for the given parameters: {filename}, {filter_col}, {filter_val}, {target}, {ml_model}, {mh_algo}, {impact_type}"
        )
        return None

    merged_filtered_df = merge_impact_files(files)

    output_filename = f"filename_{filename}_filter_col_{filter_col}_filter_val_{filter_val}_target_{target}_ml_model_{ml_model}_mh_algo_{mh_algo}_impacts_{impact_type}.csv"
    output_path = os.path.join(shap_folder, output_filename)

    merged_filtered_df.to_csv(output_path, index=False)
    print(f"Saved merged and filtered SHAP impacts ({impact_type}) to: {output_path}")

    return merged_filtered_df


def process_all_shap_files(filename, filter_col, filter_val, target, ml_model, mh_algo):

    print("Processing local SHAP impacts...")
    process_shap_files(
        filename, filter_col, filter_val, target, ml_model, mh_algo, impact_type="local"
    )

    print("Processing relative SHAP impacts...")
    process_shap_files(
        filename,
        filter_col,
        filter_val,
        target,
        ml_model,
        mh_algo,
        impact_type="local_relative",
    )

    print("Processing normalized SHAP impacts...")
    process_shap_files(
        filename,
        filter_col,
        filter_val,
        target,
        ml_model,
        mh_algo,
        impact_type="local_normalized",
    )
