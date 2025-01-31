import nshap

import joblib
import pandas as pd
import numpy as np
import os

from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from utils.logger import setup_logger

logger = setup_logger(__name__)


def get_data_model(
    sage_folder,
    shap_folder,
    nshap_folder,
    models_path,
    datasets_path,
    file,
    filename,
    filter_column,
    filter_value,
    target,
    datetime_col,
):

    sage_file = os.path.join(sage_folder, f"sage_global_impact_{file}.csv")

    if os.path.exists(sage_file):
        sage_data = pd.read_csv(sage_file)
    else:
        shap_local_impacts_file_path = os.path.join(
            shap_folder, f"{file}_impacts_local.csv"
        )
        shap_local_impacts = pd.read_csv(shap_local_impacts_file_path)
        shap_values = shap_local_impacts.drop(columns=["id"])
        sage_importance = shap_values.abs().mean()
        sage_ranking = sage_importance.sort_values(ascending=False)
        sage_df = sage_ranking.reset_index()
        sage_df.columns = ["feature", "importance"]
        sage_data = sage_df

    num_of_features = 3
    available_features = len(sage_data.feature)
    num_of_features = min(num_of_features, available_features - 3)
    feature_names = list(sage_data.feature[:num_of_features])
    feature_names_file = os.path.join(nshap_folder, file + "_feature_names.csv")
    with open(feature_names_file, "w") as f:
        f.write(",".join(feature_names))

    model_path = os.path.join(models_path, file + ".joblib")
    model = joblib.load(model_path)

    data_path = os.path.join(
        datasets_path,
        f"filename_{filename}_filter_col_{filter_column}_filter_val_{filter_value}_target_{target}.csv",
    )

    X = pd.read_csv(data_path)

    columns_to_drop = ["usage", "id", target]
    if datetime_col:
        columns_to_drop.append(datetime_col)
    X_temp = X.drop(columns_to_drop, axis=1)

    if hasattr(model, "feature_names_in_"):
        model_features = list(model.feature_names_in_)

        try:
            X_temp = X_temp[model.feature_names_in_]
        except KeyError as e:
            raise KeyError(f"Mismatch in data columns while reordering: {e}")

        if set(X_temp.columns) != set(model_features):
            raise ValueError(
                f"Mismatch between model features and data columns.\n"
                f"Model features: {model_features}\n"
                f"Data columns: {set(X_temp.columns)}"
            )

    else:

        model_features = list(X.columns)

    features = feature_names + columns_to_drop

    X = X[features]

    X_train = X[X["usage"] == "train"]
    X_test = X[X["usage"] == "test"]

    y_train = X[X["usage"] == "train"][target]
    y_test = X[X["usage"] == "test"][target]

    idx_train = X_train["id"].reset_index(drop=True)
    X_train = X_train.drop(columns_to_drop, axis=1)

    idx_test = X_test["id"].reset_index(drop=True)
    X_test = X_test.drop(columns_to_drop, axis=1)

    return model, X_train, X_test, y_train, y_test, idx_train, idx_test


def compute_n_shapley(vfunc, X_test, idx_test, instance_id):
    n_shapley_values = nshap.n_shapley_values(X_test.values[instance_id, :], vfunc)

    df = pd.DataFrame.from_dict(n_shapley_values, orient="index", columns=["value"])
    df["interaction"] = df.index
    df["order"] = df["interaction"].apply(len)
    df["id"] = idx_test[instance_id]

    return df


def compute_shapley_taylor(vfunc, X_test, idx_test, instance_id):

    shapley_taylor = nshap.shapley_taylor(X_test.values[instance_id, :], vfunc)

    df = pd.DataFrame.from_dict(shapley_taylor, orient="index", columns=["value"])
    df["interaction"] = df.index
    df["order"] = df["interaction"].apply(len)
    df["id"] = idx_test[instance_id]

    return df


def compute_faith_shap(vfunc, X_test, idx_test, instance_id):

    faith_shap = nshap.faith_shap(X_test.values[instance_id, :], vfunc)

    df = pd.DataFrame.from_dict(faith_shap, orient="index", columns=["value"])
    df["interaction"] = df.index
    df["order"] = df["interaction"].apply(len)
    df["id"] = idx_test[instance_id]

    return df


def compute_shapley_interaction_index(vfunc, X_test, idx_test, instance_id):

    shapley_interaction_index = nshap.shapley_interaction_index(
        X_test.values[instance_id, :], vfunc
    )

    df = pd.DataFrame.from_dict(
        shapley_interaction_index, orient="index", columns=["value"]
    )
    df["interaction"] = df.index
    df["order"] = df["interaction"].apply(len)
    df["id"] = idx_test[instance_id]

    return df


def compute_banzhaf_interaction_index(vfunc, X_test, idx_test, instance_id):

    banzhaf_interaction_index = nshap.banzhaf_interaction_index(
        X_test.values[instance_id, :], vfunc
    )

    df = pd.DataFrame.from_dict(
        banzhaf_interaction_index, orient="index", columns=["value"]
    )
    df["interaction"] = df.index
    df["order"] = df["interaction"].apply(len)
    df["id"] = idx_test[instance_id]

    return df


def compute_faith_banzhaf(vfunc, X_test, idx_test, instance_id):

    faith_banzhaf = nshap.faith_banzhaf(X_test.values[instance_id, :], vfunc)

    df = pd.DataFrame.from_dict(faith_banzhaf, orient="index", columns=["value"])
    df["interaction"] = df.index
    df["order"] = df["interaction"].apply(len)
    df["id"] = idx_test[instance_id]

    return df


def compute_moebius_transform(vfunc, X_test, idx_test, instance_id):

    moebius_transform = nshap.moebius_transform(X_test.values[instance_id, :], vfunc)

    df = pd.DataFrame.from_dict(moebius_transform, orient="index", columns=["value"])
    df["interaction"] = df.index
    df["order"] = df["interaction"].apply(len)
    df["id"] = idx_test[instance_id]

    return df


def reg_calculate_and_save_nshap(
    nshap_folder, file, new_model, X_train, X_test, idx_test
):

    num_samples = int(min(X_train.shape[0] * 0.1, 1000))

    # https://github#.com/tml-tuebingen/nshap/blob/d2feb1328911c66dd9027e692a5d3d02c2c919ad/notebooks/replicate-paper/compute-vfunc.ipynb#L115
    # with continuous outputs and does not include a target parameter
    vfunc = nshap.vfunc.interventional_shap(
        new_model.predict, X_train.values, random_state=0, num_samples=num_samples
    )

    logger.info("Computing n shapley values")
    with tqdm_joblib(
        tqdm(desc="Computing n shapley values", total=X_test.shape[0])
    ) as progress_bar:
        results_n_shapley = Parallel(n_jobs=-1, batch_size="auto")(
            delayed(compute_n_shapley)(vfunc, X_test, idx_test, i)
            for i in range(X_test.shape[0])
        )
    n_shapley_df = pd.concat(results_n_shapley, ignore_index=True)
    n_shapley_df.to_csv(
        os.path.join(nshap_folder, file + "_nshapley_values.csv"), index=False
    )

    logger.info("Computing shapley taylor values")
    with tqdm_joblib(
        tqdm(desc="Computing shapley taylor values", total=X_test.shape[0])
    ) as progress_bar:
        results_shapley_taylor = Parallel(n_jobs=-1, batch_size="auto")(
            delayed(compute_shapley_taylor)(vfunc, X_test, idx_test, i)
            for i in range(X_test.shape[0])
        )
    shapley_taylor_df = pd.concat(results_shapley_taylor, ignore_index=True)
    shapley_taylor_df.to_csv(
        os.path.join(nshap_folder, file + "_shapley_taylor_values.csv"), index=False
    )

    logger.info("Computing faith shap values")
    with tqdm_joblib(
        desc="Computing faith shap values", total=X_test.shape[0]
    ) as progress_bar:
        results_faith_shap = Parallel(n_jobs=-1, batch_size="auto")(
            delayed(compute_faith_shap)(vfunc, X_test, idx_test, i)
            for i in range(X_test.shape[0])
        )
    faith_shap_df = pd.concat(results_faith_shap, ignore_index=True)
    faith_shap_df.to_csv(
        os.path.join(nshap_folder, file + "_faith_shap_values.csv"), index=False
    )

    logger.info("Computing shapley interaction index values")
    with tqdm_joblib(
        desc="Computing shapley interaction index values", total=X_test.shape[0]
    ) as progress_bar:
        results_shapley_interaction_index = Parallel(n_jobs=-1, batch_size="auto")(
            delayed(compute_shapley_interaction_index)(vfunc, X_test, idx_test, i)
            for i in range(X_test.shape[0])
        )
    results_shapley_interaction_index_df = pd.concat(
        results_shapley_interaction_index, ignore_index=True
    )
    results_shapley_interaction_index_df.to_csv(
        os.path.join(nshap_folder, file + "_shapley_interaction_index_values.csv"),
        index=False,
    )

    logger.info("Computing banzhaf interaction index values")
    with tqdm_joblib(
        desc="Computing banzhaf interaction index values", total=X_test.shape[0]
    ) as progress_bar:
        results_banzhaf_interaction_index = Parallel(n_jobs=-1, batch_size="auto")(
            delayed(compute_banzhaf_interaction_index)(vfunc, X_test, idx_test, i)
            for i in range(X_test.shape[0])
        )
    results_banzhaf_interaction_index_df = pd.concat(
        results_banzhaf_interaction_index, ignore_index=True
    )
    results_banzhaf_interaction_index_df.to_csv(
        os.path.join(nshap_folder, file + "_banzhaf_interaction_index_values.csv"),
        index=False,
    )

    logger.info("Computing faith banzhaf values")
    with tqdm_joblib(
        desc="Computing faith banzhaf values", total=X_test.shape[0]
    ) as progress_bar:
        results_faith_banzhaf = Parallel(n_jobs=-1, batch_size="auto")(
            delayed(compute_faith_banzhaf)(vfunc, X_test, idx_test, i)
            for i in range(X_test.shape[0])
        )
    results_faith_banzhaf_df = pd.concat(results_faith_banzhaf, ignore_index=True)
    results_faith_banzhaf_df.to_csv(
        os.path.join(nshap_folder, file + "_faith_banzhaf_values.csv"), index=False
    )

    logger.info("Computing moebius transform values")
    with tqdm_joblib(
        desc="Computing moebius transform values", total=X_test.shape[0]
    ) as progress_bar:
        results_moebius_transform = Parallel(n_jobs=-1, batch_size="auto")(
            delayed(compute_moebius_transform)(vfunc, X_test, idx_test, i)
            for i in range(X_test.shape[0])
        )
    results_moebius_transform_df = pd.concat(
        results_moebius_transform, ignore_index=True
    )
    results_moebius_transform_df.to_csv(
        os.path.join(nshap_folder, file + "_moebius_transform_values.csv"), index=False
    )


def cls_calculate_and_save_nshap(
    nshap_folder, file, new_model, X_train, X_test, idx_test, y_train
):

    num_samples = int(min(X_train.shape[0] * 0.1, 1000))
    unique_classes = sorted(set(y_train))

    class_vfuncs = {
        cls: nshap.vfunc.interventional_shap(
            new_model.predict_proba,
            X_train.values,
            target=cls,
            random_state=0,
            num_samples=num_samples,
        )
        for cls in unique_classes
    }

    predicted_classes = new_model.predict(X_test)

    def create_vfunc(instance_id):
        predicted_class = int(predicted_classes[instance_id])
        #  logger.info(f"Instance {instance_id}: Predicted class {predicted_class}")

        vfunc = class_vfuncs[predicted_class]

        return vfunc

    logger.info("Computing n shapley values")

    with tqdm_joblib(
        tqdm(desc="Computing n shapley values", total=X_test.shape[0])
    ) as progress_bar:
        results_n_shapley = Parallel(n_jobs=-1, batch_size="auto")(
            delayed(compute_n_shapley)(create_vfunc(i), X_test, idx_test, i)
            for i in range(X_test.shape[0])
        )
    n_shapley_df = pd.concat(results_n_shapley, ignore_index=True)
    n_shapley_df.to_csv(
        os.path.join(nshap_folder, file + "_nshapley_values.csv"), index=False
    )

    logger.info("Computing shapley taylor values")
    with tqdm_joblib(
        tqdm(desc="Computing shapley taylor values", total=X_test.shape[0])
    ) as progress_bar:
        results_shapley_taylor = Parallel(n_jobs=-1, batch_size="auto")(
            delayed(compute_shapley_taylor)(create_vfunc(i), X_test, idx_test, i)
            for i in range(X_test.shape[0])
        )
    shapley_taylor_df = pd.concat(results_shapley_taylor, ignore_index=True)
    shapley_taylor_df.to_csv(
        os.path.join(nshap_folder, file + "_shapley_taylor_values.csv"), index=False
    )

    logger.info("Computing faith shap values")
    with tqdm_joblib(
        desc="Computing faith shap values", total=X_test.shape[0]
    ) as progress_bar:
        results_faith_shap = Parallel(n_jobs=-1, batch_size="auto")(
            delayed(compute_faith_shap)(create_vfunc(i), X_test, idx_test, i)
            for i in range(X_test.shape[0])
        )
    faith_shap_df = pd.concat(results_faith_shap, ignore_index=True)
    faith_shap_df.to_csv(
        os.path.join(nshap_folder, file + "_faith_shap_values.csv"), index=False
    )

    logger.info("Computing shapley interaction index values")
    with tqdm_joblib(
        desc="Computing shapley interaction index values", total=X_test.shape[0]
    ) as progress_bar:
        results_shapley_interaction_index = Parallel(n_jobs=-1, batch_size="auto")(
            delayed(compute_shapley_interaction_index)(
                create_vfunc(i), X_test, idx_test, i
            )
            for i in range(X_test.shape[0])
        )
    results_shapley_interaction_index_df = pd.concat(
        results_shapley_interaction_index, ignore_index=True
    )
    results_shapley_interaction_index_df.to_csv(
        os.path.join(nshap_folder, file + "_shapley_interaction_index_values.csv"),
        index=False,
    )

    logger.info("Computing banzhaf interaction index values")
    with tqdm_joblib(
        desc="Computing banzhaf interaction index values", total=X_test.shape[0]
    ) as progress_bar:
        results_banzhaf_interaction_index = Parallel(n_jobs=-1, batch_size="auto")(
            delayed(compute_banzhaf_interaction_index)(
                create_vfunc(i), X_test, idx_test, i
            )
            for i in range(X_test.shape[0])
        )
    results_banzhaf_interaction_index_df = pd.concat(
        results_banzhaf_interaction_index, ignore_index=True
    )
    results_banzhaf_interaction_index_df.to_csv(
        os.path.join(nshap_folder, file + "_banzhaf_interaction_index_values.csv"),
        index=False,
    )

    logger.info("Computing faith banzhaf values")
    with tqdm_joblib(
        desc="Computing faith banzhaf values", total=X_test.shape[0]
    ) as progress_bar:
        results_faith_banzhaf = Parallel(n_jobs=-1, batch_size="auto")(
            delayed(compute_faith_banzhaf)(create_vfunc(i), X_test, idx_test, i)
            for i in range(X_test.shape[0])
        )
    results_faith_banzhaf_df = pd.concat(results_faith_banzhaf, ignore_index=True)
    results_faith_banzhaf_df.to_csv(
        os.path.join(nshap_folder, file + "_faith_banzhaf_values.csv"), index=False
    )

    logger.info("Computing moebius transform values")
    with tqdm_joblib(
        desc="Computing moebius transform values", total=X_test.shape[0]
    ) as progress_bar:
        results_moebius_transform = Parallel(n_jobs=-1, batch_size="auto")(
            delayed(compute_moebius_transform)(create_vfunc(i), X_test, idx_test, i)
            for i in range(X_test.shape[0])
        )
    results_moebius_transform_df = pd.concat(
        results_moebius_transform, ignore_index=True
    )
    results_moebius_transform_df.to_csv(
        os.path.join(nshap_folder, file + "_moebius_transform_values.csv"), index=False
    )


def perform_nshap_analysis(
    nshap_folder,
    models_path,
    datasets_path,
    sage_folder,
    shap_folder,
    top_sage_features_models_path,
    best_models,
    filter_column,
    task_type,
    datetime_col,
    model_registry,
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
            model, X_train, X_test, y_train, y_test, idx_train, idx_test = (
                get_data_model(
                    sage_folder,
                    shap_folder,
                    nshap_folder,
                    models_path,
                    datasets_path,
                    file,
                    filename,
                    filter_column,
                    filter_value,
                    target,
                    datetime_col,
                )
            )

            logger.info("Training new model with the most important features from SAGE")
            model_name = model_registry[ml_model]
            hyperparameters = model.get_params()
            new_model = model_name(**hyperparameters)
            new_model.train(X_train, y_train)
            model_filename = f"filename_{filename}_filter_col_{filter_column}_filter_val_{filter_value}_target_{target}_ml_model_{ml_model}_mh_algo_{mh_algo}.joblib"
            new_model.save(os.path.join(top_sage_features_models_path, model_filename))

            logger.info("Saving predictions for the new model")
            if task_type == "regression":
                predictions = new_model.predict(X_test)
                predictions_df = pd.DataFrame(
                    {"id": idx_test, "actual": y_test.values, "prediction": predictions}
                )
            elif task_type == "classification":
                predicted_classes = new_model.predict(X_test)
                predicted_probabilities = new_model.predict_proba(X_test)
                predictions_df = pd.DataFrame(
                    {
                        "id": idx_test,
                        "actual": y_test.values,
                        "predicted_class": predicted_classes,
                        **{
                            f"probability_class_{i}": predicted_probabilities[:, i]
                            for i in range(predicted_probabilities.shape[1])
                        },
                    }
                )

            predictions_file = os.path.join(nshap_folder, file + "_predictions.csv")
            predictions_df.to_csv(predictions_file, index=False)
            logger.info(f"Predictions saved to {predictions_file}")

            logger.info("Initializing nSHAP")

            if task_type == "regression":
                reg_calculate_and_save_nshap(
                    nshap_folder, file, new_model, X_train, X_test, idx_test
                )
            elif task_type == "classification":
                cls_calculate_and_save_nshap(
                    nshap_folder, file, new_model, X_train, X_test, idx_test, y_train
                )

            logger.info("nSHAP completed")

            logger.info("Allclose nSHAP")
            check_allclose(
                nshap_folder,
                file,
                mh_algo,
            )
            logger.info("Allclose nSHAP completed")
        except Exception as e:
            logger.error(f"Error processing model {i+1}: {e}", exc_info=True)


def check_allclose(
    nshap_folder,
    file,
    mh_algo,
):

    file_paths = [
        os.path.join(nshap_folder, file + "_nshapley_values.csv"),
        os.path.join(nshap_folder, file + "_shapley_taylor_values.csv"),
        os.path.join(nshap_folder, file + "_faith_shap_values.csv"),
        os.path.join(nshap_folder, file + "_shapley_interaction_index_values.csv"),
        os.path.join(nshap_folder, file + "_banzhaf_interaction_index_values.csv"),
        os.path.join(nshap_folder, file + "_faith_banzhaf_values.csv"),
        os.path.join(nshap_folder, file + "_moebius_transform_values.csv"),
    ]

    file_labels = [
        path.split(mh_algo + "_")[-1].replace(".csv", "").replace("_values", "")
        for path in file_paths
    ]

    results = pd.DataFrame()

    for i, ref_file in enumerate(file_paths):
        ref_label = file_labels[i]
        ref_df = pd.read_csv(ref_file)
        ref_df = ref_df[ref_df["order"] != 0].reset_index(drop=True)

        results["id"] = ref_df["id"].unique()

        for j, compare_file in enumerate(file_paths):
            if i == j:
                continue

            compare_label = file_labels[j]
            compare_df = pd.read_csv(compare_file)
            compare_df = compare_df[compare_df["order"] != 0].reset_index(drop=True)

            ref_ids = set(ref_df["id"].unique())
            compare_ids = set(compare_df["id"].unique())

            if ref_ids != compare_ids:
                print(f"Mismatch in IDs between {ref_label} and {compare_label}.")
                continue

            comparison_results = []
            for unique_id in ref_ids:
                ref_id_df = ref_df[ref_df["id"] == unique_id]
                compare_id_df = compare_df[compare_df["id"] == unique_id]

                dict1 = ref_id_df.set_index(["interaction", "order"])["value"].to_dict()
                dict2 = compare_id_df.set_index(["interaction", "order"])[
                    "value"
                ].to_dict()

                is_close = nshap.allclose(dict1, dict2)

                if is_close:
                    is_close = 1
                else:
                    is_close = 0
                comparison_results.append(is_close)

            results[f"{ref_label}_vs_{compare_label}"] = comparison_results
    results.to_csv(
        os.path.join(nshap_folder, file + "_comparison_results.csv"), index=False
    )
