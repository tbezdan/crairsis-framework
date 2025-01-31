import warnings
import os
from ixai.explainer import IncrementalPFI
from ixai.explainer.sage import IncrementalSage, IntervalSage
from ixai.imputer import MarginalImputer
from ixai.storage import GeometricReservoirStorage
from ixai.utils.wrappers import SklearnWrapper
import pandas as pd
from river import metrics
from river.stream import iter_pandas
from river.metrics import MSE
from river.utils import Rolling
import joblib
from tqdm import tqdm
from utils.logger import setup_logger


warnings.simplefilter("ignore")
logger = setup_logger(__name__)


def get_data_model(
    models_path,
    datasets_path,
    file,
    filename,
    filter_column,
    filter_value,
    target,
    datetime_col,
):

    data_path = os.path.join(
        datasets_path,
        f"filename_{filename}_filter_col_{filter_column}_filter_val_{filter_value}_target_{target}.csv",
    )

    X = pd.read_csv(data_path)

    columns_to_drop = ["usage", "id", target]
    if datetime_col:
        columns_to_drop.append(datetime_col)

    data_x = X.drop(columns_to_drop, axis=1)
    data_y = X.loc[:, target]

    X_train = X[X["usage"] == "train"]
    X_test = X[X["usage"] == "test"]
    y_train = X[X["usage"] == "train"][target]
    y_test = X[X["usage"] == "test"][target]
    X_train = X_train.drop(columns_to_drop, axis=1)
    X_test = X_test.drop(columns_to_drop, axis=1)

    feature_names = X_train.columns

    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

    model_path = os.path.join(models_path, file + ".joblib")
    model = joblib.load(model_path)

    return X_train, X_test, y_train, y_test, feature_names, model, data_x, data_y


def calculate_isage(
    isage_folder,
    model,
    task_type,
    feature_names,
    X_test,
    y_test,
    data_x,
    data_y,
    file,
):
    model.score(X_test, y_test)
    model_function = SklearnWrapper(model.predict)

    if task_type == "classification":
        loss_metric = metrics.F1()
        training_metric = Rolling(metrics.F1(), window_size=1000)
    elif task_type == "regression":
        loss_metric = MSE()
        training_metric = Rolling(MSE(), window_size=1000)

    storage = GeometricReservoirStorage(size=200, store_targets=False)
    imputer = MarginalImputer(
        model_function=model_function, storage_object=storage, sampling_strategy="joint"
    )
    incremental_sage = IncrementalSage(
        model_function=model_function,
        loss_function=loss_metric,
        imputer=imputer,
        storage=storage,
        feature_names=feature_names,
        smoothing_alpha=0.001,
        n_inner_samples=1,
    )

    interval_sage = IntervalSage(
        model_function=model_function,
        loss_function=loss_metric,
        feature_names=feature_names,
        interval_length=2000,
        n_inner_samples=1,
    )
    incremental_pfi = IncrementalPFI(
        model_function=model_function,
        loss_function=loss_metric,
        imputer=imputer,
        storage=storage,
        feature_names=feature_names,
        smoothing_alpha=0.001,
        n_inner_samples=1,
    )

    inc_sage_results = []
    int_sage_results = []
    pfi_results = []

    # Main training loop
    for n, (x_i, y_i) in enumerate(tqdm(iter_pandas(data_x, data_y)), start=1):
        y_i_pred = model_function(x_i)["output"]
        training_metric.update(y_true=y_i, y_pred=y_i_pred)

        # updates feature importance values, marginal loss, and model loss internally
        _ = incremental_sage.explain_one(x_i, y_i)
        _ = interval_sage.explain_one(x_i, y_i)
        _ = incremental_pfi.explain_one(x_i, y_i, update_storage=False)

        general_result = {
            "iteration": n,
            "performance": training_metric.get(),
            "marginal_loss": incremental_sage.marginal_loss,
            "model_loss": incremental_sage.model_loss,
            "difference": incremental_sage.marginal_loss - incremental_sage.model_loss,
            "sum_sage": sum(list(incremental_sage.importance_values.values())),
            "target_value": y_i,
        }

        # Add incremental SAGE contributions
        inc_sage_result = general_result.copy()
        for key, value in incremental_sage.importance_values.items():
            inc_sage_result[key] = value
        inc_sage_results.append(inc_sage_result)

        # Add interval SAGE contributions
        int_sage_result = general_result.copy()
        for key, value in interval_sage.importance_values.items():
            int_sage_result[key] = value
        int_sage_results.append(int_sage_result)

        # Add PFI contributions
        pfi_result = general_result.copy()
        for key, value in incremental_pfi.importance_values.items():
            pfi_result[key] = value
        pfi_results.append(pfi_result)

    pd.DataFrame(inc_sage_results).to_csv(
        os.path.join(isage_folder, file + "_incremental_sage_results.csv"), index=False
    )
    pd.DataFrame(int_sage_results).to_csv(
        os.path.join(isage_folder, file + "_interval_sage_results.csv"), index=False
    )

    pd.DataFrame(pfi_results).to_csv(
        os.path.join(isage_folder, file + "_pfi_results.csv"), index=False
    )


def perform_isage_analysis(
    isage_folder,
    models_path,
    datasets_path,
    best_models,
    filter_column,
    task_type,
    datetime_col,
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
            X_train, X_test, y_train, y_test, feature_names, model, data_x, data_y = (
                get_data_model(
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

            logger.info("iSAGE started")

            calculate_isage(
                isage_folder,
                model,
                task_type,
                feature_names,
                X_test,
                y_test,
                data_x,
                data_y,
                file,
            )

            logger.info("iSAGE completed")

        except Exception as e:
            logger.error(f"Error processing model {i+1}: {e}", exc_info=True)
