import os
import pandas as pd
from utils.data_loader import load_and_preprocess_data
from utils.metrics import (
    calculate_regression_metrics,
    calculate_classification_metrics,
    confusion_matrix_report,
    log_metrics,
)

import re
from sklearn.metrics import classification_report, confusion_matrix


from optimization.optimizer import optimize
from utils.logger import setup_logger
from utils.config import algorithm_settings
import datetime
import numpy as np


from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
from utils.config import models_path, output_path, json_path, original_data_id_folder
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import json

logger = setup_logger(__name__)
random_seed = 42
np.random.seed(random_seed)


def evaluate_ml_models(
    preprocessed_data,
    filename,
    filter_value,
    filter_column,
    target,
    all_results,
    task_type,
    model_registry,
):
    logger.info(f"----------ML model evaluation----------")
    temp_results = []

    if task_type == "classification":
        target_distribution = preprocessed_data[target].value_counts()
        logger.info(f"Target distribution:\n{target_distribution}")

        if len(target_distribution) < 2:
            logger.warning(
                f"Target variable {target} contains only one unique value. Skipping evaluation."
            )
            return [], all_results

    for model_name, model_cls in model_registry.items():
        model = model_cls()
        logger.info(f"\n\nEvaluating: {model_name} CV")
        model_instance = model.get_sklearn_estimator()

        if task_type == "classification":
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)

        y_pred = cross_val_predict(
            model_instance,
            preprocessed_data.drop(columns=[target]),
            preprocessed_data[target],
            cv=cv,
        )

        if task_type == "classification":

            conf_matrix = confusion_matrix(preprocessed_data[target], y_pred)
            logger.info(f"Confusion Matrix:\n{conf_matrix}")

            y_proba = cross_val_predict(
                model_instance,
                preprocessed_data.drop(columns=[target]),
                preprocessed_data[target],
                cv=cv,
                method="predict_proba",
            )

            if len(np.unique(preprocessed_data[target])) > 2:
                # Multi-class classification
                y_proba = (
                    y_proba if y_proba.ndim > 1 else np.expand_dims(y_proba, axis=1)
                )

                metrics = calculate_classification_metrics(
                    preprocessed_data[target], y_pred, y_proba, multi_class="ovr"
                )
            else:
                # Binary classification
                y_proba = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                metrics = calculate_classification_metrics(
                    preprocessed_data[target], y_pred, y_proba
                )

            report = classification_report(
                preprocessed_data[target], y_pred, output_dict=True, zero_division=0
            )

            # Extracting metrics from the classification report
            for label, metrics_dict in report.items():
                if isinstance(metrics_dict, dict):
                    for metric_name, metric_value in metrics_dict.items():
                        metrics[f"{label} {metric_name}"] = metric_value

        else:
            metrics = calculate_regression_metrics(preprocessed_data[target], y_pred)

        result = {
            "ml_model": model_name,
            "target": target,
            "filename": filename,
            "filter_column": filter_column,
            "filter_value": filter_value,
            **metrics,
        }

        all_results.append(result)
        temp_results.append(result)
        log_metrics(metrics, logger)

    results_df = pd.DataFrame(temp_results)

    selected_columns = (
        ["ml_model", "f1_score"]
        if task_type == "classification"
        else ["ml_model", "r2"]
    )

    top_models_df = results_df.sort_values(
        by="r2" if task_type == "regression" else "f1_score", ascending=False
    ).head(3)
    top_models_df = top_models_df[selected_columns].applymap(
        lambda x: f"{x:.4f}" if isinstance(x, float) else x
    )

    logger.info(f"Top models:\n{top_models_df.to_string(index=False)}")
    top_models = top_models_df["ml_model"].tolist()
    return top_models, all_results


def optimize_and_evaluate_model(
    datetime_col,
    filtered_data,
    X,
    y,
    model_name,
    filename,
    filter_column,
    filter_value,
    target,
    mh_algorithms,
    num_epochs,
    population_size,
    optimized_results_cv,
    optimized_results,
    combined_optimization_history,
    task_type,
    model_registry,
):

    model_optim_cls = model_registry[model_name]
    model_instance = model_optim_cls()
    model_constructor = model_instance.get_sklearn_estimator

    if algorithm_settings[model_name]["bounds"]:
        for metaheuristic in mh_algorithms:
            optimized_model, best_hyperparams, optimization_history = optimize(
                model_constructor=model_constructor,
                bounds=algorithm_settings[model_name]["bounds"],
                X=X,
                y=y,
                ml_model_name=model_name,
                metaheuristic=metaheuristic,
                epoch=num_epochs,
                pop_size=population_size,
                filename=filename,
                task_type=task_type,
            )

            if task_type == "classification":
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            else:
                cv = KFold(n_splits=5, shuffle=True, random_state=42)

            y_pred_optimized_cv = cross_val_predict(optimized_model, X, y, cv=cv)

            if task_type == "classification":
                y_proba_optimized_cv = cross_val_predict(
                    optimized_model, X, y, cv=cv, method="predict_proba"
                )

                if len(np.unique(y)) > 2:
                    # Multi-class classification
                    y_proba_optimized_cv = (
                        y_proba_optimized_cv
                        if y_proba_optimized_cv.ndim > 1
                        else np.expand_dims(y_proba_optimized_cv, axis=1)
                    )
                    metrics_optimized_cv = calculate_classification_metrics(
                        y, y_pred_optimized_cv, y_proba_optimized_cv, multi_class="ovr"
                    )
                else:
                    # Binary classification
                    y_proba_optimized_cv = (
                        y_proba_optimized_cv[:, 1]
                        if y_proba_optimized_cv.ndim > 1
                        else y_proba_optimized_cv
                    )
                    metrics_optimized_cv = calculate_classification_metrics(
                        y, y_pred_optimized_cv, y_proba_optimized_cv
                    )
            else:
                metrics_optimized_cv = calculate_regression_metrics(
                    y, y_pred_optimized_cv
                )

            logger.info(f"Optimized {model_name} CV by {metaheuristic}")
            log_metrics(metrics_optimized_cv, logger)

            optimized_result_entry = {
                "ml_model": model_name,
                "metaheuristic": metaheuristic,
                "target": target,
                "filename": filename,
                "filter_column": filter_column,
                "filter_value": filter_value,
                **metrics_optimized_cv,
            }
            optimized_results_cv.append(optimized_result_entry)

            for i, (g_best_value, runtime) in enumerate(
                zip(optimization_history[0][0], optimization_history[0][1]), start=1
            ):
                temp_df = pd.DataFrame(
                    {
                        "iteration": [i],
                        "g_best": [g_best_value],
                        "runtime": [runtime],
                        "ml_model": [model_name],
                        "metaheuristic": [metaheuristic],
                        "filename": [filename],
                        "filter_column": [filter_column],
                        "filter_value": [filter_value],
                        "target": [target],
                    }
                )

                combined_optimization_history = pd.concat(
                    [combined_optimization_history, temp_df],
                    ignore_index=True,
                )

            optimized_results = train_evaluate_save_model(
                datetime_col,
                filtered_data=filtered_data,
                model_name=model_name,
                best_hyperparams=best_hyperparams,
                filename=filename,
                filter_column=filter_column,
                filter_value=filter_value,
                target=target,
                models_path=models_path,
                metaheuristic=metaheuristic,
                optimized_results=optimized_results,
                task_type=task_type,
                model_registry=model_registry,
            )
    else:
        logger.warning(f"No hyperparameters found for {model_name}")

    return (
        optimized_results,
        optimized_results_cv,
        combined_optimization_history,
    )


def train_evaluate_save_model(
    datetime_col,
    filtered_data,
    model_name,
    best_hyperparams,
    filename,
    filter_value,
    filter_column,
    target,
    models_path,
    metaheuristic,
    optimized_results,
    task_type,
    model_registry,
):
    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        filtered_data,
        target,
        datetime_col,
        filename=filename,
        filter_column=filter_column,
        filter_value=filter_value,
        target_name=target,
        split_data=True,
        task_type=task_type,
    )

    model_cls = model_registry[model_name]
    optimized_model = model_cls(**best_hyperparams)

    optimized_model.train(X_train, y_train)

    y_pred_optimized = optimized_model.predict(X_test)

    if task_type == "classification":
        y_proba_optimized = optimized_model.predict_proba(X_test)
        if len(np.unique(y_train)) > 2:
            # Multi-class classification
            metrics_optimized = calculate_classification_metrics(
                y_test, y_pred_optimized, y_proba_optimized, multi_class="ovr"
            )
        else:
            # Binary classification
            y_proba_optimized = y_proba_optimized[:, 1]
            metrics_optimized = calculate_classification_metrics(
                y_test, y_pred_optimized, y_proba_optimized
            )
    else:
        metrics_optimized = calculate_regression_metrics(y_test, y_pred_optimized)

    logger.info(f"Optimized {model_name} by {metaheuristic}")
    log_metrics(metrics_optimized, logger)

    model_filename = f"filename_{filename}_filter_col_{filter_column}_filter_val_{filter_value}_target_{target}_ml_model_{model_name}_mh_algo_{metaheuristic}.joblib"
    optimized_model.save(os.path.join(models_path, model_filename))

    optimized_result_entry = {
        "ml_model": model_name,
        "metaheuristic": metaheuristic,
        "target": target,
        "filename": filename,
        "filter_column": filter_column,
        "filter_value": filter_value,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "num_columns": X_train.shape[1],
        **metrics_optimized,
        **best_hyperparams,
    }
    optimized_results.append(optimized_result_entry)

    return optimized_results


def preprocess_dataset(dataset, datetime_col):
    """
    Preprocess the dataset by encoding categorical features.
    """
    categorical_columns = dataset.select_dtypes(include=["object"]).columns
    categorical_columns = [
        col for col in categorical_columns if col not in [datetime_col]
    ]

    for col in categorical_columns:
        encoder = LabelEncoder()
        dataset[col] = encoder.fit_transform(dataset[col])

    return dataset


def create_json_options(filter_values, targets, filename, json_path, filter_column):
    """
    Creates a JSON file with filter values and target options for a given file.

    Parameters:
    - filter_values: List of unique filter values.
    - targets: List of target categories.
    - filename: Name of the file or dataset (filename without extension).
    - json_path: Path to the directory where the JSON file should be saved.
    - filter_column: The column name used for filtering.
    """
    # Construct the options dictionary
    if filter_column == None:
        filter_values = [filter_values]
    options = {
        "filterOptions": [
            {"label": str(value), "value": str(value)} for value in filter_values
        ],
        "targetOptions": [{"label": target, "value": target} for target in targets],
        "targetList": targets,
        "filterList": filter_values,
        "filterColumn": filter_column,
    }

    # Define the JSON file path for the current dataset
    json_file_path = os.path.join(json_path, f"{filename}_options.json")

    # Write the options to the JSON file
    with open(json_file_path, "w") as file:
        json.dump(options, file, indent=4)

    logger.info(f"JSON options file saved for {filename}")


def clean_filename(filename):

    # Separate the base filename and extension
    base, ext = os.path.splitext(filename)

    # Apply cleaning to the base filename (without the extension)
    base = base.lower()
    base = base.replace(" ", "_")
    base = re.sub(r"[^a-zA-Z0-9_-]", "", base)

    # Return the cleaned base filename with the original extension
    return base  # + ext


def perform_training_and_optimization(
    datasets_path,
    num_epochs,
    population_size,
    targets,
    mh_algorithms,
    task_type,
    filter_column,
    datetime_col,
    model_registry,
):
    all_results = []
    optimized_results = []
    optimized_results_cv = []
    combined_optimization_history = pd.DataFrame()

    for file in os.listdir(datasets_path):

        if not file.endswith(".csv") or file == ".DS_Store":
            continue

        file_path = os.path.join(datasets_path, file)

        filename = clean_filename(file)
        logger.info(f"File: {filename}")
        dataset = pd.read_csv(file_path)

        dataset["id"] = range(len(dataset))

        if filter_column and filter_column in dataset.columns:
            filter_values = dataset[filter_column].unique().tolist()
            # dataset["id"] = range(len(dataset))
            dataset = preprocess_dataset(dataset, datetime_col)
            dataset.to_csv(
                os.path.join(original_data_id_folder, f"{filename}.csv"),
                index=False,
            )

            for filter_value in filter_values:
                filtered_data = dataset[dataset[filter_column] == filter_value]
                filtered_data = filtered_data.drop(filter_column, axis=1)

                for target in targets:
                    logger.info(
                        f"-------{filter_column}: {filter_value}------Target: {target}-------"
                    )

                    preprocessed_data = load_and_preprocess_data(
                        filtered_data,
                        target,
                        datetime_col,
                        filename=filename,
                        filter_column=filter_column,
                        filter_value=filter_value,
                        target_name=target,
                        split_data=False,
                        task_type=task_type,
                    )
                    top_models, all_results = evaluate_ml_models(
                        preprocessed_data,
                        filename,
                        filter_column,
                        filter_value,
                        target,
                        all_results,
                        task_type,
                        model_registry,
                    )

                    X = preprocessed_data.drop(columns=[target])
                    y = preprocessed_data[target]

                    logger.info(f"-------------Optimization-------------")

                    for model_name in top_models:
                        logger.info(f"Optimizing: {model_name} CV")
                        (
                            optimized_results,
                            optimized_results_cv,
                            combined_optimization_history,
                        ) = optimize_and_evaluate_model(
                            datetime_col,
                            filtered_data,
                            X=X,
                            y=y,
                            model_name=model_name,
                            filename=filename,
                            filter_value=filter_value,
                            filter_column=filter_column,
                            target=target,
                            mh_algorithms=mh_algorithms,
                            num_epochs=num_epochs,
                            population_size=population_size,
                            optimized_results_cv=optimized_results_cv,
                            optimized_results=optimized_results,
                            combined_optimization_history=combined_optimization_history,
                            task_type=task_type,
                            model_registry=model_registry,
                        )
                logger.info(combined_optimization_history)
        else:
            filter_values = None
            # dataset["id"] = range(len(dataset))
            filtered_data = dataset.copy()
            filtered_data = preprocess_dataset(filtered_data, datetime_col)
            filtered_data.to_csv(
                os.path.join(original_data_id_folder, f"{filename}.csv"),
                index=False,
            )

            for target in targets:
                logger.info(f"-------Target: {target}-------")

                preprocessed_data = load_and_preprocess_data(
                    filtered_data,
                    target,
                    datetime_col,
                    filename=filename,
                    target_name=target,
                    split_data=False,
                    task_type=task_type,
                )
                top_models, all_results = evaluate_ml_models(
                    preprocessed_data,
                    filename,
                    filter_column,
                    filter_values,
                    target,
                    all_results,
                    task_type,
                    model_registry,
                )

                X = preprocessed_data.drop(columns=[target])
                y = preprocessed_data[target]

                logger.info(f"-------------Optimization-------------")

                for model_name in top_models:
                    logger.info(f"Optimizing: {model_name} CV")
                    (
                        optimized_results,
                        optimized_results_cv,
                        combined_optimization_history,
                    ) = optimize_and_evaluate_model(
                        datetime_col,
                        filtered_data,
                        X=X,
                        y=y,
                        model_name=model_name,
                        filename=filename,
                        filter_value=None,
                        filter_column=filter_column,
                        target=target,
                        mh_algorithms=mh_algorithms,
                        num_epochs=num_epochs,
                        population_size=population_size,
                        optimized_results_cv=optimized_results_cv,
                        optimized_results=optimized_results,
                        combined_optimization_history=combined_optimization_history,
                        task_type=task_type,
                        model_registry=model_registry,
                    )
            logger.info(combined_optimization_history)

    create_json_options(filter_values, targets, filename, json_path, filter_column)

    combined_optimization_history.to_csv(
        os.path.join(output_path, "results", "optimization_history.csv"),
        index=False,
    )

    pd.DataFrame(all_results).to_csv(
        os.path.join(output_path, "results", "ml_models_results.csv"), index=False
    )

    pd.DataFrame(optimized_results).to_csv(
        os.path.join(output_path, "results", "optimized_models_results.csv"),
        index=False,
    )

    pd.DataFrame(optimized_results_cv).to_csv(
        os.path.join(output_path, "results", "optimized_models_results_cv.csv"),
        index=False,
    )
