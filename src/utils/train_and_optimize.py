import os
import pandas as pd
from utils.data_loader import load_and_preprocess_data
from utils.metrics import calculate_metrics, log_metrics
from optimization.optimizer import optimize
from utils.logger import setup_logger
from utils.config import algorithm_settings
import datetime
import numpy as np
from ml_models.catboost_model import CatBoostModel
from ml_models.adaboost_model import AdaBoostModel
from ml_models.lightgbm_model import LGBMModel
from ml_models.xgboost_model import XGBoostModel
from ml_models.extratrees_model import ExtraTreesModel
from ml_models.gradientboosting_model import GradientBoostingModel
from ml_models.histgradientboosting_model import HistGradientBoostingModel
from sklearn.model_selection import cross_val_predict
from utils.config import models_path, output_path, json_path, original_data_id_folder
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import json

logger = setup_logger(__name__)
random_seed = 42
np.random.seed(random_seed)


model_registry = {
    "CatBoostModel": CatBoostModel,
    "AdaBoostModel": AdaBoostModel,
    "LGBMModel": LGBMModel,
    "XGBoostModel": XGBoostModel,
    "ExtraTreesModel": ExtraTreesModel,
    "GradientBoostingModel": GradientBoostingModel,
    "HistGradientBoostingModel": HistGradientBoostingModel,
}


def evaluate_ml_models(preprocessed_data, site, covid, target, all_results):
    print()
    logger.info(f"----------ML model evaluation----------")
    temp_results = []
    for model_name, model_cls in model_registry.items():
        model = model_cls()
        print()
        logger.info(f"Evaluating: {model_name} CV")
        model_instance = model.get_sklearn_estimator()

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        y_pred = cross_val_predict(
            model_instance,
            preprocessed_data.drop(columns=[target]),
            preprocessed_data[target],
            cv=cv,
        )
        metrics = calculate_metrics(preprocessed_data[target], y_pred)

        result = {
            "ML Model": model_name,
            "Target": target,
            "Filename": site,
            "Covid": covid,
            **metrics,
        }

        all_results.append(result)
        temp_results.append(result)
        log_metrics(metrics, logger)

    results_df = pd.DataFrame(temp_results)

    top_models_df = results_df.sort_values(by="R2", ascending=False).head(3)
    top_models_df = top_models_df.applymap(
        lambda x: f"{x:.4f}" if isinstance(x, float) else x
    )

    df_string = top_models_df.to_string(index=False)
    print()
    logger.info(f"Top models:\n{df_string}")
    top_models = top_models_df["ML Model"].tolist()
    return top_models, all_results


def optimize_and_evaluate_model(
    filtered_data,
    X,
    y,
    model_name,
    site,
    covid,
    target,
    mh_algorithms,
    num_epochs,
    population_size,
    optimized_results_cv,
    optimized_results,
    combined_optimization_history,
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
                filename=site,
            )

            # Evaluate the optimized model with cross-validation and calculate metrics
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            y_pred_optimized_cv = cross_val_predict(optimized_model, X, y, cv=cv)
            metrics_optimized_cv = calculate_metrics(y, y_pred_optimized_cv)

            # Log the optimized metrics
            print()
            logger.info(f"Optimized {model_name} CV by {metaheuristic}")
            log_metrics(metrics_optimized_cv, logger)

            # Append the results to the optimized results list
            optimized_result_entry = {
                "ML Model": model_name,
                "Metaheuristic": metaheuristic,
                "Target": target,
                "Filename": site,
                "Covid": covid,
                **metrics_optimized_cv,
            }
            optimized_results_cv.append(optimized_result_entry)

            for i, (g_best_value, runtime) in enumerate(
                zip(optimization_history[0][0], optimization_history[0][1]), start=1
            ):

                temp_df = pd.DataFrame(
                    {
                        "Iteration": [i],
                        "G_Best": [g_best_value],
                        "Runtime": [runtime],
                        "ML_Model": [model_name],
                        "Metaheuristic": [metaheuristic],
                        "Site": [site],
                        "Covid": [covid],
                        "Target": [target],
                    }
                )

                combined_optimization_history = pd.concat(
                    [combined_optimization_history, temp_df],
                    ignore_index=True,
                )
            optimized_results = train_evaluate_save_model(
                filtered_data=filtered_data,
                model_name=model_name,
                best_hyperparams=best_hyperparams,
                site=site,
                covid=covid,
                target=target,
                output_dir=output_path,
                models_path=models_path,
                metaheuristic=metaheuristic,
                optimized_results=optimized_results,
            )

    else:
        logger.warning(f"No hyperparameters found for {model_name}")

    return (
        optimized_results,
        optimized_results_cv,
        combined_optimization_history,
    )


def train_evaluate_save_model(
    filtered_data,
    model_name,
    best_hyperparams,
    site,
    covid,
    target,
    output_dir,
    models_path,
    metaheuristic,
    optimized_results,
):
    # Split the data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        filtered_data,
        target,
        output_dir=output_dir,
        site=site,
        covid=covid,
        target_name=target,
        split_data=True,
    )

    # Initialize the model with optimized hyperparameters
    model_cls = model_registry[model_name]
    optimized_model = model_cls(**best_hyperparams)

    # Train the optimized model
    optimized_model.train(X_train, y_train)

    # Predict on test set and calculate metrics
    y_pred_optimized = optimized_model.predict(X_test)
    metrics_optimized = calculate_metrics(y_test, y_pred_optimized)
    print()
    logger.info(f"Optimized {model_name} by {metaheuristic}")
    log_metrics(metrics_optimized, logger)

    # Save the optimized model
    model_filename = f"site_{site}_covid_{covid}_target_{target}_ml_model_{model_name}_mh_algo_{metaheuristic}.joblib"
    optimized_model.save(os.path.join(models_path, model_filename))

    # Update the optimized results
    optimized_result_entry = {
        "ML Model": model_name,
        "Metaheuristic": metaheuristic,
        "Target": target,
        "Filename": site,
        "Covid": covid,
        **metrics_optimized,
        "Train Size": len(X_train),
        "Test Size": len(X_test),
        "Num Columns": X_train.shape[1],
        **best_hyperparams,
    }
    optimized_results.append(optimized_result_entry)

    return optimized_results


def preprocess_dataset(dataset):
    """
    Preprocess the dataset by encoding categorical features.
    """
    categorical_columns = dataset.select_dtypes(include=["object"]).columns
    categorical_columns = [
        col for col in categorical_columns if col not in ["Datetime"]
    ]

    for col in categorical_columns:
        encoder = LabelEncoder()
        dataset[col] = encoder.fit_transform(dataset[col])

    return dataset


import json
import os


def create_json_options(covid_era_values, targets, site, json_path):
    """
    Creates a JSON file with COVID era and target options for a given site.

    Parameters:
    - covid_era_values: List of unique COVID era values.
    - targets: List of target categories.
    - site: Name of the site or dataset (filename without extension).
    - json_path: Path to the directory where the JSON file should be saved.
    """
    # Construct the options dictionary
    options = {
        "covidOptions": [
            {"label": str(covid), "value": str(covid)} for covid in covid_era_values
        ],
        "targetOptions": [{"label": target, "value": target} for target in targets],
        "targetList": targets,
        "covidList": covid_era_values,
    }

    # Define the JSON file path for the current dataset
    json_file_path = os.path.join(json_path, f"{site}_options.json")

    # Write the options to the JSON file
    with open(json_file_path, "w") as file:
        json.dump(options, file, indent=4)

    logger.info(f"JSON options file saved for {site}")


def perform_training_and_optimization(
    datasets_path, num_epochs, population_size, targets, mh_algorithms
):
    all_results = []
    optimized_results = []
    optimized_results_cv = []
    combined_optimization_history = pd.DataFrame()

    for filename in os.listdir(datasets_path):

        if not filename.endswith(".csv") or filename == ".DS_Store":
            continue

        file_path = os.path.join(datasets_path, filename)
        site, _ = os.path.splitext(filename)
        logger.info(f"File: {site}")
        dataset = pd.read_csv(file_path)
        dataset["id"] = range(len(dataset))
        dataset = preprocess_dataset(dataset)
        dataset.to_csv(
            os.path.join(original_data_id_folder, f"{site}.csv"),
            index=False,
        )

        covid_era_values = dataset["covid_era"].unique().tolist()

        create_json_options(covid_era_values, targets, site, json_path)

        for covid in dataset["covid_era"].unique():
            filtered_data = dataset[dataset["covid_era"] == covid]
            filtered_data = filtered_data.drop("covid_era", axis=1)

            for target in targets:
                print()
                logger.info(f"-------Covid: {covid}------Target: {target}-------")

                preprocessed_data = load_and_preprocess_data(
                    filtered_data,
                    target,
                    output_dir=output_path,
                    site=site,
                    covid=covid,
                    target_name=target,
                    split_data=False,
                )
                top_models, all_results = evaluate_ml_models(
                    preprocessed_data, site, covid, target, all_results
                )

                X = preprocessed_data.drop(columns=[target])
                y = preprocessed_data[target]

                print()
                logger.info(f"-------------Optimization-------------")

                for model_name in top_models:
                    print()
                    logger.info(f"Optimizing: {model_name} CV")
                    (
                        optimized_results,
                        optimized_results_cv,
                        combined_optimization_history,
                    ) = optimize_and_evaluate_model(
                        filtered_data,
                        X=X,
                        y=y,
                        model_name=model_name,
                        site=site,
                        covid=covid,
                        target=target,
                        mh_algorithms=mh_algorithms,
                        num_epochs=num_epochs,
                        population_size=population_size,
                        optimized_results_cv=optimized_results_cv,
                        optimized_results=optimized_results,
                        combined_optimization_history=combined_optimization_history,
                    )
            print(combined_optimization_history)

        combined_optimization_history.to_csv(
            os.path.join(output_path, "results", "optimization_history.csv"),
            index=False,
        )

        # Save initial ML results to CSV
        pd.DataFrame(all_results).to_csv(
            os.path.join(output_path, "results", "ml_models_results.csv"), index=False
        )

        # Save optimized model results to CSV
        pd.DataFrame(optimized_results).to_csv(
            os.path.join(output_path, "results", "optimized_models_results.csv"),
            index=False,
        )

        # Save optimized model results to CSV
        pd.DataFrame(optimized_results_cv).to_csv(
            os.path.join(output_path, "results", "optimized_models_results_cv.csv"),
            index=False,
        )
