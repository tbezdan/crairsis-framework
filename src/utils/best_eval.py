import pandas as pd
import joblib
from sklearn.metrics import (
    r2_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from utils.logger import setup_logger
import os


logger = setup_logger(__name__)


def perform_best_models_evaluation(
    best_models,
    filter_column,
    task_type,
    data_usage,
    datetime_col,
    models_path,
    datasets_path,
    actual_predicted_folder,
):
    for i in range(best_models.shape[0]):
        target = best_models.loc[i, "target"]
        filename = best_models.loc[i, "filename"]
        if filter_column == None:
            filter_value = None
        else:
            filter_value = best_models.loc[i, "filter_value"]
        ml_model = best_models.loc[i, "ml_model"]
        mh_algo = best_models.loc[i, "metaheuristic"]
        file = f"filename_{filename}_filter_col_{filter_column}_filter_val_{filter_value}_target_{target}_ml_model_{ml_model}_mh_algo_{mh_algo}"
        model_path = os.path.join(models_path, file + ".joblib")
        data_path = os.path.join(
            datasets_path,
            f"filename_{filename}_filter_col_{filter_column}_filter_val_{filter_value}_target_{target}.csv",
        )

        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)

        logger.info(f"Loading data from {data_path}")
        X = pd.read_csv(data_path)

        if data_usage == "train_and_test":

            selected_data = X
        else:

            selected_data = X[X["usage"] == "test"]

        idx = selected_data["id"].reset_index(drop=True).values
        usage = selected_data["usage"].reset_index(drop=True).values

        columns_to_drop = ["usage", "id"]
        if datetime_col:
            columns_to_drop.append(datetime_col)

        selected_data = selected_data.drop(columns_to_drop, axis=1)

        x = selected_data.drop(target, axis=1)
        actual = selected_data[target]
        predictions = model.predict(x)

        data = {
            "id": idx,
            "usage": usage,
            "actual": actual,
            "predicted": predictions,
        }

        if task_type == "classification" and hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(x)
            class_labels = model.classes_

            for j, class_label in enumerate(class_labels):
                data[f"prob_class_{class_label}"] = probabilities[:, j]

            data["max_probability"] = probabilities.max(axis=1)

        df = pd.DataFrame(data)
        output_file_path = os.path.join(actual_predicted_folder, file + ".csv")
        df.to_csv(output_file_path, index=False)

        logger.info(file)

        if task_type == "regression":
            r2 = float(r2_score(actual, predictions))
            logger.info(
                f"R2 score difference: {round(r2 - best_models.loc[i, 'value'], 4)}"
            )
        else:
            accuracy = accuracy_score(actual, predictions)
            f1 = f1_score(actual, predictions, average="weighted")
            precision = precision_score(actual, predictions, average="weighted")
            recall = recall_score(actual, predictions, average="weighted")
            report = classification_report(actual, predictions, output_dict=True)

            logger.info(f"Accuracy: {accuracy}")
            logger.info(f"F1 Score: {f1}")
            logger.info(f"Precision: {precision}")
            logger.info(f"Recall: {recall}")
            logger.info(f"Classification Report: {report}")
