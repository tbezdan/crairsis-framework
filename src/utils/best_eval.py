import pandas as pd
import joblib
from sklearn.metrics import r2_score
from utils.logger import setup_logger
import os
from utils.config import models_path, datasets_path, destination_folder

logger = setup_logger(__name__)


def perform_best_models_evaluation(best_models):
    for i in range(best_models.shape[0]):
        target = best_models.loc[i, "target"]
        site = best_models.loc[i, "site"]
        covid = best_models.loc[i, "covid"]
        ml_model = best_models.loc[i, "ml_model"]
        mh_algo = best_models.loc[i, "mh_algo"]
        file = f"site_{site}_covid_{covid}_target_{target}_ml_model_{ml_model}_mh_algo_{mh_algo}"
        model_path = os.path.join(models_path, file + ".joblib")
        data_path = os.path.join(
            datasets_path, f"site_{site}_covid_{covid}_target_{target}.csv"
        )

        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)

        logger.info(f"Loading data from {data_path}")
        X = pd.read_csv(data_path)

        test_data = X[X["Usage"] == "Test"]
        idx = test_data["id"].reset_index(drop=True).values
        test_data = test_data.drop(["Usage", "Datetime", "id"], axis=1)

        x = test_data.drop(target, axis=1)
        actual = test_data[target]
        predictions = model.predict(x)
        data = {"id": idx, "actual": actual, "predicted": predictions}
        r2 = float(r2_score(actual, predictions))
        df = pd.DataFrame(data)
        output_file_path = os.path.join(destination_folder, file + ".csv")
        df.to_csv(output_file_path, index=False)

        logger.info(file)
        logger.info(
            f"R2 score difference: {round(r2 - best_models.loc[i, 'Value'], 4)}"
        )
