from utils.train_and_optimize import perform_training_and_optimization
from xai.shap_analysis import perform_shap_analysis
from xai.sage_analysis import perform_sage_analysis
from xai.shap_clustering import perform_shap_clustering
from utils.best_eval import perform_best_models_evaluation
from utils.logger import setup_logger
from utils.config import (
    best_models_path,
    original_datasets_path,
    output_path,
    json_path,
    paths_to_check,
)
from utils.result_formatter import format_best_models, format_detailed_metrics
import pandas as pd
import json
import os
import time
from datetime import datetime
from utils.timing import log_execution_time
from ml_models.adaboost_classification_model import AdaBoostClassificationModel
from ml_models.lightgbm_classification_model import LGBMClassificationModel
from ml_models.xgboost_classification_model import XGBClassificationModel
from ml_models.extratrees_classification_model import ExtraTreesClassificationModel
from ml_models.gradientboosting_classification_model import (
    GradientBoostingClassificationModel,
)
from ml_models.histgradientboosting_classification_model import (
    HistGradientBoostingCls,
)


from ml_models.adaboost_regression_model import AdaBoostRegressionModel
from ml_models.lightgbm_regression_model import LGBMRegressionModel
from ml_models.xgboost_regression_model import XGBRegressionModel
from ml_models.extratrees_regression_model import ExtraTreesRegressionModel
from ml_models.gradientboosting_regression_model import GradientBoostingRegressionModel
from ml_models.histgradientboosting_regression_model import (
    HistGradientBoostingRegressionModel,
)
from ml_models.balanced_random_forest_classification_model import (
    BalancedRandomForestCls,
)


from utils.setup_logger import log_setup_info

# /Users/timea/miniforge3/envs/crairsis/bin/python "/Users/timea/Documents/Projekti/craAIRsis/Covid BG/src/main.py"

logger = setup_logger(__name__)

# ovde ce iz config fajla uzeti parametre (from Milos)
user_name = "timea_bezdan"


num_epochs = 5
population_size = 25

# targets = ["m79", "m93"]
# targets = ['PM2_5_Aerosol_PM2_5_6001'] #, 'Carbon_monoxide_Air_10', 'Ozone_Air_7', 'Sulphur_dioxide_Air_1', 'Nitrogen_dioxide_Air_8', 'Nitrogen_monoxide_Air_38', 'Nitrogen_oxides_Air_9']
targets = [
    "Acetamiprid",
    "Azoxystrobin",
    "Boscalid",
    "Chlorantraniliprole",
    "Difenoconazole",
    "Fluopiram",
    "Fluxapyroxad",
    "Metalaxyl",
]

# mh_algorithms = ["SCA", "HHO"]
mh_algorithms = ["SCA"]


# targets = ["PM2_5_Aerosol_PM2_5_6001"]
# ["cls", "crai"]
mh_algorithms = ["SCA", "HHO"]
filter_col = "CRAI"  # "CRAI"  # None #"covid_era"
task_type = "regression"  # "classification"  # regression
data_usage = (
    "test"  # "train_and_test"  # "test" # two possible options: train_and_test, test
)
datetime_col = "Datetime"  # "Datetime" #None
threshold = 90  # default

dimensionality_reduction_method = (
    "PaCMAP"  # two possible options:"PaCMAP", "UMAP", default "PaCMAP"
)

perform_subclustering = True  # default: False

if (
    perform_subclustering
):  # if perform_subclustering True, the user can specify the value of subcluster_prob_threshold
    subcluster_prob_threshold = 0.6  # [0,1] default: 0.6
else:
    subcluster_prob_threshold = None

# model_registry


classification_model_registry = {
    "BalancedRandomForestCls": BalancedRandomForestCls,
    # "AdaBoostModel": AdaBoostClassificationModel,
    "LGBMModel": LGBMClassificationModel,
    # "XGBoostModel": XGBClassificationModel,
    "ExtraTreesModel": ExtraTreesClassificationModel,
    # "GradientBoostingModel": GradientBoostingClassificationModel, #GradientBoostingClassifier is only supported for binary classification right now!
    "HistGradientBoostingCls": HistGradientBoostingCls,
}

regression_model_registry = {
    "AdaBoostModel": AdaBoostRegressionModel,
    "LGBMModel": LGBMRegressionModel,
    "XGBoostModel": XGBRegressionModel,
    "ExtraTreesModel": ExtraTreesRegressionModel,
    "GradientBoostingModel": GradientBoostingRegressionModel,
    "HistGradientBoostingModel": HistGradientBoostingRegressionModel,
}

from datetime import datetime


def log_setup_info_func():

    model_registry = (
        classification_model_registry
        if task_type == "classification"
        else regression_model_registry
    )

    log_setup_info(
        user_name,
        num_epochs,
        population_size,
        targets,
        mh_algorithms,
        filter_col,
        task_type,
        data_usage,
        datetime_col,
        threshold,
        model_registry,
        dimensionality_reduction_method,
        perform_subclustering,
        subcluster_prob_threshold,
    )


def create_folders():
    for path in paths_to_check:
        os.makedirs(path, exist_ok=True)
        print(f"Checked/created: {path}")


def train_and_optimize_models(filter_col, datetime_col):

    model_registry = (
        classification_model_registry
        if task_type == "classification"
        else regression_model_registry
    )

    train_optimize_start_time = time.perf_counter()
    train_optimize_start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"train_and_optimize_models started at {train_optimize_start_timestamp}")

    perform_training_and_optimization(
        original_datasets_path,
        num_epochs,
        population_size,
        targets,
        mh_algorithms,
        task_type,
        filter_col,
        datetime_col,
        model_registry,
    )

    train_optimize_end_time = time.perf_counter()
    train_optimize_end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    train_optimize_exec_time = train_optimize_end_time - train_optimize_start_time
    print(f"train_and_optimize_models ended at {train_optimize_end_timestamp}")

    log_execution_time(
        "train_and_optimize_models",
        train_optimize_exec_time,
        train_optimize_start_timestamp,
        train_optimize_end_timestamp,
    )


def evaluate_best_models(best_models, task_type, data_usage, datetime_col):
    evaluate_start_time = time.perf_counter()
    evaluate_start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"evaluate_best_models started at {evaluate_start_timestamp}")

    perform_best_models_evaluation(
        best_models, filter_col, task_type, data_usage, datetime_col
    )

    evaluate_end_time = time.perf_counter()
    evaluate_end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    evaluate_exec_time = evaluate_end_time - evaluate_start_time
    print(f"evaluate_best_models ended at {evaluate_end_timestamp}")

    log_execution_time(
        "evaluate_best_models",
        evaluate_exec_time,
        evaluate_start_timestamp,
        evaluate_end_timestamp,
    )


def shap_calculation(best_models, task_type, data_usage, datetime_col):
    shap_start_time = time.perf_counter()
    shap_start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"shap_calculation started at {shap_start_timestamp}")

    perform_shap_analysis(best_models, filter_col, task_type, data_usage, datetime_col)

    shap_end_time = time.perf_counter()
    shap_end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    shap_exec_time = shap_end_time - shap_start_time
    print(f"shap_calculation ended at {shap_end_timestamp}")
    log_execution_time(
        "shap_calculation", shap_exec_time, shap_start_timestamp, shap_end_timestamp
    )


def shap_clustering(
    task_type,
    dimensionality_reduction_method,
    perform_subclustering,
    subcluster_prob_threshold,
):
    shap_cluster_start_time = time.perf_counter()
    shap_cluster_start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"shap_clustering started at {shap_cluster_start_timestamp}")

    perform_shap_clustering(
        task_type,
        dimensionality_reduction_method,
        perform_subclustering,
        subcluster_prob_threshold,
    )

    shap_cluster_end_time = time.perf_counter()
    shap_cluster_end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    shap_cluster_exec_time = shap_cluster_end_time - shap_cluster_start_time
    print(f"shap_clustering ended at {shap_cluster_end_timestamp}")
    log_execution_time(
        "shap_clustering",
        shap_cluster_exec_time,
        shap_cluster_start_timestamp,
        shap_cluster_end_timestamp,
    )


def sage_calculation(best_models, task_type, data_usage, datetime_col, threshold):

    sage_start_time = time.perf_counter()
    sage_start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"sage_calculation started at {sage_start_timestamp}")

    perform_sage_analysis(
        best_models, filter_col, task_type, data_usage, datetime_col, threshold
    )

    sage_end_time = time.perf_counter()
    sage_end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sage_exec_time = sage_end_time - sage_start_time
    print(f"sage_calculation ended at {sage_end_timestamp}")

    log_execution_time(
        "sage_calculation", sage_exec_time, sage_start_timestamp, sage_end_timestamp
    )


def execute_main_tasks():

    create_folders()
    # train_and_optimize_models(filter_col, datetime_col)
    # format_best_models(output_path, task_type=task_type)
    # best_models = pd.read_csv(best_models_path)
    # format_detailed_metrics(output_path, task_type)
    # evaluate_best_models(best_models, task_type, data_usage, datetime_col)
    # shap_calculation(best_models, task_type, data_usage, datetime_col)
    shap_clustering(
        task_type,
        dimensionality_reduction_method,
        perform_subclustering,
        subcluster_prob_threshold,
    )
    # sage_calculation(best_models, task_type, data_usage, datetime_col, threshold)


def main():

    global_start_time = time.perf_counter()
    global_start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Global execution started at {global_start_timestamp}")

    log_setup_info_func()
    execute_main_tasks()

    global_end_time = time.perf_counter()
    global_end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    global_exec_time = global_end_time - global_start_time
    print(f"Global execution ended at {global_end_timestamp}")

    log_execution_time(
        "Global execution",
        global_exec_time,
        global_start_timestamp,
        global_end_timestamp,
    )


if __name__ == "__main__":
    main()
