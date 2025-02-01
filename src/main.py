from utils.train_and_optimize import perform_training_and_optimization
from xai.shap_analysis import perform_shap_analysis, perform_shap_interactions_analysis
from xai.nshap_analysis import perform_nshap_analysis
from xai.gshap_analysis import perform_gshap_analysis
from xai.sage_analysis import perform_sage_analysis
from xai.shap_clustering import perform_shap_clustering
from xai.isage_analysis import perform_isage_analysis
from utils.best_eval import perform_best_models_evaluation
from utils.logger import setup_logger
from datetime import datetime
from utils.folder_config import configure_paths


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
from ml_models.balanced_random_forest_classification_model import (
    BalancedRandomForestCls,
)


from ml_models.adaboost_regression_model import AdaBoostRegressionModel
from ml_models.lightgbm_regression_model import LGBMRegressionModel
from ml_models.xgboost_regression_model import XGBRegressionModel
from ml_models.extratrees_regression_model import ExtraTreesRegressionModel
from ml_models.gradientboosting_regression_model import GradientBoostingRegressionModel
from ml_models.histgradientboosting_regression_model import (
    HistGradientBoostingRegressionModel,
)


from utils.setup_logger import log_setup_info


logger = setup_logger(__name__)

# targets = ["PM2_5_Aerosol_PM2_5_6001"]
# ["cls", "crai"]
# task_type = "regression"  # "classification"  # regression
# "train_and_test"  # "test" # two possible options: train_and_test, test
# "Datetime" #None
# two possible options:"PaCMAP", "UMAP", default "PaCMAP"
# if perform_subclustering True, the user can specify the value of subcluster_prob_threshold
# intergroup_difference_column_name = "HouseAge"
# independent_vars = ["AveRooms", "AveBedrms"]


ROOT = r"C:\Users\tbezdan\Desktop\crAIRsis datasets\cls_test"
paths = configure_paths(ROOT)


original_datasets_path = paths["original_datasets_path"]
best_models_path = paths["best_models_path"]
output_path = paths["output_path"]
paths_to_check = paths["paths_to_check"]
models_path = paths["models_path"]
datasets_path = paths["datasets_path"]
actual_predicted_folder = paths["actual_predicted_folder"]
json_path = paths["json_path"]
original_data_id_folder = paths["original_data_id_folder"]
gshap_folder = paths["gshap_folder"]
isage_folder = paths["isage_folder"]
nshap_folder = paths["nshap_folder"]
sage_folder = paths["sage_folder"]
shap_folder = paths["shap_folder"]
top_sage_features_models_path = paths["top_sage_features_models_path"]
interactions_folder = paths["interactions_folder"]
interactions_feature_folder = paths["interactions_feature_folder"]
shap_clusters_folder = paths["shap_clusters_folder"]
shap_subclusters_folder = paths["shap_subclusters_folder"]


user_name = "timea_bezdan"
num_epochs = 2
population_size = 5
targets = ["HouseAgeBinaryCat"]
mh_algorithms = ["SCA"]
filter_col = None
task_type = "classification"
data_usage = "test"
datetime_col = None
threshold = 90
dimensionality_reduction_method = "PaCMAP"
perform_subclustering = True

if perform_subclustering:
    subcluster_prob_threshold = 0.6
else:
    subcluster_prob_threshold = None


##########################################################################
# gshap params
##########################################################################
# mediation
gshap_mediation_independent_vars = ["Latitude", "Longitude"]
# intergroup difference
gshap_intergroup_difference_column_name = "MedInc"
gshap_intergroup_difference_selected_values = None
gshap_intergroup_difference_grouping_method = "quantile"
gshap_intergroup_difference_grouping_value = 75


# hypothesis testing
gshap_hypothesis_testing_hypothesis_treshold_method = "input"
gshap_hypothesis_testing_hypothesis_treshold_value = 2

gshap_hypothesis_testing_sample_treshold_method = "input"
gshap_hypothesis_testing_sample_treshold_value = 1.5


gshap_hypothesis_testing_condition = "greater"

##########################################################################


tasks_to_execute = [
    # "train_and_optimize_models",
    # "shap_calculation",
    # "shap_interaction",
    # "shap_clustering",
    # "sage_calculation",
    # "nshap_calculation",
    "gshap_calculation",
    "isage_calculation",
]

ml_models = [
    "BalancedRandomForestCls",
    "LGBMModel",
    "ExtraTreesModel",
    "HistGradientBoostingCls",
]

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

model_registry_temp = (
    classification_model_registry
    if task_type == "classification"
    else regression_model_registry
)

model_registry = {
    model_name: model_func
    for model_name, model_func in model_registry_temp.items()
    if model_name in ml_models
}


def log_setup_info_func():

    # model_registry = (
    #     classification_model_registry
    #     if task_type == "classification"
    #     else regression_model_registry
    # )

    log_setup_info(
        ROOT,
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

    create_folders()

    train_optimize_start_time = time.perf_counter()
    train_optimize_start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"train_and_optimize_models started at {train_optimize_start_timestamp}")

    perform_training_and_optimization(
        models_path,
        output_path,
        json_path,
        original_data_id_folder,
        original_datasets_path,
        datasets_path,
        num_epochs,
        population_size,
        targets,
        mh_algorithms,
        task_type,
        filter_col,
        datetime_col,
        model_registry,
    )

    format_best_models(output_path, task_type=task_type)
    format_detailed_metrics(output_path, task_type)
    best_models = pd.read_csv(best_models_path)
    evaluate_best_models(best_models, task_type, data_usage, datetime_col)

    train_optimize_end_time = time.perf_counter()
    train_optimize_end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    train_optimize_exec_time = train_optimize_end_time - train_optimize_start_time
    print(f"train_and_optimize_models ended at {train_optimize_end_timestamp}")

    log_execution_time(
        ROOT,
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
        best_models,
        filter_col,
        task_type,
        data_usage,
        datetime_col,
        models_path,
        datasets_path,
        actual_predicted_folder,
    )

    evaluate_end_time = time.perf_counter()
    evaluate_end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    evaluate_exec_time = evaluate_end_time - evaluate_start_time
    print(f"evaluate_best_models ended at {evaluate_end_timestamp}")

    log_execution_time(
        ROOT,
        "evaluate_best_models",
        evaluate_exec_time,
        evaluate_start_timestamp,
        evaluate_end_timestamp,
    )


def shap_calculation(task_type, data_usage, datetime_col):
    shap_start_time = time.perf_counter()
    shap_start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"shap_calculation started at {shap_start_timestamp}")
    best_models = pd.read_csv(best_models_path)
    perform_shap_analysis(
        shap_folder,
        interactions_folder,
        models_path,
        datasets_path,
        interactions_feature_folder,
        best_models,
        filter_col,
        task_type,
        data_usage,
        datetime_col,
    )

    shap_end_time = time.perf_counter()
    shap_end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    shap_exec_time = shap_end_time - shap_start_time
    print(f"shap_calculation ended at {shap_end_timestamp}")
    log_execution_time(
        ROOT,
        "shap_calculation",
        shap_exec_time,
        shap_start_timestamp,
        shap_end_timestamp,
    )


def shap_interaction(task_type, data_usage, datetime_col):
    shap_interactions_start_time = time.perf_counter()
    shap_interactions_start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"shap_interactions started at {shap_interactions_start_timestamp}")
    best_models = pd.read_csv(best_models_path)
    perform_shap_interactions_analysis(
        interactions_folder,
        models_path,
        datasets_path,
        interactions_feature_folder,
        best_models,
        filter_col,
        task_type,
        data_usage,
        datetime_col,
    )

    shap_interactions_end_time = time.perf_counter()
    shap_interactions_end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    shap_interactions_exec_time = (
        shap_interactions_end_time - shap_interactions_start_time
    )
    print(f"shap_interactions ended at {shap_interactions_end_timestamp}")
    log_execution_time(
        ROOT,
        "shap_interactions",
        shap_interactions_exec_time,
        shap_interactions_start_timestamp,
        shap_interactions_end_timestamp,
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
        shap_clusters_folder,
        shap_subclusters_folder,
        shap_folder,
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
        ROOT,
        "shap_clustering",
        shap_cluster_exec_time,
        shap_cluster_start_timestamp,
        shap_cluster_end_timestamp,
    )


def sage_calculation(task_type, data_usage, datetime_col, threshold):

    sage_start_time = time.perf_counter()
    sage_start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"sage_calculation started at {sage_start_timestamp}")
    best_models = pd.read_csv(best_models_path)
    perform_sage_analysis(
        sage_folder,
        datasets_path,
        models_path,
        best_models,
        filter_col,
        task_type,
        data_usage,
        datetime_col,
        threshold,
    )

    sage_end_time = time.perf_counter()
    sage_end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sage_exec_time = sage_end_time - sage_start_time
    print(f"sage_calculation ended at {sage_end_timestamp}")

    log_execution_time(
        ROOT,
        "sage_calculation",
        sage_exec_time,
        sage_start_timestamp,
        sage_end_timestamp,
    )


def nshap_calculation(task_type, datetime_col):
    nshap_start_time = time.perf_counter()
    nshap_start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"nshap_calculation started at {nshap_start_timestamp}")
    best_models = pd.read_csv(best_models_path)

    # model_registry = (
    #     classification_model_registry
    #     if task_type == "classification"
    #     else regression_model_registry
    # )

    perform_nshap_analysis(
        nshap_folder,
        models_path,
        datasets_path,
        sage_folder,
        shap_folder,
        top_sage_features_models_path,
        best_models,
        filter_col,
        task_type,
        datetime_col,
        model_registry,
    )

    nshap_end_time = time.perf_counter()
    nshap_end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    nshap_exec_time = nshap_end_time - nshap_start_time
    print(f"nshap_calculation ended at {nshap_end_timestamp}")
    log_execution_time(
        ROOT,
        "nshap_calculation",
        nshap_exec_time,
        nshap_start_timestamp,
        nshap_end_timestamp,
    )


def gshap_calculation(task_type, datetime_col):
    gshap_start_time = time.perf_counter()
    gshap_start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"gshap_calculation started at {gshap_start_timestamp}")
    best_models = pd.read_csv(best_models_path)
    # model_registry = (
    #     classification_model_registry
    #     if task_type == "classification"
    #     else regression_model_registry
    # )

    perform_gshap_analysis(
        datasets_path,
        gshap_folder,
        models_path,
        best_models,
        filter_col,
        task_type,
        datetime_col,
        gshap_intergroup_difference_column_name,
        gshap_intergroup_difference_selected_values,
        gshap_intergroup_difference_grouping_method,
        gshap_intergroup_difference_grouping_value,
        gshap_mediation_independent_vars,
        gshap_hypothesis_testing_hypothesis_treshold_method,
        gshap_hypothesis_testing_hypothesis_treshold_value,
        gshap_hypothesis_testing_sample_treshold_method,
        gshap_hypothesis_testing_sample_treshold_value,
        gshap_hypothesis_testing_condition,
        model_registry,
    )

    gshap_end_time = time.perf_counter()
    gshap_end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    nshap_exec_time = gshap_end_time - gshap_start_time
    print(f"gshap_calculation ended at {gshap_end_timestamp}")
    log_execution_time(
        ROOT,
        "gshap_calculation",
        nshap_exec_time,
        gshap_start_timestamp,
        gshap_end_timestamp,
    )


def isage_calculation(task_type, datetime_col):
    isage_start_time = time.perf_counter()
    isage_start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"gshap_calculation started at {isage_start_timestamp}")
    best_models = pd.read_csv(best_models_path)

    perform_isage_analysis(
        isage_folder,
        models_path,
        datasets_path,
        best_models,
        filter_col,
        task_type,
        datetime_col,
    )

    isage_end_time = time.perf_counter()
    isage_end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    isage_exec_time = isage_end_time - isage_start_time
    print(f"isage_calculation ended at {isage_end_timestamp}")
    log_execution_time(
        ROOT,
        "isage_calculation",
        isage_exec_time,
        isage_start_timestamp,
        isage_end_timestamp,
    )


def execute_main_tasks():

    if "train_and_optimize_models" in tasks_to_execute:
        train_and_optimize_models(filter_col, datetime_col)

    if "shap_calculation" in tasks_to_execute:
        shap_calculation(task_type, data_usage, datetime_col)

    if "shap_interaction" in tasks_to_execute:
        shap_interaction(task_type, data_usage, datetime_col)

    if "shap_clustering" in tasks_to_execute:
        shap_clustering(
            task_type,
            dimensionality_reduction_method,
            perform_subclustering,
            subcluster_prob_threshold,
        )

    if "sage_calculation" in tasks_to_execute:
        sage_calculation(task_type, data_usage, datetime_col, threshold)

    if "nshap_calculation" in tasks_to_execute:
        nshap_calculation(task_type, datetime_col)

    if "gshap_calculation" in tasks_to_execute:
        gshap_calculation(task_type, datetime_col)

    if "isage_calculation" in tasks_to_execute:
        isage_calculation(task_type, datetime_col)


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
        ROOT,
        "Global execution",
        global_exec_time,
        global_start_timestamp,
        global_end_timestamp,
    )


if __name__ == "__main__":
    main()
