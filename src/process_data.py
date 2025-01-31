import json
import sys
import os
from utils.folder_config import configure_paths
from utils.setup_logger import log_setup_info
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

logger = setup_logger(__name__)


def create_folders(paths_to_check):
    for path in paths_to_check:
        os.makedirs(path, exist_ok=True)
        print(f"Checked/created: {path}")


def evaluate_best_models(
    best_models,
    filter_col,
    task_type,
    data_usage,
    datetime_col,
    models_path,
    datasets_path,
    actual_predicted_folder,
):
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


def train_and_optimize_models(
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
    paths_to_check,
    best_models_path,
    data_usage,
    actual_predicted_folder,
):

    create_folders(paths_to_check)

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
    evaluate_best_models(
        best_models,
        filter_col,
        task_type,
        data_usage,
        datetime_col,
        models_path,
        datasets_path,
        actual_predicted_folder,
    )

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


def shap_calculation(
    best_models_path,
    task_type,
    data_usage,
    datetime_col,
    shap_folder,
    interactions_folder,
    models_path,
    datasets_path,
    interactions_feature_folder,
    filter_col,
):
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


def shap_interaction(
    interactions_folder,
    models_path,
    datasets_path,
    interactions_feature_folder,
    filter_col,
    task_type,
    data_usage,
    datetime_col,
    best_models_path,
):
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
    shap_clusters_folder,
    shap_subclusters_folder,
    shap_folder,
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


def sage_calculation(
    sage_folder,
    datasets_path,
    models_path,
    filter_col,
    task_type,
    data_usage,
    datetime_col,
    threshold,
    best_models_path,
):

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


def nshap_calculation(
    nshap_folder,
    models_path,
    datasets_path,
    sage_folder,
    shap_folder,
    top_sage_features_models_path,
    filter_col,
    task_type,
    datetime_col,
    model_registry,
    best_models_path,
):
    nshap_start_time = time.perf_counter()
    nshap_start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"nshap_calculation started at {nshap_start_timestamp}")
    best_models = pd.read_csv(best_models_path)

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


def gshap_calculation(
    datasets_path,
    gshap_folder,
    models_path,
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
    best_models_path,
):
    gshap_start_time = time.perf_counter()
    gshap_start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"gshap_calculation started at {gshap_start_timestamp}")
    best_models = pd.read_csv(best_models_path)

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
    print(f"nshap_calculation ended at {gshap_end_timestamp}")
    log_execution_time(
        ROOT,
        "nshap_calculation",
        nshap_exec_time,
        gshap_start_timestamp,
        gshap_end_timestamp,
    )


def isage_calculation(
    isage_folder,
    models_path,
    datasets_path,
    filter_col,
    task_type,
    datetime_col,
    best_models_path,
):
    isage_start_time = time.perf_counter()
    isage_start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"isage_calculation started at {isage_start_timestamp}")
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


def main(data_file_path, ROOT):
    global_start_time = time.perf_counter()
    global_start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

    create_folders(paths_to_check)

    with open(data_file_path, "r") as file:
        data = json.load(file)

    # Extract values from the JSON data
    user_name = data.get("user_name", "")
    num_epochs = data.get("num_epochs")
    population_size = data.get("population_size")
    targets = data.get("targets", [])
    mh_algorithms = data.get("mh_algorithms", [])
    filter_col = data.get("filter_col", "")
    filter_col = None if not filter_col else filter_col
    task_type = data.get("task_type", "")
    data_usage = data.get("data_usage", "")
    datetime_col = data.get("datetime_col", "")
    datetime_col = None if not datetime_col else datetime_col[0]
    threshold = data.get("sage_treshold")
    dimensionality_reduction_method = data.get("dimensionality_reduction_method", "")
    perform_subclustering = data.get("perform_subclustering")
    subcluster_prob_threshold = data.get("subcl_treshold")
    ml_models = data.get("ml_models", [])
    tasks_to_execute = data.get("tasks_to_execute", [])
    ##########################################################################
    # gshap params
    ##########################################################################
    # mediation
    gshap_mediation_independent_vars = ["Latitude", "Longitude"]
    # intergroup difference
    gshap_intergroup_difference_column_name = "HouseAge"
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

    if "train_and_optimize_models" in tasks_to_execute:
        train_and_optimize_models(
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
            paths_to_check,
            best_models_path,
            data_usage,
            actual_predicted_folder,
        )

    if "shap_calculation" in tasks_to_execute:
        shap_calculation(
            best_models_path,
            task_type,
            data_usage,
            datetime_col,
            shap_folder,
            interactions_folder,
            models_path,
            datasets_path,
            interactions_feature_folder,
            filter_col,
        )
    if "shap_interaction" in tasks_to_execute:
        shap_interaction(
            interactions_folder,
            models_path,
            datasets_path,
            interactions_feature_folder,
            filter_col,
            task_type,
            data_usage,
            datetime_col,
            best_models_path,
        )
    if "shap_clustering" in tasks_to_execute:
        shap_clustering(
            shap_clusters_folder,
            shap_subclusters_folder,
            shap_folder,
            task_type,
            dimensionality_reduction_method,
            perform_subclustering,
            subcluster_prob_threshold,
        )
    if "sage_calculation" in tasks_to_execute:
        sage_calculation(
            sage_folder,
            datasets_path,
            models_path,
            filter_col,
            task_type,
            data_usage,
            datetime_col,
            threshold,
            best_models_path,
        )
    if "nshap_calculation" in tasks_to_execute:
        nshap_calculation(
            nshap_folder,
            models_path,
            datasets_path,
            sage_folder,
            shap_folder,
            top_sage_features_models_path,
            filter_col,
            task_type,
            datetime_col,
            model_registry,
            best_models_path,
        )
    if "gshap_calculation" in tasks_to_execute:
        gshap_calculation(
            datasets_path,
            gshap_folder,
            models_path,
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
            best_models_path,
        )
    if "isage_calculation" in tasks_to_execute:
        isage_calculation(
            isage_folder,
            models_path,
            datasets_path,
            filter_col,
            task_type,
            datetime_col,
            best_models_path,
        )
    # Print the values or use them as needed
    # print("user_name:", user_name)
    # print("num_epochs:", num_epochs)
    # print("population_size:", population_size)
    # print("targets:", targets)
    # print("mh_algorithms:", mh_algorithms)
    # print("filter_col:", filter_col)
    # print("task_type:", task_type)
    # print("data_usage:", data_usage)
    # print("datetime_col:", datetime_col)
    # print("threshold:", threshold)
    # print("dimensionality_reduction_method:", dimensionality_reduction_method)
    # print("perform_subclustering:", perform_subclustering)
    # print("subcluster_prob_threshold:", subcluster_prob_threshold)
    print("ml_models:", ml_models)
    print("ROOT", ROOT)

    global_end_time = time.perf_counter()
    global_end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    global_exec_time = global_end_time - global_start_time

    log_execution_time(
        ROOT,
        "Global execution",
        global_exec_time,
        global_start_timestamp,
        global_end_timestamp,
    )


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python process_data.py <data_file_path>")
        sys.exit(1)

    data_file_path = sys.argv[1]
    ROOT = sys.argv[2]

    main(data_file_path, ROOT)
