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

# /Users/timea/miniforge3/envs/crairsis/bin/python "/Users/timea/Documents/Projekti/craAIRsis/Covid BG/src/main.py"

logger = setup_logger(__name__)

num_epochs = 2
population_size = 5

targets = ["m79", "m93"]
mh_algorithms = ["SCA", "HHO"]


def create_folders():
    for path in paths_to_check:
        os.makedirs(path, exist_ok=True)
        print(f"Checked/created: {path}")


def train_and_optimize_models():
    perform_training_and_optimization(
        original_datasets_path, num_epochs, population_size, targets, mh_algorithms
    )


def evaluate_best_models(best_models):
    perform_best_models_evaluation(best_models)


def shap_calculation(best_models):
    perform_shap_analysis(best_models)


def shap_clustering():
    perform_shap_clustering()


def sage_calculation(best_models):
    perform_sage_analysis(best_models)


def main():
    create_folders()
    train_and_optimize_models()
    format_best_models(output_path)
    best_models = pd.read_csv(best_models_path)
    format_detailed_metrics(output_path)
    evaluate_best_models(best_models)
    shap_calculation(best_models)
    perform_shap_clustering()
    sage_calculation(best_models)


if __name__ == "__main__":
    main()
