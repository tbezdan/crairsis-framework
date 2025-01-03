from datetime import datetime
from utils.config import ROOT
import os


def log_setup_info(
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
):
    setup_info = {
        "user_name": user_name,
        "num_epochs": num_epochs,
        "population_size": population_size,
        "targets": targets,
        "mh_algorithms": mh_algorithms,
        "dimensionality_reduction_method": dimensionality_reduction_method,
        "perform_subclustering": perform_subclustering,
        "subcluster_prob_threshold": subcluster_prob_threshold,
        "filter_col": filter_col,
        "task_type": task_type,
        "data_usage": data_usage,
        "datetime_col": datetime_col,
        "threshold": threshold,
        "model_registry": list(model_registry.keys()),
    }

    # Get the current date and time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Log setup information in a more readable format
    log_message = (
        f"\nExecution Setup Information\n"
        f"------------------------------------------------------\n"
        f"Date and Time:                    {current_time}\n"
        f"User Name:                        {setup_info['user_name']}\n"
        f"Task Type:                        {setup_info['task_type']}\n"
        f"Data Usage:                       {setup_info['data_usage']}\n"
        f"Number of Epochs:                 {setup_info['num_epochs']}\n"
        f"Population Size:                  {setup_info['population_size']}\n"
        f"SAGE Threshold:                   {setup_info['threshold']}\n"
        f"Datetime Column:                  {setup_info['datetime_col']}\n"
        f"Filter Column:                    {setup_info['filter_col']}\n"
        f"Dimensionality Reduction Method:  {setup_info['dimensionality_reduction_method']}\n"
        f"Subclustering:                    {setup_info['perform_subclustering']}\n"
        f"Subcluster Probability Threshold: {setup_info['subcluster_prob_threshold']}\n"
    )

    # Add targets
    log_message += "\nTargets:\n"
    for target in setup_info["targets"]:
        log_message += f"    - {target}\n"

    # Add metaheuristic algorithms
    log_message += "\nMetaheuristic Algorithms:\n"
    for algo in setup_info["mh_algorithms"]:
        log_message += f"    - {algo}\n"

    # Add ML models
    log_message += "\nML Models:\n"
    for model in setup_info["model_registry"]:
        log_message += f"    - {model}\n"

    # Log the message to the console and to the file
    print(log_message)

    with open("execution_times.log", "a") as log_file:
        log_file.write(log_message + "\n")

    # Experiment-specific log in output_path
    log_file_path = os.path.join(ROOT, "execution_times.log")
    with open(log_file_path, "w") as log_file:
        log_file.write(log_message + "\n")

    return setup_info
