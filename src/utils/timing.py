from utils.config import ROOT
import os


def format_time(seconds):
    days = seconds // (24 * 3600)
    seconds %= 24 * 3600
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return int(days), int(hours), int(minutes), seconds


def log_execution_time(
    function_name, execution_time_seconds, start_timestamp, end_timestamp
):
    days, hours, minutes, seconds = format_time(execution_time_seconds)

    log_message = (
        f"{function_name} started at {start_timestamp} and ended at {end_timestamp}, "
        f"executed in {days}d {hours}h {minutes}m {seconds:.4f}s"
    )

    with open("execution_times.log", "a") as log_file:
        log_file.write(log_message + f" ({execution_time_seconds:.4f} seconds)\n")

    # Experiment-specific log in output_path
    log_file_path = os.path.join(ROOT, "execution_times.log")
    with open(log_file_path, "a") as log_file:
        log_file.write(log_message + f" ({execution_time_seconds:.4f} seconds)\n")
