import logging
import os


def setup_logger(name, log_file="file.log", level=logging.INFO):
    """Function to set up a logger with a stream handler and a file handler."""

    # Create a custom logger
    logger = logging.getLogger(name)

    # Check if logger has handlers already to avoid duplication
    if not logger.hasHandlers():
        # Set the log level
        logger.setLevel(level)

        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(log_file)

        # Create formatters and add them to the handlers
        c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        f_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

    return logger
