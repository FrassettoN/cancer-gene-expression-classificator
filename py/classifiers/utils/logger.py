import sys
import logging
import os

# Configure the root logger to display INFO messages
logger = logging.getLogger("")
logger.setLevel(logging.INFO)


def log_to_file(msg, file_path):

    # Log a message to both the console and a specified file
    file_handler = logging.FileHandler(file_path, mode="a")
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info(msg)

    logger.removeHandler(file_handler)
    logger.removeHandler(stream_handler)

    file_handler.close()


def suppress_output():
    sys.stdout = open(os.devnull, "w")  # Temporarily suppress standard output


def restore_output():
    sys.stdout = sys.__stdout__  # Restore standard output
