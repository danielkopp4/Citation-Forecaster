import logging
from src.model.model import get_model
from src.shared.utils import load_params
from contextlib import redirect_stderr, redirect_stdout
import traceback
import sys
import os
from datetime import datetime

def train(params: dict):
    """
    Function to initiate model training based on provided parameters.

    Args:
    - params (dict): Parameters required for configuring and training the model.
    """
    # Initialize the logger
    logging.basicConfig(filename='training_log.txt', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    model = get_model(params)
    model.print_params()

    # Set up logging for loss and accuracy metrics
    logging.info("Training started...")

    model.train()

    logging.info("Training completed.")

if __name__ == '__main__':
    # Fetching parameter file name from command line argument
    file_name = sys.argv[1]
    params = load_params(file_name)
    log_dir = params['log_dir']

    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_dir = os.path.join(log_dir, params['model_name'])

    # Create model-specific log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_file = os.path.join(log_dir, 'log_file.txt')
    err_file = os.path.join(log_dir, 'errors.txt')

    # Redirect stdout and stderr to log files for logging
    with open(err_file, 'w') as error_log:
        with redirect_stderr(error_log):
            with open(log_file, 'w') as log:
                with redirect_stdout(log):
                    try:
                        # Initiate model training
                        train(params)
                    except Exception as e:
                        # Handle exceptions by logging them with timestamp
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        error_msg = f"{now} - Error occurred during training: {str(e)}\n"
                        error_log.write(error_msg)
                        traceback.print_exc(file=error_log)
