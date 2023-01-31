from src.model.model import get_model
from src.shared.utils import load_params
from contextlib import redirect_stderr, redirect_stdout
import traceback
import sys
import os

def train(params: dict):
    model = get_model(params)
    model.print_params()
    model.train()

if __name__ == '__main__':
    file_name = sys.argv[1]
    params = load_params(file_name)
    log_dir = params['log_dir']

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_dir = os.path.join(log_dir, params['model_name'])

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_file = os.path.join(log_dir, "log_file.txt")
    err_file = os.path.join(log_dir, "errors.txt")

    with open(err_file, 'w') as error_log:
        with redirect_stderr(error_log):
            with open(log_file, 'w') as log:
                with redirect_stdout(log):
                    try:
                        train(params)
                    except Exception as e:
                        traceback.print_exc(file=sys.stderr)
