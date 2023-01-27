from src.shared.utils import load_params
import sys


# when called preprocess the dataset
def preprocess_data(params: dict):
    pass

if __name__ == "__main__":
    file_name = sys.argv[1]
    params = load_params(file_name)
    preprocess_data(params)