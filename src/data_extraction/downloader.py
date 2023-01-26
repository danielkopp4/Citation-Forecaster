from src.shared.utils import load_params
import sys


# when called downloads and parses the dataset
# for future use in the dataset_api
def download_data(params: dict):
    pass

if __name__ == "__main__":
    file_name = sys.argv[1]
    params = load_params(file_name)
    download_data(params)