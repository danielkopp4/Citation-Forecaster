import os
from src.shared.utils import load_params
import sys
import opendatasets as od
from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile
api = KaggleApi()
api.authenticate()


# when called downloads and parses the dataset
# for future use in the dataset_api
def download_data(params: dict):
    
    # download the data
    dataset_link = params["data_link"]
    od.download(dataset_link)
    loc = os.path.join(params['data_folder'], params['processed_name'])
    newpath = params['data_folder']
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    os.rename(params["initial_data_location"], loc)

    if not os.path.exists(loc):
        os.mkdir(loc)
    pass


if __name__ == "__main__":
    file_name = sys.argv[1]
    params = load_params(file_name)
    download_data(params)