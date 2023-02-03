import os
from src.shared.utils import load_params
import sys
import opendatasets as od
from scholarly import scholarly


# when called downloads and parses the dataset
# for future use in the dataset_api
def download_data(params: dict):
    
    # download the data
    dataset_link = params["data_link"]
    loc = os.path.join(params['data_folder'], params['dataset_folder'])
    od.download(dataset_link, data_dir=loc)
    # newpath = params['data_folder']
    # if not os.path.exists(newpath):
    #     os.makedirs(newpath)

    # newpath2 = params['dataset_folder']
    # if not os.path.exists(newpath2):
    #     os.makedirs(newpath2)


    # loc = os.path.join(params['dataset_folder'], params['processed_name'])
    
    # os.rename(params["initial_data_location"], loc)

    # if not os.path.exists(loc):
    #     os.mkdir(loc)






if __name__ == "__main__":
    file_name = sys.argv[1]
    params = load_params(file_name)
    download_data(params)