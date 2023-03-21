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
    search_query = scholarly.search_author('Steven A Cholewiak')





if __name__ == '__main__':
    file_name = sys.argv[1]
    params = load_params(file_name)
    download_data(params)