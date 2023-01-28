from src.shared.utils import load_params
import sys
from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile
api = KaggleApi()
api.authenticate()


# when called downloads and parses the dataset
# for future use in the dataset_api
def download_data(params: dict):
    api.dataset_download_file(params["data_set"], params["json_file"])
    zipFileName = params["json_file"] + ".zip"
    zf = ZipFile(zipFileName)
    zf.extractall()
    zf.close()
    loc = os.path.join(params['data_folder'], params['processed_name'])

    if not os.path.exists(loc):
        os.mkdir(loc)
    pass


if __name__ == "__main__":
    file_name = sys.argv[1]
    params = load_params(file_name)
    download_data(params)