from requests import get
from threading import Thread
import numpy as np
from fp.fp import FreeProxy
import time
from typing import List
from requests.packages.urllib3.exceptions import MaxRetryError
from requests.packages.urllib3.exceptions import ProxyError as urllib3_ProxyError
from requests.exceptions import ConnectionError
import traceback
from src.shared.utils import load_params
import opendatasets as od
import os
import sys
import shutil
from typing import List
import json

max_cutoff = 100
batch = 50
interval = 0.01


def get_dois(file_name: str) -> List[str]:
    dois = []
    n_points = 0
    with open(file_name) as json_file:
        line = json_file.readline()
        while line and n_points < max_cutoff:
            json_data = json.loads(line)
            if 'doi' not in json_data:
                dois.append(None)
            else:
                dois.append(json_data['doi'])
                n_points += 1

            line = json_file.readline() 

    return dois

# when called downloads and parses the dataset
# for future use in the dataset_api
def download_data(params: dict):
    loc = os.path.join(params['data_folder'], params['dataset_folder'])

    # dataset_link = params["data_link"]
    # od.download(dataset_link, data_dir=loc)

    # shutil.move(
    #     os.path.join(loc, params['output_file']), 
    #     os.path.join(loc, params['target_name'])
    # )

    file_name = [x for x in os.listdir(loc) if ".json" in x][0]
    
    # dois = ["10.1103/PhysRevD.76.013009" for _ in range(iters)]
    dois = get_dois(os.path.join(loc, file_name))
    complete = np.zeros((len(dois),))#.astype(np.bool_)

    for id in range(0, len(dois), batch):
        start_thread(id, dois[id:id+batch], complete)
        time.sleep(interval)

    while np.any(complete == False):
        time.sleep(1)
        print(complete)

    print("req per second", (time.time() - prev_time) / len(dois))




def get_citation(id: int, doi: List[str], complete: np.ndarray) -> None:
    proxy = FreeProxy().get()
    proxies = {
        "http": proxy
    }
    # API_CALL = "https://opencitations.net/index/api/v1/citation-count/{}".format(doi)

    
    # print(doi)
    for i, curDoi in enumerate(doi):
        if curDoi == None:
            complete[id+i] = -100
            continue

        # print(complete)
        
        url = f"http://api.crossref.org/works/{curDoi}"
        # try:

        success = False
        
        while not success:
            citationNumber = get(url, proxies=proxies)
            # print(citationNumber.status_code)
            # print(url)
            # print("got rate limited")
            # print(citationNumber.raw)
            proxy = FreeProxy().get()
            proxies = {
                "http": proxy
            }
            try:
                citationNumber = get(url, proxies=proxies)

                if citationNumber.status_code == 200:
                    success = True
            except Exception:
                pass

        try:
            count = int(citationNumber.json()['message']['reference-count'])
            complete[id+i] = count
        

        # except ConnectionError:
        #     print("received connection error")
        #     traceback.print_exc(file=sys.stdout)
        #     complete[id+i] = True
        #     continue
        except Exception:
            print("received unhandled error")
            print(citationNumber.text)
            print(url)
            traceback.print_exc(file=sys.stdout)
            complete[id+i] = True
            continue
        

def start_thread(id, doi, complete):
    Thread(target=get_citation, args=(id, doi, complete)).start()



prev_time = time.time()




if __name__ == '__main__':
    file_name = sys.argv[1]
    params = load_params(file_name)
    download_data(params)