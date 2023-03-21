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
import random
import socket
import struct

max_cutoff = 20
batch = 2
interval = 2


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

def complete_filename(params: dict) -> str:
    return os.path.join(os.path.join(params['data_folder'], params['dataset_folder']), params['doi_tempfile'])

def load_complete(params: dict, default_len: int) -> np.ndarray:
    # if not found return empty nd_arr

    filename = complete_filename(params)
    if os.path.exists(filename):
        return np.loadtxt(filename)

    return np.zeros((default_len,)).astype(int)

def save_complete(params: dict, complete: np.ndarray):
    np.savetxt(complete_filename(params), complete.astype(int), fmt='%i', delimiter=",")

def get_incomplete(complete: np.ndarray):
    return np.arange(len(complete))[complete == 0]

# when called downloads and parses the dataset
# for future use in the dataset_api
def download_data(params: dict):
    loc = os.path.join(params['data_folder'], params['dataset_folder'])
    files = [x for x in os.listdir(loc) if ".json" in x]

    if len(files) == 0:
        dataset_link = params["data_link"]
        od.download(dataset_link, data_dir=loc)

        shutil.move(
            os.path.join(loc, params['output_file']), 
            os.path.join(loc, params['target_name'])
        )
        files = [x for x in os.listdir(loc) if ".json" in x]

    file_name = files[0]
    
    dois = get_dois(os.path.join(loc, file_name))
    complete = load_complete(params, len(dois))
    incomplete = get_incomplete(complete)

    id = 0
    while np.any(complete == False):
        if id < len(dois):
            start_thread(incomplete[id:id+batch], dois, complete)
            id += batch
        time.sleep(interval)
        print("\033[Kn remaining: {}".format(np.sum([1 if x == 0 else 0 for x in complete])), end="\r")
        save_complete(params, complete)

    print("req per second",  len(dois) / (time.time() - prev_time))




def get_citation(ids: List[int], dois: List[str], complete: np.ndarray) -> None:
    proxy = FreeProxy().get()
    proxies = {
        "http": proxy
    }
    # API_CALL = "https://opencitations.net/index/api/v1/citation-count/{}".format(doi)

    
    # print(doi)
    for id in ids:
        curDoi = dois[id]
        if curDoi == None:
            complete[id] = -100
            continue

        # print(complete)
        
        url = f"http://api.crossref.org/works/{curDoi}"

        success = False
        
        while not success:
            # rand_ip = socket.inet_ntoa(struct.pack('>I', random.randint(1, 0xffffffff)))
            rand_ip = "127.0.0.1"
            headers = [[y.strip() for y in x.strip().split(":")] for x in f"""
            X-Originating-IP: {rand_ip}
            X-Forwarded-For: {rand_ip}
            X-Remote-IP: {rand_ip}
            X-Remote-Addr: {rand_ip}
            X-Client-IP: {rand_ip}
            X-Host: {rand_ip}
            X-Forwared-Host: {rand_ip}
            """.strip().split("\n")]
            headers = {x: y for x,y in headers}




            # citationNumber = get(url, proxies=proxies, headers=headers)
            # print(citationNumber.status_code)
            # print(url)
            # print("got rate limited")
            # print(citationNumber.raw)
            
            try:
                # citationNumber = get(url, proxies=proxies, timeout=10)
                citationNumber = get(url, headers=headers, proxies=proxies, timeout=100)

                if citationNumber.status_code == 200:
                    success = True
            except ConnectionError:
                # print("received connection error")
                # traceback.print_exc(file=sys.stdout)
                # complete[id+i] = 1
                pass
            except Exception:
                print("received unhandled error")
                print(url)
                traceback.print_exc(file=sys.stdout)
                # complete[id] = 0
                sys.exit()

            if not success:
                proxy = FreeProxy().get()
                proxies = {
                    "http": proxy
                }

        try:
            count = int(citationNumber.json()['message']['reference-count'])
            complete[id] = count + 1
        

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
            # complete[id] = 0
        

def start_thread(ids, dois, complete):
    # print("starting thread", id, doi)
    Thread(target=get_citation, args=(ids, dois, complete)).start()



prev_time = time.time()




if __name__ == '__main__':
    file_name = sys.argv[1]
    params = load_params(file_name)
    download_data(params)