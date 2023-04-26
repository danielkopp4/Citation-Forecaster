from requests import get
from threading import Thread
import numpy as np
from fp.fp import FreeProxy
import time
from typing import List
from requests.exceptions import ConnectionError
import traceback
from src.shared.utils import load_params
import opendatasets as od
import os
import sys
import shutil
from typing import List
import json
from datetime import timedelta
import signal

max_cutoff = np.inf
batch = 100
interval = 2

'''
gets the dois of the papers which will then be used as a unique identifier
to then be able to get a current citation from Open Citations
'''
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
'''
where the data will be stored
'''
def complete_filename(params: dict) -> str:
    return os.path.join(os.path.join(params['data_folder'], params['dataset_folder']), params['doi_tempfile'])

'''
loads the data to then bne able to get modified
'''
def load_complete(params: dict, default_len: int) -> np.ndarray:
    # if not found return empty nd_arr
    filename = complete_filename(params)
    if os.path.exists(filename):
        with open('lines.txt', 'r') as file:
            lines = int(file.readline())

        arr = np.loadtxt(filename)
        if len(arr) != lines:
            print('ERR LENGTH NOT EQUAL')
            sys.exit(0)

    return np.zeros((default_len,)).astype(int)

'''
checks to see if saving is possible then saves to the file
'''
def save_complete(params: dict, complete: np.ndarray, length: int):
    if len(complete) != length:
        print('ERR ON SAVE: len(complete) != length')
        sys.exit(0)

    np.savetxt(complete_filename(params), complete.astype(int), fmt='%i', delimiter=',')

def get_incomplete(complete: np.ndarray):
    return np.arange(len(complete))[complete == 0]

def iterate_exp_average(current_val: float, moving_average: float, beta_1: float, beta_2: float, iters: int) -> float:
    movement = np.log(beta_1) / np.log(beta_2)
    adj_beta = -np.power(beta_2, iters + movement) + beta_1
    return adj_beta * moving_average + (1 - adj_beta) * current_val

'''
downloads and parses the dataset from arXiv and uses multithreading / proxies to get the citations
'''
def download_data(params: dict):
    '''
    arxiv download part
    '''
    loc = os.path.join(params['data_folder'], params['dataset_folder'])
    files = [x for x in os.listdir(loc) if '.json' in x]

    if len(files) == 0:
        dataset_link = params['data_link']
        od.download(dataset_link, data_dir=loc)

        shutil.move(
            os.path.join(loc, params['output_file']), 
            os.path.join(loc, params['target_name'])
        )
        files = [x for x in os.listdir(loc) if '.json' in x]

    file_name = files[0]
    
    dois = get_dois(os.path.join(loc, file_name))
    length = len(dois)
    complete = load_complete(params, len(dois))
    incomplete = get_incomplete(complete)

    id = 0
    rate = 0
    prev_time = time.time()
    prev_n_remaining = np.sum([1 if x == 0 else 0 for x in complete])
    iters = 0

    def sigterm_handler(signal, frame):
        save_complete(params, complete, length)
        sys.exit(0)

    # print(f'-- pid: {os.getpid()} --')
    with open('pid.txt', 'w') as file:
        file.write(f'PID: {os.getpid()}\n')
    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGINT, sigterm_handler)

    '''
    starts the multithreading process to then be able to get the citations in a parallel methedology
    '''
    while np.any(complete == False):
        if id < len(dois):
            start_thread(incomplete[id:id+batch], dois, complete)
            id += batch
        time.sleep(interval)

        delta = time.time() - prev_time
        prev_time = time.time()
        n_remaining = np.sum([1 if x == 0 else 0 for x in complete])
        delta_n = prev_n_remaining - n_remaining
        prev_n_remaining = n_remaining
        current_rate = delta_n / delta

        rate = iterate_exp_average(current_rate, rate, 0.97, 0.5, iters)
        iters += 1
        time_remaining = n_remaining / rate if rate > 1E-5 else np.inf
        time_remaining = timedelta(seconds=int(time_remaining)) if not np.isinf(time_remaining) else 'inf'

        print('\033[Kn remaining: {} | {:0.03f} req/s | {} seconds'.format(n_remaining, rate, time_remaining), end='\r')
        save_complete(params, complete, length)

    save_complete(params, complete, length)
    print('req per second',  len(dois) / (time.time() - prev_time))
    with open('status.txt', 'w') as file:
        file.write('DONE')



'''
gets the citations by sneding a reques to the OpenCitations url using a list of dois
'''
def get_citation(ids: List[int], dois: List[str], complete: np.ndarray) -> None:
    proxy = FreeProxy().get()
    proxies = {
        'http': proxy
    }
    # API_CALL = 'https://opencitations.net/index/api/v1/citation-count/{}'.format(doi)

    
    for id in ids:
        curDoi = dois[id]
        if curDoi == None:
            complete[id] = -100
            continue

        
        url = f'http://api.crossref.org/works/{curDoi}'

        success = False
        '''
        if not sucessfull then need to change the proxy to then be able to get aroudn the rate limiting
        '''
        while not success:
            # rand_ip = socket.inet_ntoa(struct.pack('>I', random.randint(1, 0xffffffff)))
            rand_ip = '127.0.0.1'
            headers = [[y.strip() for y in x.strip().split(':')] for x in f'''
            X-Originating-IP: {rand_ip}
            X-Forwarded-For: {rand_ip}
            X-Remote-IP: {rand_ip}
            X-Remote-Addr: {rand_ip}
            X-Client-IP: {rand_ip}
            X-Host: {rand_ip}
            X-Forwared-Host: {rand_ip}
            '''.strip().split('\n')]
            headers = {x: y for x,y in headers}

            try:
                citationNumber = get(url, headers=headers, proxies=proxies, timeout=100)

                if citationNumber.status_code == 200:
                    success = True
            except ConnectionError:
                pass
            except Exception:
                print('received unhandled error')
                print(url)
                traceback.print_exc(file=sys.stdout)
                sys.exit()

            if not success:
                proxy = FreeProxy().get()
                proxies = {
                    'http': proxy
                }

        try:
            count = int(citationNumber.json()['message']['reference-count'])
            complete[id] = count + 1 # important -> subtract by 1 when loading
        except Exception:
            print('received unhandled error')
            print(citationNumber.text)
            print(url)
            traceback.print_exc(file=sys.stdout)
        
'''
multithreading base logic
'''
def start_thread(ids, dois, complete):
    return Thread(target=get_citation, args=(ids, dois, complete), daemon=True).start()



prev_time = time.time()

if __name__ == '__main__':
    file_name = sys.argv[1]
    params = load_params(file_name)
    download_data(params)