from requests import get
from threading import Thread
import numpy as np
from fp.fp import FreeProxy
import time
from typing import List


iters = 80
batch = 40
interval = 2
complete = np.zeros((iters,)).astype(np.bool_)

def get_citation(id: int, doi: List[str], proxy: str) -> None:
    proxies = {
        "http": proxy
    }
    # API_CALL = "https://opencitations.net/index/api/v1/citation-count/{}".format(doi)

    for i, curDoi in enumerate(doi):
        print(complete)
        url = f"http://api.crossref.org/works/{curDoi}"
        try:
            citationNumber = get(url, proxies=proxies)
        except ProxyError as e:
            if e == "Cannot connect to proxy.":
                proxy = proxy = FreeProxy().get()
                proxies = {
                    "http": proxy
                }
                

            print("got some sort of error")
            print(e)
            complete[id+i] = True
            continue
        except Exception as e:
            print("got some sort of error")
            print(e)
            complete[id+i] = True
            continue
        # return citationNumber.json()[0]["count"]
        # assert(citationNumber.json()[0]['count'] == 46)
        # print(citationNumber.json())
        if citationNumber.status_code != 200:
            print("got rate limited")
            print(citationNumber.raw)
            complete[id+i] = True
            continue
        count = int(citationNumber.json()['message']['reference-count'])
        if count != 46:
            print('err', count)
        complete[id+i] = True


def start_thread(id, doi, proxy):
    
    Thread(target=get_citation, args=(id, doi, proxy)).start()



prev_time = time.time()





# [start_thread(id, "10.1103/PhysRevD.76.013009") for id in range(iters)]
# [start_thread(id, "10.1103/PhysRevD.76.013009") for id in range(iters)]

dois = ["10.1103/PhysRevD.76.013009" for _ in range(iters)]
for id, doi in enumerate(dois):
    if id % batch == 0:
        proxy = FreeProxy().get()
        start_thread(id, doi[id:id+40], proxy)
        time.sleep(interval)

    




while np.any(complete == False):
    time.sleep(0.1)

print("req per second", (time.time() - prev_time) / iters)

# pg = ProxyGenerator()
# scholarly.pprint(next(search_query))

# # Retrieve the first result from the iterator
# first_author_result = next(search_query)
# scholarly.pprint(first_author_result)

# # Retrieve all the details for the author
# author = scholarly.fill(first_author_result )
# scholarly.pprint(author)

# # Take a closer look at the first publication
# first_publication = author['publications'][0]
# first_publication_filled = scholarly.fill(first_publication)
# scholarly.pprint(first_publication_filled)

# # Print the titles of the author's publications
# publication_titles = [pub['bib']['title'] for pub in author['publications']]
# print(publication_titles)

# # Which papers cited that publication?
# citations = [citation['bib']['title'] for citation in scholarly.citedby(first_publication_filled)]
# print(citations)