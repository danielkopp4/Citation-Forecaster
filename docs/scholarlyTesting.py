from requests import get
from threading import Thread
import numpy as np
from fp.fp import FreeProxy
import time


iters = 400
batch = 40
interval = 2
complete = np.zeros((iters,)).astype(np.bool_)

def get_citation(id, doi):
    proxy = FreeProxy().get()
    proxies = {
        "http": proxy
    }
    # API_CALL = "https://opencitations.net/index/api/v1/citation-count/{}".format(doi)
    url = f"http://api.crossref.org/works/{doi}"
    try:
        citationNumber = get(url, proxies=proxies)
    except:
        print("proxy didnt work")
        complete[id] = True
        return
   





# [start_thread(id, "10.1103/PhysRevD.76.013009") for id in range(iters)]
# [start_thread(id, "10.1103/PhysRevD.76.013009") for id in range(iters)]

dois = ["10.1103/PhysRevD.76.013009" for _ in range(iters)]
for id, doi in enumerate(dois):
    if id % batch == 0:
        time.sleep(interval)

    start_thread(id, doi)




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