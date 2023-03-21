from src.shared.utils import load_params
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
import pandas as pd
import numpy as np
import sys

# data we need
# title, abstract, journal, and add in citation # to the csv
# title -> sbert
# abstract -> sbert
# journal -> one_hot
# citation -> identity

def load_paper_data(params: dict) -> pd.DataFrame:
    return pd.DataFrame({
        'title': ['titles', 'c', 'a', 'titles', 'c', 'a', 'titles', 'c', 'a', 'titles'],
        'abstract': ['an abstract', 'b', 'f', 'an abstract', 'b', 'f', 'an abstract', 'b', 'f', 'f'],
        'journal': ['Nature', 'random', 'other', 'Nature', 'random', 'other', 'Nature', 'random', 'other', 'a']
    })


def load_citations(params: dict) -> np.ndarray:
    return np.arange(10) # [0,1,2,3,4,5,6,7,8,9]


def save_data(location: str, data: np.ndarray):
    np.save(location + '.npy', data)


def journal_pre_transform(journal: str) -> str:
    return journal.lower()


# when called preprocess the dataset
def preprocess_data(params: dict):
    print('load papers and citations')
    paper_df = load_paper_data(params)
    citations = load_citations(params)
    # not found citations should be na
    assert len(paper_df) == len(citations)

    print('get SBERT model')
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print('transform journal')
    journals = paper_df['journal'].transform(journal_pre_transform)
    journals = OneHotEncoder(sparse=False).fit_transform(journals.values.reshape(-1, 1))

    print('transform titles')
    titles = model.encode(paper_df['title'].tolist())
    
    print('transform abstracts')
    abstracts = model.encode(paper_df['abstract'].tolist())

    print('concatenate')
    final_data = np.array([np.concatenate((
        title, 
        abstract, 
        journal, 
        [citation]))
        for title, abstract, journal, citation in zip(titles, abstracts, journals, citations)
    ])

    print('shuffle and split')
    np.random.seed(params['random_seed'])
    np.random.shuffle(final_data)

    length = len(final_data)
    train_index = int(length * params['train_percent'])
    val_index = train_index + int(length * params['val_percent'])

    train_data = final_data[:train_index]
    val_data = final_data[train_index:val_index]
    test_data = final_data[val_index:]

    print('scale')
    scaler = StandardScaler()
    train_data[:,:-1] = scaler.fit_transform(train_data[:,:-1])
    val_data[:,:-1] = scaler.transform(val_data[:,:-1])
    test_data[:,:-1] = scaler.transform(test_data[:,:-1])

    print('save')
    loc = os.path.join(params['data_folder'], params['processed_name'])

    if not os.path.exists(loc):
        os.mkdir(loc)

    save_data(os.path.join(loc, 'train'), train_data)
    save_data(os.path.join(loc, 'val'), val_data)
    save_data(os.path.join(loc, 'test'), test_data)


if __name__ == '__main__':
    file_name = sys.argv[1]
    params = load_params(file_name)
    preprocess_data(params)
