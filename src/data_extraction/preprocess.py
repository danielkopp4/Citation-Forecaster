from src.shared.utils import load_params
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import os
import pandas as pd
import numpy as np
import sys
from sqlalchemy import create_engine, Column, Integer, String, Engine, text, inspect
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

max_cutoff = np.inf
# max_cutoff = 1000
target_type = np.float32

# data we need
# title, abstract, journal, and add in citation # to the csv
# title -> sbert
# abstract -> sbert
# journal -> one_hot (no longer in use -> too many journal names)
# date (string) -> epoch time (int)
# citation -> identity

'''
creates a dataframe with all the data extracted from the .json file
sets date to null if not found
'''
def load_paper_data(engine, params: dict) -> pd.DataFrame:
    command = text(f"SELECT doi, title, abstract, journal, cast(date as date), citationcount FROM {params['paper_metadata_table']} mt join {params['citation_count_table']} c on mt.doi = c.doi;")
    print("executing:", command)
    with engine.connect() as conn:
        r = conn.execute(command)
    
    return pd.DataFrame(r, columns=["doi", "title", "abstract", "journal", "date", "citation_count"])
    

def save_data(location: str, data: np.ndarray):
    np.save(location + '.npy', data)


def journal_pre_transform(journal: str) -> str:
    if journal == None:
        return None

    return journal.lower().strip()

# when called preprocess the dataset
'''
uses sbert to then be able to quantify the words and then assign a score on the abstract as a whole
the print statenents offer a good description of what is going on
'''
def preprocess_data(params: dict):
    print('load papers and citations')
    paper_df = load_paper_data(params)
    # citations = load_citations(params)

    if max_cutoff != np.inf:
        citations = citations[:max_cutoff]

    # not found citations should be na
    assert len(paper_df) == len(citations)

    print(f'loaded {len(paper_df)} papers')

    print('get SBERT model')
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # include for journal preprocessing
    # currently journals with 
    # print('transform journal')
    journals = paper_df['journal'].transform(journal_pre_transform)
    journals = OrdinalEncoder().fit_transform(journals)

    # print('transform date')
    # dates = paper_df['date'].transform(format_date)
    dates = paper_df['date']

    print('transform titles')
    titles = model.encode(paper_df['title'].tolist())
    
    print('transform abstracts')
    abstracts = model.encode(paper_df['abstract'].tolist())

    
    print('find non-nulls')
    non_nulls = np.zeros(len(paper_df)).astype(bool)
    title_nulls = paper_df['title'].isnull()
    abstract_nulls = paper_df['abstract'].isnull()
    for i in range(len(paper_df)):
        if not title_nulls.iloc[i] and not abstract_nulls.iloc[i] and dates[i] != None and citations[i] != None:
            non_nulls[i] = True

    width = int(len(titles[0]) + len(abstracts[0]) + 1 + 1)

    n_non_nulls = int(np.sum(non_nulls))

    print(f'final shape ({n_non_nulls}, {width})')
    final_data = np.zeros(shape=(n_non_nulls,width),dtype=target_type)

    print('concatenate')
    counter = 0
    for i in range(len(paper_df)):
        if non_nulls[i]:
            final_data[counter] = np.concatenate((titles[i], abstracts[i], journals[i], [dates[i]], [citations[i]]))
            counter += 1

    assert counter == n_non_nulls

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

    print('change types')
    train_data = train_data.astype(target_type)
    val_data = val_data.astype(target_type)
    test_data = test_data.astype(target_type)

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
