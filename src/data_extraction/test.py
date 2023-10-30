import pytest
from your_script import create_engine, get_dois, get_remaining_dois
import pandas as pd


engine = create_engine(params['database_url'])

def test_doi_retrieval():
    dois = get_dois(params, engine)
    assert isinstance(dois, list) and all(isinstance(doi, str) for doi in dois)

def test_remaining_dois():
    remaining_dois = get_remaining_dois(params, engine)
    assert isinstance(remaining_dois, list) and all(isinstance(doi, str) for doi in remaining_dois)


if __name__ == '__main__':
    pytest.main()
