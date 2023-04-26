# Data Extraction

## Baseline
- [x] ArXiv Dataset (Kaggle)
- [x] Some Google Scholar API to retrieve citations
    - [x] Decide Scholarly / SerpAPI

- [x] Merge the data to get a preprocessed dataset for model use
- [x] Create Keras / pyTorch Dataset API with options to modify inputs to model (keras:Sequence / alternatively use pyTorch)
- [x] Use SBERT model to preprocess text into vectors 

## Some Useful Features 
- [ ] Store invalid papers for future inspection and decide if they should be included due to error
- [ ] Download Kaggle dataset first and progressively create a new dataset with saved values (Kaggle dataset is static, processed one is not) that can be stopped and started at will

## Dataset Layout / Update Instructions
our dataset contains scientific papaer information from the arXiv dataset as well as OpenCitations. The title, abstract, journal, and date all come from the arXiv dataset(https://www.kaggle.com/datasets/Cornell-University/arxiv). The citations are scrapped using OpenCitations(https://opencitations.net/). The current data.json file contains data from 4/3/2023.

In order to get this dataset with updated data, refer to top readme
