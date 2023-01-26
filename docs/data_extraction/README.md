# Data Extraction

## Baseline
- [ ] ArXiv Dataset (Kaggle)
- [ ] Some Google Scholar API to retrieve citations
    - [ ] Decide Scholarly / SerpAPI

- [ ] Merge the data to get a preprocessed dataset for model use
- [ ] Create Keras / pyTorch Dataset API with options to modify inputs to model (keras:Sequence / alternatively use pyTorch)
- [ ] Use SBERT model to preprocess text into vectors 

## Some Useful Features 
- [ ] Store invalid papers for future inspection and decide if they should be included due to error
- [ ] Download Kaggle dataset first and progressively create a new dataset with saved values (Kaggle dataset is static, processed one is not) that can be stopped and started at will