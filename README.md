# Citation Prediction Engine

This project analyzes publications to predict the number of citations of a publication. This work introduces a dataset that contains the publications from the [arxiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv) and combines it with the citation count from the [cross ref API](https://www.crossref.org/services/cited-by/). The download script merges these two sources to get a novel dataset that can be used to determine relationships in acedemic citations. 

The citation count has alreay been downloaded and included in the data folder as it requires muliple days to download the full count for all 1.7 million publications. The download was run April 2023. To update the citation count simple delete the `complete.data` file and see the downloading section to see how to redownload the data.

To download the code run `git clone git@github.com:danielkopp4/citation_prediction.git` or `git clone https://github.com/danielkopp4/citation_prediction.git` for HTTP.

NOTE: Instructions below assumes using Unix-based operating system

## Downloading
`./download.sh` to download the data from kaggle. This requires a Kaggle API key in order to download. The url and target directory information is stored in the `download_config.json` file. The number of citations are included with the clone to avoid rate limiting.

## Preprocessing
`./preprocess.sh` to preprocess the data from the download. If you want a different dataset for other purposes modify the `preprocess.py` script in the `src/data_extraction` folder. If you want to run the training script with a modifed dataset you will have to update the `dataset_api.py` file also in the same folder. Parameters for the preprocess state is also in the `download_config.json` file.

## Training
`./train.sh` to run the training script. All the parameters that defines the model architecture and hyperparameters are defined in `model_params.json` file.
