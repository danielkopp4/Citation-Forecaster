# Citation Prediction Engine


## Downloading
`./download.sh` to download the data from kaggle. This requires a Kaggle API key in order to download. The url and target directory information is stored in the `download_config.json` file. The number of citations are included with the clone to avoid rate limiting.

## Preprocessing
`./preprocess.sh` to preprocess the data from the download. If you want a different dataset for other purposes modify the `preprocess.py` script in the `src/data_extraction` folder. If you want to run the training script with a modifed dataset you will have to update the `dataset_api.py` file also in the same folder. Parameters for the preprocess state is also in the `download_config.json` file.

## Training
`./train.sh` to run the training script. All the parameters that defines the model architecture and hyperparameters are defined in `model_params.json` file.
