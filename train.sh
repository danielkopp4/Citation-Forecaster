#!/bin/bash
# tmux new-session -d -s "training_session"\; \
#     setenv CONDA_DEFAULT_ENV $CONDA_DEFAULT_ENV\; \
#     send-keys 
python3 -m src.model.train model_params.json
# eval "$(conda shell.bash hook)" && conda activate $CONDA_DEFAULT_ENV && python3 -m src.model.train model_params.json