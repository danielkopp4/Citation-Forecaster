#!/bin/bash
tmux new-session -d -s "training_session" "python3 -m src.model.train model_params.json"
# python3 -m src.model.train model_params.json