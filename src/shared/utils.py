import json

def load_params(file_name: str) -> dict:
    with open(file_name, 'r') as file:
        return json.load(file)
    