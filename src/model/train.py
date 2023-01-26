from src.model.model import get_model
from src.shared.utils import load_params
import sys

def train(params: dict):
    model = get_model(params)
    model.train()

if __name__ == "__main__":
    file_name = sys.argv[1]
    params = load_params(file_name)
    train(params)