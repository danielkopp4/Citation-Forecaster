from .model import get_model

def load_params() -> dict:
    pass

def train():
    params = load_params()
    model = get_model(params)
    model.train()

if __name__ == "__main__":
    pass