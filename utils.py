import pickle


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
