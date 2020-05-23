import pickle

def load_pickle(file):
    with open(file, 'rb') as file:
        return pickle.load(file)