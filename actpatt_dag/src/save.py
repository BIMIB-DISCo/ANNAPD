import os
import pickle


def save_data(data, file_name=""):

    if not os.path.exists("data"):
        os.mkdir("data")

    check = os.path.exists("data/" + file_name.split('/')[0])
    if "/" in file_name and not check:
        os.mkdir("data/" + file_name.split('/')[0])
     
    with open("data/" + file_name, 'wb') as f:
        pickle.dump(data, f)
