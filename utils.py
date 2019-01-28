from glob import glob
import pickle

def load_bout_data(path_glob):
    bout_file_names = sorted(glob(path_glob))
    bout_data = {}
    for name in bout_file_names:
        date = name.split('_')[1][:-4]
        with open(name, 'rb') as f:
            bout_data[date] = pickle.load(f)

    return list(filter(lambda item: len(item[1].keys()) > 0, bout_data.items()))

def load_banzuke_data(path_glob):
    banzuke_file_names = sorted(glob(path_glob))
    banzuke_data = {}
    for name in banzuke_file_names:
        date = name.split('_')[1][:-4]
        with open(name, 'rb') as f:
            banzuke_data[date] = pickle.load(f)

    return list(filter(lambda item: len(item[1]) > 0, banzuke_data.items()))