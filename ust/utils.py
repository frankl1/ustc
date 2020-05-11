from pandas import DataFrame, Series
from scipy.io import arff
from os import path
from sktime.utils.load_data import load_from_arff_to_dataframe

def format_dataset(data, err, is_relative_err=False):
    ts_formatted = DataFrame()
    udata = []
    if is_relative_err:
        for i, row in data[data.columns[:-1]].iterrows():
            # convert error to relative
            tmp = [(row[c], err.iloc[i][f"{c}_err"]*abs(row[c])) for c in row.index]
            udata.append(Series(tmp))
        ts_formatted["dim_0"] = udata
    else:
        for i, row in data[data.columns[:-1]].iterrows():
        # convert error to relative
            tmp = [(row[c], err.iloc[i][f"{c}_err"]) for c in row.index]
            udata.append(Series(tmp))
        ts_formatted["dim_0"] = udata
    return ts_formatted

def load_dataset(name, dataset_folder):
    train_fs = path.join(dataset_folder, name, f"{name}_TRAIN.arff")
    test_fs = path.join(dataset_folder, name, f"{name}_TEST.arff")
    train_X, train_y = load_from_arff_to_dataframe(train_fs)
    test_X, test_y = load_from_arff_to_dataframe(test_fs)
    return (train_X, train_y), (test_X, test_y)

def load_uncertain_dataset_split(name, dataset_folder, split="TRAIN", is_relative_err=False):
    ts_fs = path.join(dataset_folder, name, f"{name}_{split}.arff")
    ts_err_fs = path.join(dataset_folder, name, f"{name}_NOISE_{split}.arff")
    ts = DataFrame(arff.loadarff(ts_fs)[0])
    ts_err = DataFrame(arff.loadarff(ts_err_fs)[0])
    ts_y = ts[ts.columns[-1]].values.astype(float).astype(int)
    ts_X = format_dataset(ts, ts_err, is_relative_err=is_relative_err)
    return ts_X, ts_y
    
def load_uncertain_dataset(name, dataset_folder, is_relative_err=False):
    train_X, train_y = load_uncertain_dataset_split(name, split='TRAIN', dataset_folder=dataset_folder, is_relative_err=is_relative_err)
    test_X, test_y = load_uncertain_dataset_split(name, split='TEST', dataset_folder=dataset_folder, is_relative_err=is_relative_err)
    
    return train_X, train_y, test_X, test_y
