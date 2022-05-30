from imp import load_source
import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import torch

from src.settings import DATA_ROOT


class TSQuantileTransformer:
    def __init__(self, *args, n_quantiles: int, **kwargs):
        self.n_quantiles = n_quantiles
        self._args = args
        self._kwargs = kwargs
        self.transforms = {}

    def fit(self, features: np.ndarray):
        for i in range(features.shape[1]):
            self.transforms[i] = QuantileTransformer(
                *self._args, n_quantiles=self.n_quantiles, **self._kwargs
            ).fit(features[:, i, :])
        return self

    def transform(self, features: np.ndarray):
        result = np.empty_like(features, dtype=np.int32)
        for i in range(features.shape[1]):
            result[:, i, :] = (
                self.transforms[i].transform(features[:, i, :]) * self.n_quantiles
            ).astype(np.int32)
        return result


# There are 2 different datasets COBRE and ABIDE,
# each contain two classes of subjects: patients and controls.
# There are multiple files to be read,
# and although they are all available in the repository,
# I am attaching what you need to reproduce a dataset for ABIDE.
# It will get you data in the shape: (569, 53, 140), where you have 569 subjects.
# Each has 53 channels with 140 time points.
# Learn from the code how to apply the same to COBRE.
def load_ABIDE1(
    dataset_path: str = DATA_ROOT.joinpath("abide/ABIDE1_AllData.h5"),
    indices_path: str = DATA_ROOT.joinpath("abide/correct_indices_GSP.csv"),
    labels_path: str = DATA_ROOT.joinpath("abide/labels_ABIDE1.csv"),
):
    # pdb.set_trace()
    hf = h5py.File(dataset_path, "r")
    data = hf.get("ABIDE1_dataset")
    data = np.array(data)
    num_subjects = data.shape[0]
    num_components = 100
    data = data.reshape(num_subjects, num_components, -1)

    # take only those brain networks that are not noise
    df = pd.read_csv(indices_path, header=None)
    c_indices = df.values
    c_indices = c_indices.astype("int")
    c_indices = c_indices.flatten()
    c_indices = c_indices - 1
    finalData = data[:, c_indices, :]

    df = pd.read_csv(labels_path, header=None)
    labels = df.values.flatten() - 1

    return finalData, labels


def load_FBIRN(
    dataset_path: str = DATA_ROOT.joinpath("fbirn/FBIRN_AllData.h5"),
    indices_path: str = DATA_ROOT.joinpath("fbirn/correct_indices_GSP.csv"),
    labels_path: str = DATA_ROOT.joinpath("fbirn/labels_FBIRN_new.csv"),
):
    hf = h5py.File(dataset_path, "r")
    data = hf.get("FBIRN_dataset")
    data = np.array(data)
    num_subjects = data.shape[0]
    num_components = 100
    data = data.reshape(num_subjects, num_components, -1)

    # take only those brain networks that are not noise
    df = pd.read_csv(indices_path, header=None)
    c_indices = df.values
    c_indices = c_indices.astype("int")
    c_indices = c_indices.flatten()
    c_indices = c_indices - 1
    finalData = data[:, c_indices, :]
    # 311 - sessions - data.shape[0]
    # 53 - components - data.shape[1]
    # 140 - time points - data.shape[2]

    df = pd.read_csv(labels_path, header=None)
    labels = df.values.flatten() - 1
    # 311 - sessions - data.shape[0]

    return finalData, labels


def load_OASIS(
    only_first_sessions: bool = True,
    only_two_classes: bool = True,
    dataset_path: str = DATA_ROOT.joinpath("oasis/OASIS3_AllData_allsessions.npz"),
    indices_path: str = DATA_ROOT.joinpath("oasis/correct_indices_GSP.csv"),
    labels_path: str = DATA_ROOT.joinpath("oasis/labels_OASIS_6_classes.csv"),
    sessions_path: str = DATA_ROOT.joinpath("oasis/oasis_first_sessions_index.csv"),
):
    data = np.load(dataset_path)
    # 2826 - sessions - data.shape[0]
    # 100 - components - data.shape[1]
    # 160 - time points - data.shape[2]

    indices = pd.read_csv(indices_path, header=None)
    idx = indices[0].values - 1

    data = data[:, idx, :156]

    labels = pd.read_csv(labels_path, header=None)
    labels = labels.values.flatten().astype("int") - 1

    if only_first_sessions:
        sessions = pd.read_csv(sessions_path, header=None)
        first_session = sessions[0].values - 1

        data = data[first_session, :, :]
        # 912 - sessions - data.shape[0] - only first session
        labels = labels[first_session]

    if only_two_classes:
        # leaves values with labels 0 and 1 only
        filter_array = []
        for label in labels:
            if label in (0, 1):
                filter_array.append(True)
            else:
                filter_array.append(False)

        data = data[filter_array, :, :]
        labels = labels[filter_array]

    return data, labels


def load_balanced_OASIS():
    features, labels = load_OASIS()

    filter_array_0 = []
    filter_array_1 = []

    for label in labels:
        if label == 0:
            filter_array_0.append(True)
            filter_array_1.append(False)
        else:
            filter_array_0.append(False)
            filter_array_1.append(True)

    features_0 = features[filter_array_0]
    labels_0 = labels[filter_array_0]
    features_1 = features[filter_array_1]
    labels_1 = labels[filter_array_1]

    features_0 = features_0[:150]
    labels_0 = labels_0[:150]
    features_1 = features_1[:150]
    labels_1 = labels_1[:150]

    features = np.concatenate((features_0, features_1), axis=0)
    labels = np.concatenate((labels_0, labels_1), axis=0)

    return features, labels


def _find_indices_of_each_class(all_labels):
    HC_index = (all_labels == 0).nonzero()
    SZ_index = (all_labels == 1).nonzero()

    return HC_index, SZ_index
