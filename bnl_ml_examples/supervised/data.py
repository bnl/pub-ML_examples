from pathlib import Path
import h5py
from sklearn.model_selection import train_test_split
import numpy as np


def load_data(data_dir, uniform=True, seed=1234):
    """
    Loads min/max normalized data according to split preference

    Parameters
    ----------
    data_dir : Path
        Directory containing .hdf5 files
    uniform : bool
        To load data randomly uniform or to split according to data quality
        False will give the failure mode presented in the paper which requires feature engineering
    seed : int
        Seed for random shuffle

    Returns
    -------
    X_train, y_train, X_test, y_test

    """

    def split_file(path):
        with h5py.File(path, "r") as f:
            score = list()
            mu = list()
            energy = list()
            for uid in f.keys():
                score.append(int(f[uid].attrs["score"]))
                mu.append(list(f[uid]["mu"]))
                energy.append(list(f[uid]["energy"]))
        return score, mu, energy

    scores = []
    data = []
    energies = []
    if uniform:
        paths = list(data_dir.glob("*.hdf5"))
        for path in paths:
            score, mu, energy = split_file(path)
            scores.extend(score)
            data.extend(mu)
            energies.extend(energy)
        X_train, X_test, y_train, y_test = train_test_split(
            data, scores, test_size=0.2, shuffle=True, random_state=seed
        )
    else:
        train_paths = [
            data_dir / "fluorescence_training_set.hdf5",
            data_dir / "transmission_training_set.hdf5",
        ]
        test_paths = [data_dir / "verygood_training_set.hdf5"]
        for path in train_paths:
            score, mu, energy = split_file(path)
            scores.extend(score)
            data.extend(mu)
            energies.extend(energy)
        test_scores = []
        test_data = []
        test_energies = []
        for path in test_paths:
            score, mu, energy = split_file(path)
            test_scores.extend(score)
            test_data.extend(mu)
            test_energies.extend(energy)
        X_train, X_test, y_train, y_test = train_test_split(
            data, scores, test_size=0.1, shuffle=True, random_state=seed
        )
        X_test = np.concatenate([np.array(test_data), X_test])
        y_test = np.concatenate([np.array(test_scores), y_test])

    # Normalization
    X_train = (X_train - np.min(X_train, axis=1, keepdims=True)) / (
        np.max(X_train, axis=1, keepdims=True)
        - np.min(X_train, axis=1, keepdims=True)
        + 1e-8
    )
    X_test = (X_test - np.min(X_test, axis=1, keepdims=True)) / (
        np.max(X_test, axis=1, keepdims=True)
        - np.min(X_test, axis=1, keepdims=True)
        + 1e-8
    )
    y_train = np.array(y_train)

    return X_train, y_train, X_test, y_test


def featurization(X):
    def autocorr(x, t=1):
        return np.corrcoef(np.array([x[:-t], x[t:]]))[0, 1]

    def extract_autocorrelation_features(x):
        ac1 = autocorr(x, 1)
        ac2 = autocorr(x, 2)
        ac3 = autocorr(x, 3)
        ac4 = autocorr(x, 4)
        corr_coeffs = [ac1, ac2, ac3, ac4]
        labels = ["_ac" + str(i) for i in range(1, 5)]
        return corr_coeffs, labels

    def extract_start_end(X):
        start_end = abs(X[:, :5].mean(axis=1) - X[:, -5:].mean(axis=1))
        return start_end

    def basic_stats(X):
        mean = np.mean(X, axis=1)
        var = np.var(X, axis=1)
        _sum = np.sum(X, axis=1)
        argmax = np.argmax(X, axis=1)
        return np.stack([mean, _sum, var, argmax], axis=1)

    corr_coeffs = []
    for i in range(X.shape[0]):
        cc, l = extract_autocorrelation_features(X[i, :])
        cc = np.nan_to_num(cc, nan=0)
        corr_coeffs.append(cc)
    corr_coeffs = np.array(corr_coeffs)
    start_end = np.expand_dims(extract_start_end(X), axis=1)
    basic = basic_stats(X)

    # repete for derivative
    diff = X[:, 1:] - X[:, :-1]
    d_corr_coeffs = []
    for i in range(diff.shape[0]):
        cc, l = extract_autocorrelation_features(diff[i, :])
        cc = np.nan_to_num(cc, nan=0)
        d_corr_coeffs.append(cc)
    d_corr_coeffs = np.array(corr_coeffs)
    d_start_end = np.expand_dims(extract_start_end(diff), axis=1)
    d_basic = basic_stats(diff)

    return np.concatenate(
        [corr_coeffs, start_end, basic, d_corr_coeffs, d_start_end, d_basic], axis=1
    )
