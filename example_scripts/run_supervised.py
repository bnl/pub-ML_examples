from typing import Union

from numpy import ndarray

from bnl_ml_examples.supervised.data import load_data, featurization
from bnl_ml_examples.supervised.training import train_all_models
from pathlib import Path
import numpy as np


def featurize_and_normalize(X_train, X_test):
    fe_train = featurization(X_train)
    fe_test = featurization(X_test)
    train_max: Union[ndarray, int, float, complex] = np.max(fe_train, axis=0)
    fe_train = fe_train / train_max
    fe_test = fe_test / train_max
    return fe_train, fe_test


def prettyprint_namedtuple(namedtuple, field_spaces):
    assert len(field_spaces) == len(namedtuple._fields)
    string = "{0.__name__}( ".format(type(namedtuple))
    for f_n, f_v, f_s in zip(namedtuple._fields, namedtuple, field_spaces):
        string += "{f_n}={f_v:<{f_s}.3f}".format(f_n=f_n, f_v=f_v, f_s=f_s)
    return string + ")"


def print_results(the_results):
    print("Training results")
    print("".join(["=" for _ in range(80)]))
    for key in the_results:
        print(f"{key}: ", end="")
        print(prettyprint_namedtuple(the_results[key].training, (8, 8, 8, 8)))
    print("Testing results")
    print("".join(["=" for _ in range(80)]))
    for key in the_results:
        print(f"{key}: ", end="")
        print(prettyprint_namedtuple(the_results[key].testing, (8, 8, 8, 8)))


def main():
    # Uniform training
    X_train, y_train, X_test, y_test = load_data(
        Path(__file__).parents[1] / "example_data" / "BMM_startup", uniform=True
    )
    uniform = train_all_models(X_train, y_train, X_test, y_test)
    fe_train, fe_test = featurize_and_normalize(X_train, X_test)
    fe_uniform = train_all_models(fe_train, y_train, fe_test, y_test)
    print("Uniform data splits")
    print_results(uniform)
    print("\nUniform data splits with feature engineering")
    print_results(fe_uniform)

    # Non-uniform training
    X_train, y_train, X_test, y_test = load_data(
        Path(__file__).parents[1] / "example_data" / "BMM_startup", uniform=False
    )
    split = train_all_models(X_train, y_train, X_test, y_test)
    fe_train, fe_test = featurize_and_normalize(X_train, X_test)
    fe_split = train_all_models(fe_train, y_train, fe_test, y_test)

    print("\nNon-uniform data splits")
    print_results(split)
    print("\nNon-uniform data splits with feature engineering")
    print_results(fe_split)
    return uniform, fe_uniform, split, fe_split


if __name__ == "__main__":
    results = main()
