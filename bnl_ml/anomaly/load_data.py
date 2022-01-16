#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from bnl_ml.anomaly.extract_features import get_features_single_datum
from pathlib import Path

# import random


def process_files(path, save_file_name="CSX_data.csv"):
    """
    Reads raw data, generate features, construct a pandas DataFrame and saves it as csv file

    Parameters
    ----------
    path : Path
        location of the raw hdf5 files.
    save_file_name : string, optional
        name of the file to save the data frame. The default is 'CSX_data.csv'.

    Returns
    -------
    data_frame : pandas.DataFrame
        processed data.

    """

    df = []
    files = list(path.glob("*.h5"))
    for file in files:

        d = pd.read_hdf(file, key="scan")
        features = get_features_single_datum(d)
        df.append(features)

    data_frame = pd.DataFrame(df)
    data_frame.to_csv(save_file_name)
    print(f"data is processed and saved in {save_file_name}")
    return data_frame
