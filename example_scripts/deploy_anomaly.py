from pickle import dump, load
import pandas as pd
import numpy as np
from pathlib import Path
import os.path
from bnl_ml_examples.anomaly.extract_features import get_features_single_datum


class AnomalyAgent:
    def __init__(self):
        """
        AnomalyAgent that uses predetermined model and reports on whether
        the recent observation is considered as anomalous.
        The agent is initialized by loading the model from a binary file.
        """
        self.independent = []  # contains unique ID of the experiments
        self.dependent = []  # contains model predictions

        folder = Path(os.path.abspath(""))
        model_file = folder / "anomaly_detection_model.pk"
        with open(model_file, "rb") as open_model_file:
            self.model = load(open_model_file)

    def tell(self, uid, y):
        """
        Classify a new observation with the model

        Parameters
        ----------
        uid: str
            uid of experimental time series
        y: array
            Relevant detector data to investigate
        """

        self.independent.append(uid)
        features = get_features_single_datum(y)
        scan_series = pd.Series(features)
        model_input = scan_series.drop(["target", "roi"]).values
        result = self.model.predict(model_input.reshape(1, -1))
        self.dependent.append(result)

    def report(self):
        """Report on the most recent classification"""
        result = self.dependent[-1]
        if result == -1:
            return "anomaly"
        else:
            return "normal"

    def ask(self):
        """Ask the agent for some advice"""
        raise NotImplementedError


class CSXDataEvaluation:
    """
    A machine learning model to check if the recent time-series
    experiment has any istabilities (anomalies).
    """

    def __init__(self, agent):
        self.agent = agent

    def evaluate(self, path, uid):
        """
        Perform an evaluation of a measurement. The time series are
        extracted from a hdf5 report file, processed into model's
        arguments, then get subjected to the model. The model return
        a prediction of whether the data can be anomalous.

        Parameters
        ----------
        path : Path
            location of report files
        uid : str
            uid of data to be evaluated
        """
        self.agent.tell(uid, self.process_datafile(path, uid))
        return self.agent.report()

    def process_datafile(self, path, uid):
        """
        Reads raw data, generate features, constract a pandas DataFrame.
        In the report files, if the data was not evaluated with the model,
        the classification label is np.nan.

        Parameters
        ----------
        path : Path
            location of the raw hdf5 files.

        Returns
        -------
        model_input : numpy.array,
            features generated from raw data to be used as input to the model

        """

        file = list(path.glob(f"*{uid}*.h5"))[0]
        d = pd.read_hdf(file, key="scan")

        return d
