"""
Borrowed code from https://github.com/NSLS-II-BMM/profile_collection/blob/master/startup/BMM/ml.py
modified to use the standard tell--report--ask interface.

Used only as a readable example, as the requirements would involve the full BMM environment

Deployment would proceed as seen here:
https://github.com/NSLS-II-BMM/profile_collection/blob/master/startup/BMM/xafs.py#L1167-L1169
"""

from joblib import dump, load
import h5py
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from BMM import user_ns as user_ns_module
from BMM.user_ns.bmm import BMMuser

user_ns = vars(user_ns_module)


class ClassificationAgent:
    def __init__(self):
        """
        ClassificationAgent that uses predetermined model development, and reports with emoji's for Slack output.
        In this context, instantiating the model will train a RandomForest from scratch.
        """
        self.independent = []
        self.dependent = []
        self.mode = None  # Updatable mode
        self.good_emoji = ":heavy_check_mark:"
        self.bad_emoji = ":heavy_multiplication_x:"
        # self.folder = os.path.join(
        #     os.getenv("HOME"), ".ipython", "profile_collection", "startup", "ML"
        # )
        self.folder = Path(__file__).parents[1] / "example_data" / "BMM_startup"
        self.model_path = self.folder / "data_evaluation.joblib"
        self.hdf5 = [
            self.folder / "fluorescence_training_set.hdf5",
            self.folder / "transmission_training_set.hdf5",
            self.folder / "verygood_training_set.hdf5",
        ]
        if self.model_path.is_file:
            self.clf = load(self.model_path)
        else:
            self.train()

    def tell(self, x, y):
        """
        Tell the classifier about something new
        Parameters
        ----------
        x: str
            These are the interesting parameters, kept as UIDs here
        y: array
            Relevant spectrum to investigate

        Returns
        -------

        """
        self.independent.append(x)
        result = self.clf.predict(y)[0]
        self.dependent.append(result)

    def report(self):
        """Report on the most recent classification"""
        result = self.dependent[-1]
        if result == 1:
            return result, self.good_emoji
        else:
            return result, self.bad_emoji

    def ask(self):
        """Ask the agent for some advice"""
        raise NotImplementedError

    def train(self):
        """
        Using all the hdf5 files of interpolated, scored data, create the
        evaluation model, saving it to a joblib dump file.
        """
        scores = list()
        data = list()
        for h5file in self.hdf5:
            if os.path.isfile(h5file):
                print(f"reading data from {h5file}")
                f = h5py.File(h5file, "r")
                for uid in f.keys():
                    score = int(f[uid].attrs["score"])
                    mu = list(f[uid]["mu"])
                    scores.append(score)
                    data.append(mu)

        print("training model...")
        self.clf = RandomForestClassifier(random_state=0)

        self.clf.fit(data, scores)
        dump(self.clf, self.model_path)
        print(f"wrote model to {self.model_path}")
        return ()


class BMMDataEvaluation:
    """A very simple machine learning model for recognizing when an XAS
    scan goes horribly awry.
    """

    def __init__(self, agent):
        self.GRIDSIZE = 401
        self.agent = agent

    def evaluate(self, uid, mode=None):
        """Perform an evaluation of a measurement.  The data will be
        interpolated onto the same grid used for the training set,
        then get subjected to the model.  This returns a tuple with
        the score (1 or 0) and the Slack-appropriate value (green
        check or red cross).
        Parameters
        ----------
        uid : str
            uid of data to be evaluated
        mode : bool
            when not None, used to specify fluorescence or transmission (for a data set that has both)
        """
        self.agent.tell(uid, self.extract_data(uid, mode))
        self.agent.report()

    def rationalize_mu(self, en, mu):
        """Return energy and mu on a "rationalized" grid of equally spaced points.  See slef.GRIDSIZE"""
        ee = list(
            np.arange(
                float(en[0]),
                float(en[-1]),
                (float(en[-1]) - float(en[0])) / self.GRIDSIZE,
            )
        )
        mm = np.interp(ee, en, mu)
        return (ee, mm)

    def extract_data(self, uid, mode=None):
        """Extract data from a uid
         The data will be
        interpolated onto the same grid used for the training set,
        then get subjected to the model.  This returns a tuple with
        the score (1 or 0) and the Slack-appropriate value (green
        check or red cross).
        Parameters
        ----------
        uid : str
            uid of data to be evaluated
        """
        if mode == "xs":
            t = user_ns["db"][-1].table()
            el = BMMuser.element
            i0 = np.array(t["I0"])
            en = np.array(t["dcm_energy"])
            dtc1 = np.array(t[el + "1"])
            dtc2 = np.array(t[el + "2"])
            dtc3 = np.array(t[el + "3"])
            dtc4 = np.array(t[el + "4"])
            signal = dtc1 + dtc2 + dtc3 + dtc4
            mu = signal / i0
        else:
            this = user_ns["db"].v2[uid]
            if mode is None:
                mode = this.metadata["start"]["XDI"]["_mode"][0]
            element = this.metadata["start"]["XDI"]["Element"]["symbol"]
            i0 = this.primary.read()["I0"]
            en = this.primary.read()["dcm_energy"]
            if mode == "transmission":
                it = this.primary.read()["It"]
                mu = np.log(abs(i0 / it))
            elif mode == "reference":
                it = this.primary.read()["It"]
                ir = this.primary.read()["Ir"]
                mu = np.log(abs(it / ir))
            else:
                if element in str(this.primary.read()["vor:vor_names_name3"][0].values):
                    signal = (
                        this.primary.read()["DTC1"]
                        + this.primary.read()["DTC2"]
                        + this.primary.read()["DTC3"]
                        + this.primary.read()["DTC4"]
                    )
                elif element in str(
                    this.primary.read()["vor:vor_names_name15"][0].values
                ):
                    signal = (
                        this.primary.read()["DTC2_1"]
                        + this.primary.read()["DTC2_2"]
                        + this.primary.read()["DTC2_3"]
                        + this.primary.read()["DTC2_4"]
                    )
                elif element in str(
                    this.primary.read()["vor:vor_names_name19"][0].values
                ):
                    signal = (
                        this.primary.read()["DTC3_1"]
                        + this.primary.read()["DTC3_2"]
                        + this.primary.read()["DTC3_3"]
                        + this.primary.read()["DTC3_4"]
                    )
                else:
                    print("cannot figure out fluorescence signal")
                    # print(f'vor:vor_names_name3 {}')
                    return ()
                mu = signal / i0
        e, m = self.rationalize_mu(en, mu)
        if len(m) > self.GRIDSIZE:
            m = m[:-1]

        return m
