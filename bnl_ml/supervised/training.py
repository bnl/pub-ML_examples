from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from collections import namedtuple

Metrics = namedtuple("Metrics", "accuracy recall precision f1_score")
Result = namedtuple("Result", "training testing")


def train_all_models(X_train, y_train, X_test, y_test, seed=1234):
    model_dict = {
        "RF": RandomForestClassifier(random_state=seed),
        "SVM": SVC(random_state=seed),
        "KNeigh": KNeighborsClassifier(),
        "GP": GaussianProcessClassifier(random_state=seed),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(10,), max_iter=1000, random_state=seed
        ),
    }

    results = {}
    for key, clf in model_dict.items():
        clf.fit(X_train, y_train)
        pred_train = clf.predict(X_train)
        pred_test = clf.predict(X_test)
        results[key] = Result(
            training=Metrics(
                accuracy=accuracy_score(y_train, pred_train),
                recall=recall_score(y_train, pred_train),
                precision=precision_score(y_train, pred_train),
                f1_score=f1_score(y_train, pred_train),
            ),
            testing=Metrics(
                accuracy=accuracy_score(y_test, pred_test),
                recall=recall_score(y_test, pred_test),
                precision=precision_score(y_test, pred_test),
                f1_score=f1_score(y_test, pred_test),
            ),
        )
    return results
