from __future__ import division
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from Preprocessing.process_text import process
from pickle import dump, load
from pathlib import Path
import numpy as np

class Classifier:
    def __init__(self,method):
        if method == "AdaBoost":
            self.clf = AdaBoostClassifier()
        elif method == "DecisionTrees":
            self.clf = DecisionTreeClassifier(max_depth=10)
        elif method == "GaussianProcess":
            self.clf = GaussianProcessClassifier(1.0 * RBF(1.0))
        elif method == "LogisticRegression":
            self.clf = LogisticRegression()
        elif method == "LinearSVM":
            self.clf = SVC(kernel="linear", C=0.025)
        elif method == "NaiveBayes":
            self.clf = GaussianNB()
        elif method == "NearestNeighbors":
            self.clf = KNeighborsClassifier(3)
        elif method == "NeuralNetwork":
            self.clf = MLPClassifier(alpha=1)
        elif method == "RandomForest":
            self.clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
        elif method == "RBFSVM":
            self.clf = SVC(gamma=2, C=1)
        elif method == "QDA":
            self.clf = QuadraticDiscriminantAnalysis()
        else:
            self.clf = None
            print("Uh oh")
        self.method = method

    def cross_validate(self, features, labels, num_folds=5):
        scores = cross_val_score(self.clf, features, labels, cv=StratifiedKFold(num_folds, shuffle=True)) #scoring='custom_eval_method'
        return ("%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    def train(self, features, labels):
        if Path('models/' + self.method + ".pkl").is_file():
            print(self.method + ": loading pre-trained model")
            self.clf = load(open('models/' + self.method + ".pkl", "rb"))
        else:
            print(self.method + ": training new model")
            self.clf.fit(features, labels)
            dump(self.clf, open('models/' + self.method + ".pkl", "wb"))

    def predict_class(self, features, labels, feature_dict, summary):
        self.train(features, labels)

        for w in process(summary):
            feature_dict[w] = 1
        feature_vector = [[feature_dict[key] for key in sorted([*feature_dict])]]
        return int(np.asscalar(self.clf.predict(feature_vector)))
