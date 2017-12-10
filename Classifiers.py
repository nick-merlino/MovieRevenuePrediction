from __future__ import division
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from pickle import dump, load
from pathlib import Path
import numpy as np

class Classifier:
    def __init__(self,method):
        if method == "AdaBoost":
            self.clf = AdaBoostClassifier()
        elif method == "DecisionTrees5":
            self.clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)
        elif method == "DecisionTrees10":
            self.clf = DecisionTreeClassifier(criterion="entropy", max_depth=10)
        elif method == "DecisionTrees20":
            self.clf = DecisionTreeClassifier(criterion="entropy", max_depth=20)
        elif method == "GaussianProcessClassifier":
            self.clf = GaussianProcessClassifier(n_jobs=-1)
        elif method == "LogisticRegression":
            self.clf = LogisticRegression(n_jobs=-1)
        elif method == "LinearSVC":
            self.clf = LinearSVC()
        elif method == "GaussianNaiveBayes":
            self.clf = GaussianNB()
        elif method == "MultinomialNaiveBayes":
            self.clf = MultinomialNaiveBayes()
        elif method == "BernoulliNaiveBayes":
            self.clf = BernoulliNB()
        elif method == "KNeighborsClassifier":
            self.clf = KNeighborsClassifier(n_jobs=-1)
        elif method == "NeuralNetwork":
            self.clf = MLPClassifier()
        elif method == "NearestCentroid":
            self.clf = NearestCentroid()
        elif method == "RandomForestClassifier":
            self.clf = RandomForestClassifier(n_jobs=-1)
        elif method == "RBFSVC":
            self.clf = SVC()
        elif method == "SGDClassifier":
            self.clf = SGDClassifier(n_jobs=-1)
        elif method == "Perceptron":
            self.clf = Perceptron(n_jobs=-1)
        elif method == "QuadraticDiscriminantAnalysis":
            self.clf = QuadraticDiscriminantAnalysis()
        elif method == "PassiveAggressiveClassifier":
            self.clf = PassiveAggressiveClassifier(n_jobs=-1)
        elif method == "RidgeClassifier":
            self.clf = RidgeClassifier()
        else:
            self.clf = None
            print("Uh oh - classifer name unknown")
        self.method = method

    def cross_validate(self, features, labels, num_folds=5):
        scores = cross_val_score(self.clf, features, labels, n_jobs=-1, cv=StratifiedKFold(num_folds, shuffle=True))
        return ("%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    def train(self, features, labels):
        if Path('models/' + self.method + ".pkl").is_file():
            print(self.method + ": loading pre-trained model")
            self.clf = load(open('models/' + self.method + ".pkl", "rb"))
        else:
            print(self.method + ": training new model")
            self.clf.fit(features, labels)
            dump(self.clf, open('models/' + self.method + ".pkl", "wb"))

    def predict_class(self, features, labels, vec):
        self.train(features, labels)
        return int(np.asscalar(self.clf.predict(vec)))
