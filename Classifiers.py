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
    def __init__(self,method, train_labels, train_freq, train_binary, test_freq=None, test_binary=None):
        self.method = method
        self.train_vec = train_freq
        self.test_vec = test_freq
        self.train_labels = train_labels
        if method == "AdaBoost":
            self.clf = AdaBoostClassifier()
        elif method == "DecisionTrees5":
            self.clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)
            self.train_vec = train_binary
            self.test_vec = test_binary
        elif method == "DecisionTrees10":
            self.clf = DecisionTreeClassifier(criterion="entropy", max_depth=10)
            self.train_vec = train_binary
            self.test_vec = test_binary
        elif method == "DecisionTrees20":
            self.clf = DecisionTreeClassifier(criterion="entropy", max_depth=20)
            self.train_vec = train_binary
            self.test_vec = test_binary
        elif method == "LogisticRegressionSagaL1":
            self.clf = LogisticRegression(solver="saga", penalty='l1', multi_class='multinomial', n_jobs=-1)
        elif method == "LogisticRegressionSagaL2":
            self.clf = LogisticRegression(solver="saga", penalty='l2', multi_class='multinomial', n_jobs=-1)
        elif method == "LinearSVC":
            self.clf = LinearSVC()
        elif method == "MultinomialNaiveBayes0.1":
            self.clf = MultinomialNB(alpha=0.1)
        elif method == "MultinomialNaiveBayes1":
            self.clf = MultinomialNB()
        elif method == "MultinomialNaiveBayes5":
            self.clf = MultinomialNB(alpha=5)
        elif method == "RandomForestClassifier5":
            self.clf = RandomForestClassifier(criterion="entropy", max_depth=5, n_jobs=-1)
            self.train_vec = train_binary
            self.test_vec = test_binary
        elif method == "RandomForestClassifier10":
            self.clf = RandomForestClassifier(criterion="entropy", max_depth=10, n_jobs=-1)
            self.train_vec = train_binary
            self.test_vec = test_binary
        elif method == "RandomForestClassifier20":
            self.clf = RandomForestClassifier(criterion="entropy", max_depth=20, n_jobs=-1)
            self.train_vec = train_binary
            self.test_vec = test_binary
        elif method == "SGDClassifier":
            self.clf = SGDClassifier(n_jobs=-1, max_iter=1000, tol=1e-3)
        elif method == "Perceptron":
            self.clf = Perceptron(n_jobs=-1, max_iter=1000, tol=1e-3)
        elif method == "PassiveAggressiveClassifier":
            self.clf = PassiveAggressiveClassifier(n_jobs=-1, max_iter=1000, tol=1e-3)
        elif method == "RidgeClassifier":
            self.clf = RidgeClassifier()
        else:
            self.clf = None
            print("Uh oh - classifer name unknown")

    def cross_validate(self):
        scores = cross_val_score(self.clf, self.train_vec, self.train_labels, n_jobs=-1, cv=StratifiedKFold(5, shuffle=True))
        return ("%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    def train(self):
        if Path('models/' + self.method + ".pkl").is_file():
            print(self.method + ": loading pre-trained model")
            self.clf = load(open('models/' + self.method + ".pkl", "rb"))
        else:
            print(self.method + ": training new model")
            self.clf.fit(self.train_vec, self.train_labels)
            dump(self.clf, open('models/' + self.method + ".pkl", "wb"))

    def predict_class(self):
        self.train()
        return int(self.clf.predict(self.test_vec)[0])
