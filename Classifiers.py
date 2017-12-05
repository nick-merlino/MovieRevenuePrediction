from __future__ import division
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from Preprocessing.process_text import process

class Classifier:
    def __init__(self,method):
        if method == "AdaBoost":
            self.clf = AdaBoostClassifier()
        elif method == "DecisionTrees":
            self.clf = DecisionTreeClassifier(max_depth=5)
        elif method == "GaussianProcess":
            self.clf = GaussianProcessClassifier(1.0 * RBF(1.0))
        elif method == "LinearRegression":
            self.clf = LinearRegression()
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

    def cross_validate(self, features, labels, num_folds=5):
        score = cross_val_score(self.clf, features, labels, cv=StratifiedKFold(num_folds, shuffle=True)) #scoring='custom_eval_method'
        return (sum(score) / len(score))

    def train(self, features, labels):
        self.clf.fit(features, labels)

    def predict(self, summary):
        features = process(summary)
        return self.clf.predict(features)
