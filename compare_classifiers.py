# Python 3
import pandas
from Classifiers import Classifier
from pickle import load
import sys
from datetime import datetime

print("Reading in data")
vec = load(open("Preprocessing/preproc.pkl", "rb"))
revenue_classes = load(open("Preprocessing/revenue_classes.pkl", "rb"))
# TODO: Need dense matrix... GaussianProcessClassifier, LogisticRegression, GaussianNaiveBayes, NeuralNetwork, RBFSVC, QuadraticDiscriminantAnalysis
classifiers = ["AdaBoost", "DecisionTrees5", "DecisionTrees10", "DecisionTrees20", "RidgeClassifier", "LinearSVC", "SGDClassifier", "Perceptron", "PassiveAggressiveClassifier", "BernoulliNB", "MultinomialNB", "KNeighborsClassifier", "NearestCentroid", "RandomForestClassifier"]
for classifier in classifiers:
    print("Evaluating using " + classifier + " at " + str(datetime.now()))
    try:
        clf = Classifier(classifier)
        accuracy = clf.cross_validate(vec, revenue_classes)
        print(classifier + " : " + accuracy)
    except Exception as e:
        print(classifier + " : error")# + str(e))
