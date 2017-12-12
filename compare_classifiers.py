# Python 3
import pandas
from Classifiers import Classifier
from pickle import load
from datetime import datetime
from scipy.sparse import hstack
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

print("Reading in data")
vec_freq = load(open("Preprocessing/preproc-freq.pkl", "rb"))
vec_binary = load(open("Preprocessing/preproc-binary.pkl", "rb"))
vec_keywords = load(open("Preprocessing/preproc-keywords.pkl", "rb"))
train_vec_freq = hstack([vec_freq, vec_keywords])
train_vec_binary = hstack([vec_binary, vec_keywords])
train_labels = load(open("Preprocessing/train_labels.pkl", "rb"))

del vec_freq
del vec_binary
del vec_keywords

classifiers = ["DecisionTrees5", "DecisionTrees10", "DecisionTrees20", "RandomForestClassifier5", "RandomForestClassifier10", "RandomForestClassifier20", "MultinomialNaiveBayes1", "MultinomialNaiveBayes0.1", "MultinomialNaiveBayes5", "LogisticRegressionSagaL1", "LogisticRegressionSagaL2", "AdaBoost", "RidgeClassifier", "LinearSVC", "SGDClassifier", "Perceptron", "PassiveAggressiveClassifier"]
clf = None
for classifier in classifiers:
    print("Evaluating using " + classifier + " at " + str(datetime.now()))
    try:
        clf = Classifier(classifier, train_labels, train_vec_freq, train_vec_binary)
        accuracy = clf.cross_validate()
        print(classifier + " : " + accuracy)
    except Exception as e:
        print(classifier + " : " + str(e))
