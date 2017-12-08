# Python 3
import pandas
from Classifiers import Classifier
from pickle import load

tfidf = load(open("Preprocessing/preproc.pkl", "rb"))
revenue_classes = load(open("Preprocessing/revenue_classes.pkl", "rb"))

classifiers = ["DecisionTrees"]#'"AdaBoost", "DecisionTrees", "GaussianProcess", "LogisticRegression", "LinearSVM", "NaiveBayes", "NearestNeighbors", "NeuralNetwork", "RandomForest", "RBFSVM", "QDA"]
for classifier in classifiers:
    print("Evaluating using " + classifier)
    try:
        clf = Classifier(classifier)
        accuracy = clf.cross_validate(tfidf, revenue_classes)
        print(classifier + " : " + accuracy)
    except Exception as e:
        print(classifier + " : " + "error")#str(e))
