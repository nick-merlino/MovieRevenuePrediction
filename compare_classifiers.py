# Python 3
import pandas
from Classifiers import Classifier
from multiprocessing import Pool

classifiers = ["DecisionTrees", "AdaBoost", "DecisionTrees", "GaussianProcess", "LogisticRegression", "LinearSVM", "NaiveBayes", "NearestNeighbors", "NeuralNetwork", "RandomForest", "RBFSVM", "QDA"]
df = pandas.read_pickle('Preprocessing/preproc.pkl')
features = df['Feature Vector'].tolist()
labels = [str(lbl) for lbl in df['Class'].tolist()]

def evaluate_classifier(classifier):
    print("Evaluating using " + classifier)
    try:
        clf = Classifier(classifier)
        accuracy = clf.cross_validate(features, labels)
        print(classifier + " : " + accuracy)
    except:
        print(classifier + " : error")

p = Pool(5)
p.map(evaluate_classifier, classifers)
