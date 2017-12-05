# Python 3
import pandas
from Classifiers import Classifier
from multiprocessing import Pool

classifiers = ["AdaBoost", "DecisionTrees", "GaussianProcess", "LinearRegression", "LinearSVM", "NaiveBayes", "NearestNeighbors", "NeuralNetwork", "RandomForest", "RBFSVM", "QDA"]
df = pandas.read_pickle('Preprocessing/preproc.pkl')
features = df['Feature Vector'].tolist()
labels = [str(lbl) for lbl in df['Class'].tolist()]

def evaluate_classifier(classifier):
    print("Evaluating using " + classifier)
    try:
        clf = Classifier(classifier)
        average = clf.cross_validate(features, labels)
        print(classifier + " : " + str(average))
    except:
        print(classifier + " : error")

p = Pool(4)
p.map(evaluate_classifier, classifiers)
