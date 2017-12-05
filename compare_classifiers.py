# Python 3
import pandas
from Classifiers import Classifier

classifiers = ["AdaBoost", "DecisionTrees", "GaussianProcess", "LogisticRegression", "LinearSVM", "NaiveBayes", "NearestNeighbors", "NeuralNetwork", "RandomForest", "RBFSVM", "QDA"]
df = pandas.read_pickle('Preprocessing/preproc.pkl')
features = df['Feature Vector'].tolist()
labels = [str(lbl) for lbl in df['Class'].tolist()]

for classifier in classifiers:
    print("Evaluating using " + classifier)
    try:
        clf = Classifier(classifier)
        accuracy = clf.cross_validate(features, labels)
        print(classifier + " : " + accuracy)
    except:
        print(classifier + " : error")
