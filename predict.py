# Python 3
from Classifiers import Classifier
import argparse
from multiprocessing import Pool
import pandas
from pickle import load

parser = argparse.ArgumentParser(description='Predict the revenue of a plot summary.')
parser.add_argument('summary_file', help='Path to the file containing the summary')
parser.add_argument('-c', help='DecisionTrees / LinearRegression / NaiveBayes / NeuralNetwork / SVM / RandomForest')
args = parser.parse_args()

classifiers = [args.c] if args.c else ["DecisionTrees", "AdaBoost", "DecisionTrees", "GaussianProcess", "LogisticRegression", "LinearSVM", "NaiveBayes", "NearestNeighbors", "NeuralNetwork", "RandomForest", "RBFSVM", "QDA"]

summary = open(args.summary_file, 'r').read()

df = pandas.read_pickle('Preprocessing/preproc.pkl')
features = df['Feature Vector'].tolist()
labels = [str(lbl) for lbl in df['Class'].tolist()]

feature_dict = load(open("Preprocessing/wordList.pkl", "rb"))

bin_file = open('Preprocessing/bins.txt', 'r')
bins = []
for line in bin_file:
    bins.append(int(line.rstrip('\n')))

def predict_class(classifier):
    print("Calculating " + classifier)
    try:
        clf = Classifier(classifier)
        class_num = clf.predict_class(features, labels, feature_dict, summary)
        print(classifier + ": class " + str(class_num) + "($" + str(bins[class_num-1]) + "-$" + str(bins[class_num]) + ")")
    except:
        print(classifier + " : error")

p = Pool(5)
p.map(predict_class, classifiers)

bin_file.close()
