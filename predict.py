# Python 3
import pandas
from Classifiers import Classifier
import argparse
from pickle import load

parser = argparse.ArgumentParser(description='Predict the revenue of a plot summary.')
parser.add_argument('summary_file', help='Path to the file containing the summary')
parser.add_argument('-c', help='DecisionTrees / LinearRegression / NaiveBayes / NeuralNetwork / SVM / RandomForest')
args = parser.parse_args()

classifiers = [args.c] if args.c else ["DecisionTrees", "LinearRegression", "NaiveBayes", "NeuralNetwork", "SVM", "RandomForest"]
summary = open(args.summary_file, 'r').read()

df = pandas.read_pickle('Preprocessing/preproc.pkl')
features = df['Feature Vector'].tolist()
labels = [str(lbl) for lbl in df['Class'].tolist()]

bin_file = open('Preprocessing/bins.txt', 'r')
bins = []
for line in bin_file:
    bins.append(int(line.rstrip('\n')))
bin_file.close()

empty_feature_vector = load( open( "Preprocessing/wordList.pkl", "rb" ) )

for classifier in classifiers:
    clf = Classifier(classifier)
    plot_summaries_and_revenue.at[index, 'Feature Vector'] = [feature_vector[key] for key in sorted([*feature_vector])]

    clf.train(features, labels)
    print(classifier + " : " + str(clf.predict(summary)))
