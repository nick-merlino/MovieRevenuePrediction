# Python 3
from Classifiers import Classifier
import argparse
from pickle import load
from Preprocessing.process_text import process, extract_key_phrases
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from numpy import mean
import locale

len_n_grams = 5
locale.setlocale( locale.LC_ALL, '' )

parser = argparse.ArgumentParser(description='Predict the revenue of a plot summary.')
parser.add_argument('summary_file', help='Path to the file containing the summary')
parser.add_argument('-c', help='Specify classifier')
args = parser.parse_args()

classifiers = [args.c] if args.c else ["DecisionTrees5", "DecisionTrees10", "DecisionTrees20", "RandomForestClassifier5", "RandomForestClassifier10", "RandomForestClassifier20", "MultinomialNaiveBayes1", "MultinomialNaiveBayes0.1", "MultinomialNaiveBayes5", "LogisticRegressionSagaL1", "LogisticRegressionSagaL2", "AdaBoost", "RidgeClassifier", "LinearSVC", "SGDClassifier", "Perceptron", "PassiveAggressiveClassifier"]
summary = [' '.join(process(open(args.summary_file, 'r').read()))]
del parser
del args

vec_freq = load(open("Preprocessing/preproc-freq.pkl", "rb"))
vec_binary = load(open("Preprocessing/preproc-binary.pkl", "rb"))
vec_keywords = load(open("Preprocessing/preproc-keywords.pkl", "rb"))
train_vec_freq = hstack([vec_freq, vec_keywords])
train_vec_binary = hstack([vec_binary, vec_keywords])
train_labels = load(open("Preprocessing/train_labels.pkl", "rb"))
vocab_summary = load(open("Preprocessing/vocab-summary.pkl", "rb"))
vocab_keywords = load(open("Preprocessing/vocab-keywords.pkl", "rb"))

print("Calculating and vectorizing keywords")
vectorizer = TfidfVectorizer(vocabulary=vocab_keywords, strip_accents='unicode', decode_error='ignore', tokenizer=extract_key_phrases)
vec_keywords = vectorizer.fit_transform(summary)
print("Vectorizing the summary (binary)")
vectorizer = TfidfVectorizer(vocabulary=vocab_summary, ngram_range=(1, len_n_grams), binary=True, strip_accents='unicode', decode_error='ignore', analyzer='word', tokenizer=process)
vec_binary = vectorizer.fit_transform(summary)
print("Vectorizing the summary (frequency)")
vectorizer = TfidfVectorizer(vocabulary=vocab_summary, ngram_range=(1, len_n_grams), strip_accents='unicode', decode_error='ignore', analyzer='word', tokenizer=process)
vec_freq = vectorizer.fit_transform(summary)
test_vec_freq = hstack([vec_freq, vec_keywords])
test_vec_binary = hstack([vec_binary, vec_keywords])

del vectorizer
del vec_freq
del vec_binary
del vec_keywords

train_classes_dict = load(open("Preprocessing/train_classes_dict.pkl", "rb"))
predictions = []
for classifier in classifiers:
    print("Calculating " + classifier)
    try:
        clf = Classifier(classifier, train_labels, train_vec_freq, train_vec_binary, test_vec_freq, test_vec_binary)
        pred = clf.predict_class()
        print(classifier + ": class " + str(pred) + ' (' + locale.currency(train_classes_dict[pred], grouping=True) + ' - ' + locale.currency(train_classes_dict[pred+1], grouping=True) + ')')
        predictions.append(pred)
    except Exception as e:
        print(classifier + ": " + str(e))

pred = int(round(mean(predictions)))
print("Average Prediction: class " + str(pred) + ' (' + locale.currency(train_classes_dict[pred], grouping=True) + ' - ' + locale.currency(train_classes_dict[pred+1], grouping=True) + ')')
