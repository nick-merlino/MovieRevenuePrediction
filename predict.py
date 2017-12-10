# Python 3
from Classifiers import Classifier
import argparse
from pickle import load
from Preprocessing.process_text import process, extract_key_phrases
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

parser = argparse.ArgumentParser(description='Predict the revenue of a plot summary.')
parser.add_argument('summary_file', help='Path to the file containing the summary')
parser.add_argument('-c', help='DecisionTrees / LinearRegression / NaiveBayes / NeuralNetwork / SVM / RandomForest')
args = parser.parse_args()

classifiers = [args.c] if args.c else ["AdaBoost", "DecisionTrees", "GaussianProcess", "LogisticRegression", "LinearSVM", "NaiveBayes", "NearestNeighbors", "NeuralNetwork", "RandomForest", "RBFSVM", "QDA"]

summary = process(open(args.summary_file, 'r').read())

dense_vector = load(open("Preprocessing/preproc.pkl", "rb"))
revenue_classes = load(open("Preprocessing/revenue_classes.pkl", "rb"))
summary_vocab = load(open("Preprocessing/summary_vocab.pkl", "rb"))
keyword_vocab = load(open("Preprocessing/keyword_vocab.pkl", "rb"))

vec_summary = TfidfVectorizer(vocabulary=summary_vocab, ngram_range=(1, len_n_grams), strip_accents='unicode', decode_error='ignore', analyzer='word', tokenizer=process)
vec_summary_fit_transformed = vec_summary.fit_transform(summary)
vec_keywords = TfidfVectorizer(vocabulary=keyword_vocab, strip_accents='unicode', decode_error='ignore', tokenizer=extract_key_phrases)
vec_keywords_fit_transformed = vec_keywords.fit_transform(summary)
vec_combined = hstack([vec_summary_fit_transformed, vec_keywords_fit_transformed])

bin_dict = load(open('Preprocessing/bin_dict.pkl", "rb"))
for classifier in classifiers:
    print("Calculating " + classifier)
    try:
        clf = Classifier(classifier)
        class_num = clf.predict_class(dense_vector, revenue_classes, vec_combined)
        print(classifier + ": class " + str(class_num) + "($" + str(bin_dict[class_num-1]) + "-$" + str(bin_dict[class_num]) + ")")
    except:
        print(classifier + " : error")

bin_file.close()
