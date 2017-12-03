# Python 3
import pandas

from Classification.DecisionTrees import predict_DecisionTrees

df = pandas.read_pickle('Preprocessing/preproc.pkl')

features = df['Plot Lemmatized'].tolist()
labels = df['Class'].tolist()
# print(predict_DecisionTrees(features, labels, "this is a test"))
