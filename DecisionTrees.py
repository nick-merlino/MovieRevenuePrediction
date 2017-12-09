from sklearn import tree
import pandas as pd
import graphviz
from pickle import load

df = pd.read_pickle('Preprocessing/preproc.pkl')

train_x = df['Feature Vector'].tolist()
train_y = df['Class'].tolist()

clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=10)
clf.fit(train_x, train_y)

wordlist = load(open('Preprocessing/wordList.pkl', 'rb'))


clf_data = tree.export_graphviz(clf, out_file='graph',
                         feature_names=sorted(wordlist.keys()),
                         class_names=['1','2','3','4','5'],
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(clf_data)
graph

