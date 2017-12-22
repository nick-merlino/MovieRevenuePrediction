import graphviz
from pickle import load
from sklearn import tree

clf = load(open("models/DecisionTrees10.pkl", "rb"))
vocab = load(open("Preprocessing/vocab-summary.pkl", "rb"))
vocab_keywords = load(open("Preprocessing/vocab-keywords.pkl", "rb"))
vocab.extend(vocab_keywords)

graph = tree.export_graphviz(clf, out_file='image.dot', feature_names=vocab, class_names=['1', '2', '3', '4', '5'], filled=True, rounded=True, special_characters=True)

# Then do 'dot -Tpng image.dot -o image.png'
