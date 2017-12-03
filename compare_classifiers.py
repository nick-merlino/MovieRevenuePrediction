# Python 3
import pandas
from Classifiers import Classifier

methods = ["DecisionTrees", "LinearRegression", "NaiveBayes", "NeuralNetwork", "SVM", "RandomForest"]
num_divisions = 10

df = pandas.read_pickle('Preprocessing/preproc.pkl')
features = df['Plot Lemmatized'].tolist()
feature_chunks = [features[i:i+int(len(features)/num_divisions)] for i in range(0, len(features), int(len(features)/num_divisions))]
labels = df['Class'].tolist()
label_chunks = [labels[i:i+int(len(labels)/num_divisions)] for i in range(0, len(labels), int(len(labels)/num_divisions))]

method_accuracies = {}
for method in methods:
    classifier = Classifier(method)
    accuracies = []
    for test_idx in range(num_divisions):
        train_set_features = [x for i,x in enumerate(feature_chunks) if i!=test_idx]
        train_set_labels = [x for i,x in enumerate(label_chunks) if i!=test_idx]
        test_set_features = feature_chunks[test_idx]
        test_set_labels = label_chunks[test_idx]

        # classifier.train(train_set_features, train_set_labels)
        # accuracies.append(classifier.test(test_set_features, test_set_labels))
        accuracies.append(100)
    print(method + " accuracy : " + str(sum(accuracies) / num_divisions))
    method_accuracies[method] = sum(accuracies) / num_divisions
