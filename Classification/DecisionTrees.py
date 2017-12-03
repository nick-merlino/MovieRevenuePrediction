from sklearn import tree

def predict_DecisionTrees(features, labels, test):
    print(features)
    print(labels)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features, labels)
    return True#clf.predict(test)
