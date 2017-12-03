from sklearn import tree

class Classifier:
    def __init__(self,method):
        if method == "DecisionTrees":
            self.clf = tree.DecisionTreeClassifier()
        elif method == "LinearRegression":
            self.clf = None
        elif method == "NaiveBayes":
            self.clf = None
        elif method == "NeuralNetwork":
            self.clf = None
        elif method == "SVM":
            self.clf = None
        elif method == "RandomForest":
            self.clf = None
        else:
            return False
        self.features = []
        self.labels = []

    def train(df):

        self.clf.fit(features, labels)

    def test(features):
        return self.clf.predict(features)
