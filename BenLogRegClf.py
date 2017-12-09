import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
import pandas as pd
from nltk import FreqDist
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline


df = pd.read_pickle('Preprocessing/ben-preproc.pkl')

bin_map = {}
for ix, c in enumerate(sorted(df['Class'].unique())):
    bin_map[c] = ix

train_y = []
for x in df['Class']:
    train_y.append(bin_map[x])

count_vectorizer = CountVectorizer()
documents = [" ".join(lemmas) for lemmas in df['Plot Lemmatized']]

print('Training LR Classifier...')
clf = LogisticRegressionCV()
pipe = make_pipeline(clf, count_vectorizer)
pipe.fit(count_vectorizer, train_y)
print('Trained LR Classifier')

#count_tf = count_vectorizer.fit(documents)
#counts = count_vectorizer.transform(documents)





#print("Estimated out of Sample Error", cross_val_score(clf, counts, train_y, n_jobs=-1, cv=KFold(5, shuffle=True)))