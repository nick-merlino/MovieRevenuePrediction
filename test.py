import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
import pandas as pd
from nltk import FreqDist
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfTransformer


df = pd.read_pickle('Preprocessing/ben-preproc.pkl')

count_vectorizer = CountVectorizer()
documents = [" ".join(lemmas) for lemmas in df['Plot Lemmatized']]
count_tf = count_vectorizer.fit(documents)
counts = count_vectorizer.transform(documents)

tfidf = TfidfTransformer()
tfidf_transformed = tfidf.fit_transform(counts)

bin_map = {}
for ix, c in enumerate(sorted(df['Class'].unique())):
    bin_map[c] = ix

#word_counts = FreqDist()

#for index, row in df.iterrows():
#    for lemma in row['Plot Lemmatized']:
#        word_counts[lemma] += 1

train_y = []
for x in df['Class']:
    train_y.append(bin_map[x])

print('Training MNB Classifier...')
clf1 = MultinomialNB()
targets = df['Class']
clf1.fit(tfidf_transformed, train_y)

clf2 = MultinomialNB()
clf2.fit(tfidf_transformed, train_y)
print('Trained MNB Classifier')


print("Estimated out of Sample Error", cross_val_score(clf1, counts, train_y, n_jobs=-1, cv=KFold(5, shuffle=True)))
print("Estimated out of Sample Error", cross_val_score(clf2, counts, train_y, n_jobs=-1, cv=KFold(5, shuffle=True)))