import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from nltk import FreqDist


full_df = pd.read_pickle('Preprocessing/ben-preproc.pkl').iloc[0:5100]
df = full_df.iloc[0:5000]
pdf = full_df.iloc[5000:5100]
count_vectorizer = CountVectorizer()
documents = [" ".join(lemmas) for lemmas in full_df['Plot Lemmatized']]
count_tf = count_vectorizer.fit(documents)
counts = count_vectorizer.transform(documents)

print(count_vectorizer.get_feature_names())

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

clf = MultinomialNB()
targets = df['Class']
clf.fit(counts[:5000], train_y)

#print(df['Plot Lemmatized'].iloc[50])
#pred_vectorizer = CountVectorizer()
#pred_counts = count_vectorizer.fit_transform([" ".join(df['Plot Lemmatized'].iloc[50])])
#print("Predicted:", , "Actual:", )

predicted = clf.predict(counts[5000:5100])
total = 0
for ix,y in enumerate(predicted):
    if y == bin_map[full_df['Class'].iloc[5000+ix]]:
        total+= 1
        #print(bin_map[full_df['Class'].iloc[1000+ix]])

print(total/100)
print(predicted)