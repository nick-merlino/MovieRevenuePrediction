# Python 3

import pandas as pd # pip3 intsall pandas
from pickle import dump
from process_text import process, extract_key_phrases
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import json
import locale
import sys
# USER DEFINED
num_output_classes = 5
len_n_grams = 5
locale.setlocale( locale.LC_ALL, '' )

print("Reading in movie data")

# Read in movie metadata and plot summaries
movie_headers =['Wikipedia movie ID', 'Freebase movie ID', 'Movie name','Movie release date','Movie box office revenue',
        'Movie runtime','Movie languages','Movie countries','Movie genres']
plot_headers = ['Wikipedia movie ID','Plot Summary']
movie_metadata = pd.read_csv('MovieSummaries/movie.metadata.tsv',sep='\t',header=None,names=movie_headers,index_col=0)
plot_summaries = pd.read_csv('MovieSummaries/plot_summaries.txt',sep='\t',index_col=0,names=plot_headers)

# Only look at English movies in the US from after 1946
movie_metadata['Movie languages'] = movie_metadata['Movie languages'].apply(lambda x: 1 if "English Language" in json.loads(x).values() else 0)
movie_metadata['Movie countries'] = movie_metadata['Movie countries'].apply(lambda x: 1 if "United States of America" in json.loads(x).values() else 0)
movie_metadata = movie_metadata.dropna(subset=['Movie release date'])
movie_metadata['Movie release year'] = movie_metadata['Movie release date'].apply(lambda x: str(x).split('-')[0])
movie_metadata = movie_metadata[(movie_metadata['Movie languages'] == 1) & (movie_metadata['Movie countries'] == 1)]

print("Determining revenue classes from inflation adjusted revenues")

# Inflation Data
cpi = pd.read_csv('cpi_data.csv', header=1,names=['Year','CPI'])
cpi['Year'] = cpi['Year'].apply(lambda x: x.split('-')[0])

# just take average over the year because movies aren't always specific with release date
avg_cpi = cpi.groupby(['Year']).mean()

# Constant 2000 dollars
const_2000 = avg_cpi.loc['2000']

# movies made after 1946 and calculate inflation adjusted gross revenue in constant year 2000 dollars
movie_metadata = movie_metadata[(movie_metadata['Movie release year'].notnull()) & (movie_metadata['Movie release year'].astype(int) > 1946)]
movie_metadata['Movie release date cpi'] = movie_metadata['Movie release year'].apply(lambda x: const_2000/avg_cpi.loc[x])
movie_metadata['Movie revenue adjusted'] = movie_metadata['Movie box office revenue']*movie_metadata['Movie release date cpi']

joined = plot_summaries.join(movie_metadata)
plot_summaries_and_revenue = joined[['Plot Summary','Movie revenue adjusted']].dropna()

# Extract revenue classes and plot summaries
train_labels, train_classes = pd.qcut(plot_summaries_and_revenue['Movie revenue adjusted'], num_output_classes, labels=list(range(1, num_output_classes+1)), retbins=True)
summaries = plot_summaries_and_revenue['Plot Summary'].tolist()
dump(train_labels, open("train_labels.pkl", "wb"))

# Save revenue class labels
train_classes_dict = dict()
for i in range(num_output_classes):
    train_classes_dict[i+1] = str(locale.currency(train_classes[i], grouping=True) + ' - ' + locale.currency(train_classes[i+1], grouping=True))
dump(train_classes_dict, open("train_classes_dict.pkl", "wb"))

# Drop memory no longer needed
del joined
del plot_summaries_and_revenue
del movie_headers
del const_2000
del avg_cpi
del cpi
del num_output_classes
del movie_metadata
del plot_headers
del plot_summaries
del train_labels
del train_classes

print("Calculating and vectorizing keywords")
vectorizer = CountVectorizer(strip_accents='unicode', decode_error='ignore', tokenizer=extract_key_phrases)
vec = vectorizer.fit_transform(summaries)
dump(vec, open('preproc-keywords.pkl', "wb"))
dump(vectorizer.get_feature_names(), open('vocab-keywords.pkl', "wb"))

print("Vectorizing each summary")
vectorizer = TfidfVectorizer(ngram_range=(1, len_n_grams), strip_accents='unicode', decode_error='ignore', analyzer='word', tokenizer=process)
vec = vectorizer.fit_transform(summaries)
dump(vec, open('preproc-freq.pkl', "wb"))
dump(vectorizer.get_feature_names(), open('vocab-summary.pkl', "wb"))

print("Saving binary version of summary vector")
vectorizer = TfidfVectorizer(ngram_range=(1, len_n_grams), binary=True, strip_accents='unicode', decode_error='ignore', analyzer='word', tokenizer=process)
vec = vectorizer.fit_transform(summaries)
dump(vec, open('preproc-binary.pkl', "wb"))
