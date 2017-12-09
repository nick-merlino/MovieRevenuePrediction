# Python 3

import pandas as pd # pip3 intsall pandas
from pickle import dump
import progressbar # pip3 install progressbar2
from process_text import process, extract_key_phrases
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from pickle import dump
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack
import json

# USER DEFINED
num_output_classes = 5
len_n_grams = 1

# Show progress
bar = progressbar.ProgressBar(value=0, max_value=5, widgets=[progressbar.Percentage(), progressbar.Bar(), ' [', progressbar.Timer(), ']'])

# Inflation Data
cpi = pd.read_csv('cpi_data.csv', header=1,names=['Year','CPI'])
cpi['Year'] = cpi['Year'].apply(lambda x: x.split('-')[0])
# just take average over the year because movies aren't always specific with release date
avg_cpi = cpi.groupby(['Year']).mean()

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

# Constant 2000 dollars
const_2000 = avg_cpi.loc['2000']

# movies made after 1946 and calculate inflation adjusted gross revenue in constant year 2000 dollars
movie_metadata = movie_metadata[(movie_metadata['Movie release year'].notnull()) & (movie_metadata['Movie release year'].astype(int) > 1946)]
movie_metadata['Movie release date cpi'] = movie_metadata['Movie release year'].apply(lambda x: const_2000/avg_cpi.loc[x])
movie_metadata['Movie revenue adjusted'] = movie_metadata['Movie box office revenue']*movie_metadata['Movie release date cpi']

joined = plot_summaries.join(movie_metadata)
plot_summaries_and_revenue = joined[['Plot Summary','Movie revenue adjusted']].dropna()

# Extract revenue classes and plot summaries
revenue_classes, bins = pd.qcut(plot_summaries_and_revenue['Movie revenue adjusted'], num_output_classes, labels=list(range(1, num_output_classes+1)), retbins=True)
summaries = plot_summaries_and_revenue['Plot Summary'].tolist()
dump(revenue_classes, open("revenue_classes.pkl", "wb"))

# Drop memory no longer needed
del plot_summaries_and_revenue
del movie_headers
del movie_metadata
del plot_headers
del plot_summaries

bar.update(1)

# Save revenue class labels
bin_dict = dict()
for i in range(1, num_output_classes+1):
    bin_dict[i] = bins[i-1]

for i in range(len_n_grams):
    # Vectorize each summary
    count_vect_summary = CountVectorizer(ngram_range=(1, len_n_grams), strip_accents='unicode', decode_error='ignore', analyzer='word', tokenizer=process)
    train_counts_summary = count_vect_summary.fit_transform(summaries)

    dump(train_counts_summary, open('cvpreproc' + i + '.pkl', "wb"))

    bar.update(2)

    # Vectorize each summary overview
    # count_vect_keywords = CountVectorizer(strip_accents='unicode', decode_error='ignore', tokenizer=extract_key_phrases)
    # train_counts_keywords = count_vect_keywords.fit_transform(summaries)

    bar.update(3)

    # Combine vectors
    # combined_features = hstack([train_counts_summary, train_counts_keywords])
    combined_features = train_counts_summary

    # Normalize vector
    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(combined_features)

    bar.update(4)

    dump(tfidf, open('tfidfpreproc' + i + '.pkl', "wb"))

    bar.update(5)
