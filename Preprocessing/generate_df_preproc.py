# Python 3

import pandas as pd # pip3 intsall pandas
from pickle import dump
import progressbar # pip3 install progressbar2
from process_text import process, extract_key_phrases
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from pickle import dump
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack

# USER DEFINED
num_output_classes = 3
len_n_grams = 5

# Show progress
bar = progressbar.ProgressBar(value=0, max_value=5, widgets=[progressbar.Percentage(), progressbar.Bar(), ' [', progressbar.Timer(), ']'])

# Read in movie metadata and plot summaries
movie_headers =['Wikipedia movie ID', 'Freebase movie ID', 'Movie name','Movie release date','Movie box office revenue',
        'Movie runtime','Movie languages','Movie countries (Freebase ID:name tuples)','Movie genres']
plot_headers = ['Wikipedia movie ID','Plot Summary']
movie_metadata = pd.read_csv('MovieSummaries/movie.metadata.tsv',sep='\t',header=None,names=movie_headers,index_col=0)
plot_summaries = pd.read_csv('MovieSummaries/plot_summaries.txt',sep='\t',index_col=0,names=plot_headers)
joined = plot_summaries.join(movie_metadata)
plot_summaries_and_revenue = joined[['Plot Summary','Movie box office revenue']].dropna()

# Extract revenue classes and plot summaries
revenue_classes, bins = pd.qcut(plot_summaries_and_revenue['Movie box office revenue'], num_output_classes, labels=list(range(1, num_output_classes+1)), retbins=True)
summaries = plot_summaries_and_revenue['Plot Summary'].tolist()

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

# Vectorize each summary
count_vect_summary = CountVectorizer(ngram_range=(1, len_n_grams), strip_accents='unicode', decode_error='ignore', analyzer='word', tokenizer=process)
train_counts_summary = count_vect_summary.fit_transform(summaries)

bar.update(2)

# Vectorize each summary overview
count_vect_keywords = CountVectorizer(strip_accents='unicode', decode_error='ignore', tokenizer=extract_key_phrases)
train_counts_keywords = count_vect_keywords.fit_transform(summaries)

bar.update(3)

# Combine vectors
combined_features = hstack([train_counts_summary, train_counts_keywords])

# Normalize vector
tfidf_transformer = TfidfTransformer()
tfidf = tfidf_transformer.fit_transform(combined_features)

bar.update(4)

dump(revenue_classes, open("revenue_classes.pkl", "wb"))
dump(tfidf, open("preproc.pkl", "wb"))

bar.update(5)
