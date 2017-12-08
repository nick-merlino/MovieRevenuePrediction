# Python 3

import pandas as pd # pip3 intsall pandas
from pickle import dump
import progressbar # pip3 install progressbar2
from process_text import process
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from pickle import dump
from textrank import extract_key_phrases # pip3 install git+git://github.com/davidadamojr/TextRank.git
from sklearn.pipeline import FeatureUnion

# USER DEFINED
num_output_classes = 3
len_n_grams = 1

# Show progress
bar = progressbar.ProgressBar(max_value=5, widgets=[progressbar.Percentage(), progressbar.Bar(), ' [', progressbar.Timer(), ']'])
bar.update(1)

# Read in movie metadata and plot summaries
movie_headers =['Wikipedia movie ID', 'Freebase movie ID', 'Movie name','Movie release date','Movie box office revenue',
        'Movie runtime','Movie languages','Movie countries (Freebase ID:name tuples)','Movie genres']
movie_metadata = pd.read_csv('MovieSummaries/movie.metadata.tsv',sep='\t',header=None,names=movie_headers,index_col=0)
plot_summaries = pd.read_csv('MovieSummaries/plot_summaries.txt',sep='\t',index_col=0,names=['Wikipedia movie ID','Plot Summary'])
joined = plot_summaries.join(movie_metadata)
plot_summaries_and_revenue = joined[['Plot Summary','Movie box office revenue']].dropna().iloc[:10]

# Extract revenue classes and plot summaries
revenue_classes, bins = pd.qcut(plot_summaries_and_revenue['Movie box office revenue'],num_output_classes, labels=list(range(1, num_output_classes+1)), retbins=True)
summaries = plot_summaries_and_revenue['Plot Summary'].tolist()

# Drop memory no longer needed
del plot_summaries_and_revenue

# Create summary overviews
summary_overviews = []
for s in summaries:
    summary_overviews.append(' '.join(extract_key_phrases(s)))

bar.update(2)

# Save revenue class labels
bin_dict = dict()
for i in range(1, num_output_classes+1):
    bin_dict[i] = bins[i-1]

# Vectorize each summary
count_vect = CountVectorizer(ngram_range=(1, len_n_grams), strip_accents='unicode', decode_error='ignore', analyzer='word', tokenizer=process, stop_words='english')
train_counts_summary = count_vect.fit_transform(summaries)

# Vectorize each summary overview
count_vect = CountVectorizer(strip_accents='unicode', decode_error='ignore', analyzer='word', tokenizer=process, stop_words='english')
train_counts_overview = count_vect.fit_transform(summaries)

bar.update(3)

# TODO: Combine vectors
combined_features = FeatureUnion([('train_counts_summary', train_counts_summary), ('train_counts_overview', train_counts_overview)])

# Normalize vector
tfidf_transformer = TfidfTransformer()
tfidf = tfidf_transformer.fit_transform(combined_features)

bar.update(4)

dump(revenue_classes, open("revenue_classes.pkl", "wb"))
dump(tfidf, open("preproc.pkl", "wb"))

bar.update(5)
