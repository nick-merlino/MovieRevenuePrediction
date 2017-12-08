# Python 3

import pandas as pd # pip3 intsall pandas
from pickle import dump
import progressbar # pip3 install progressbar2
from process_text import process
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from pickle import dump

# USER DEFINED
num_output_classes = 4
len_n_grams = 5
bar = progressbar.ProgressBar(max_value=100, widgets=[progressbar.Percentage(), progressbar.Bar(), ' [', progressbar.Timer(), '] ', ' (', progressbar.ETA(), ') ',
])

# Read in movie metadata and plot summaries
movie_headers =['Wikipedia movie ID', 'Freebase movie ID', 'Movie name','Movie release date','Movie box office revenue',
        'Movie runtime','Movie languages','Movie countries (Freebase ID:name tuples)','Movie genres']
movie_metadata = pd.read_csv('MovieSummaries/movie.metadata.tsv',sep='\t',header=None,names=movie_headers,index_col=0)
plot_summaries = pd.read_csv('MovieSummaries/plot_summaries.txt',sep='\t',index_col=0,names=['Wikipedia movie ID','Plot Summary'])
joined = plot_summaries.join(movie_metadata)
plot_summaries_and_revenue = joined[['Plot Summary','Movie box office revenue']].dropna()

# Determine the revenue class ranges and assign the class to each object
revenue_classes, bins = pd.qcut(plot_summaries_and_revenue['Movie box office revenue'],num_output_classes, labels=list(range(1, num_output_classes+1)), retbins=True)
summaries = plot_summaries_and_revenue['Plot Summary'].tolist()

# Drop useless memory
plot_summaries_and_revenue.drop('Plot Summary', axis=1, inplace=True)
plot_summaries_and_revenue.drop('Movie box office revenue', axis=1, inplace=True)

bin_dict = dict()
for i in range(1, num_output_classes+1):
    bin_dict[i] = bins[i-1]

one_percent = round(len(plot_summaries_and_revenue)/100)
progress = 0
bar_progress = 0
bar.update(bar_progress)
for i in range(len(summaries)):
    progress += 1
    if progress % one_percent == 0:
        bar.update(bar_progress) if bar_progress < 100 else bar.update(100)
        bar_progress += 1
        progress = 0

    summaries[i] = " ".join(process(summaries[i]))
bar.update(100)

count_vect = CountVectorizer(ngram_range=(1, len_n_grams))
train_counts = count_vect.fit_transform(summaries)
tfidf_transformer = TfidfTransformer()
tfidf = tfidf_transformer.fit_transform(train_counts)

dump(revenue_classes, open("revenue_classes.pkl", "wb"))
dump(tfidf, open("preproc.pkl", "wb"))
