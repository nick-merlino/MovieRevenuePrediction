# Python 3

import pandas as pd # pip3 intsall pandas
from pickle import dump
import progressbar # pip3 install progressbar2
from process_text import process
from collections import defaultdict
import sys
# USER DEFINED
num_output_classes = 5
bar = progressbar.ProgressBar(max_value=100, widgets=[progressbar.Percentage(), progressbar.Bar(), ' [', progressbar.Timer(), '] ', ' (', progressbar.ETA(), ') ',
])

# Read in movie metadata and plot summaries
movie_headers =['Wikipedia movie ID', 'Freebase movie ID', 'Movie name','Movie release date','Movie box office revenue',
        'Movie runtime','Movie languages','Movie countries (Freebase ID:name tuples)','Movie genres']
movie_metadata = pd.read_csv('MovieSummaries/movie.metadata.tsv',sep='\t',header=None,names=movie_headers,index_col=0)
plot_summaries = pd.read_csv('MovieSummaries/plot_summaries.txt',sep='\t',index_col=0,names=['Wikipedia movie ID','Plot Summary'])
joined = plot_summaries.join(movie_metadata)

# Only keep objects with both revenue and summary
plot_summaries_and_revenue = joined[['Plot Summary','Movie box office revenue']].dropna()

# Determine the revenue class ranges and assign the class to each object
revenue_class, bins = pd.qcut(plot_summaries_and_revenue['Movie box office revenue'],num_output_classes, labels=[1,2,3,4,5], retbins=True)

# Assign class to each movie
plot_summaries_and_revenue['Class'] = revenue_class

# Write categories to file
bin_file = open('bins.txt', 'w')
for b in bins:
    bin_file.write(str(int(b)) + '\n')
bin_file.close()
sys.exit(0)
# Create empty list for lemmatized plot in each summary object
plot_summaries_and_revenue['Plot Lemmatized'] = pd.np.empty((len(plot_summaries_and_revenue), 0)).tolist()

# Setup for loading bar
two_percent = int(len(plot_summaries_and_revenue)/50)
progress = 0
bar_progress = 0

full_dictionary = defaultdict( int )
for index, row in plot_summaries_and_revenue.iterrows():
    # Update loading bar
    progress += 1
    if progress % two_percent == 0:
        bar.update(bar_progress)
        bar_progress += 1

    # Insert processed text at each movie
    processed_text = process(row['Plot Summary'])
    plot_summaries_and_revenue.at[index, 'Plot Lemmatized'] = processed_text

    # Update full dictionary to contain all words
    for w in processed_text:
        full_dictionary[w] = 1

# Create empty list for feature vector in each summary object
plot_summaries_and_revenue['Feature Vector'] = pd.np.empty((len(plot_summaries_and_revenue), 0)).tolist()

# Generate empty feature vector from all known words
empty_feature_vector = dict.fromkeys(full_dictionary, 0)
dump( empty_feature_vector, open( "wordList.pkl", "wb" ) )

for index, row in plot_summaries_and_revenue.iterrows():
    # Update loading bar
    progress += 1
    if index % two_percent == 0:
        bar.update(bar_progress)
        bar_progress += 1

    # Update feature vector per movie
    feature_vector = empty_feature_vector
    for w in row['Plot Lemmatized']:
        feature_vector[w] = 1
    plot_summaries_and_revenue.at[index, 'Feature Vector'] = [feature_vector[key] for key in sorted([*feature_vector])]
    # for key, value in sorted(feature_vector.items()):
    #    plot_summaries_and_revenue.at[index, 'Feature Vector'].append(value)

# Write to file the class and feature vector
plot_summaries_and_revenue.drop('Plot Summary', axis=1, inplace=True)
plot_summaries_and_revenue.drop('Movie box office revenue', axis=1, inplace=True)
plot_summaries_and_revenue.drop('Plot Lemmatized', axis=1, inplace=True)

plot_summaries_and_revenue.to_pickle('preproc.pkl')
