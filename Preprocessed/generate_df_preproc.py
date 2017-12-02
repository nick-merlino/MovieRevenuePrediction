# Python 3

import pandas as pd # pip3 intsall pandas
import progressbar # pip3 install progressbar2
from process_text import process_text

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
revenue_class = pd.qcut(plot_summaries_and_revenue['Movie box office revenue'],num_output_classes)
plot_summaries_and_revenue['Class'] = revenue_class

# Create empty list for lemmatized plot in each summary object
plot_summaries_and_revenue['Plot Lemmatized'] = pd.np.empty((len(plot_summaries_and_revenue), 0)).tolist()

one_percent = int(len(plot_summaries_and_revenue)/100)
progress = 0
for index, row in plot_summaries_and_revenue.iterrows():
    if index % one_percent == 0:
        bar.update(progress)
        progress += 1

    row['Plot Lemmatized'] = process_text(row['Plot Summary'])

# Write to file
plot_summaries_and_revenue.to_pickle('preproc.pkl')
