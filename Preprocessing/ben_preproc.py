import pandas as pd
import nltk
from nltk import bigrams, FreqDist, word_tokenize, ConditionalFreqDist
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
import textrank
import json


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return 'n'

# USER DEFINED
num_output_classes = 5

# Inflation Data
cpi = pd.read_csv('cpi_data.csv', header=1,names=['Year','CPI'])
cpi['Year'] = cpi['Year'].apply(lambda x: x.split('-')[0])
# just take average over the year because movies aren't always specific with release date
avg_cpi = cpi.groupby(['Year']).mean()

# Load movie metadata
movie_headers =['Wikipedia movie ID', 'Freebase movie ID', 'Movie name','Movie release date','Movie box office revenue',
        'Movie runtime','Movie languages','Movie countries','Movie genres']
movie_metadata = pd.read_csv('MovieSummaries/movie.metadata.tsv',sep='\t',header=None,names=movie_headers,index_col=0)

# Load plot summaries
plot_summaries = pd.read_csv('MovieSummaries/plot_summaries.txt',sep='\t',index_col=0,names=['Wikipedia movie ID','Plot Summary'])

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

# Join on Wikipedia ID
joined = plot_summaries.join(movie_metadata)
plot_summaries_and_revenue = joined[['Plot Summary','Movie revenue adjusted']].dropna()

# Split data into classes
revenue_class = pd.qcut(plot_summaries_and_revenue['Movie revenue adjusted'], num_output_classes)
plot_summaries_and_revenue['Class'] = revenue_class

# General plot summary processing
stop_list = stopwords.words('english')
wordnet_lemmatizer = nltk.WordNetLemmatizer()
plot_summaries_and_revenue['Plot Lemmatized'] = pd.np.empty((len(plot_summaries_and_revenue), 0)).tolist()

number_of_summaries = len(plot_summaries_and_revenue)

for index, row in plot_summaries_and_revenue.iterrows():
    if index %  750 == 0:
        print("another 10 percent done")
    for word, tag in nltk.pos_tag(word_tokenize(row['Plot Summary'])):
        if word.lower() not in stop_list and word.isalpha():
            lemma = wordnet_lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag))
            row['Plot Lemmatized'].append(lemma)
    row['Plot Keywords'] = textrank.extract_key_phrases(row['Plot Summary'])

print('Writing to file...')
plot_summaries_and_revenue.to_pickle('ben-preproc_w_txtrnk.pkl')
