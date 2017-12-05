import pandas as pd
import nltk
from nltk import bigrams, FreqDist, word_tokenize, ConditionalFreqDist
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet



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


movie_headers =['Wikipedia movie ID', 'Freebase movie ID', 'Movie name','Movie release date','Movie box office revenue',
        'Movie runtime','Movie languages','Movie countries (Freebase ID:name tuples)','Movie genres']
movie_metadata = pd.read_csv('MovieSummaries/movie.metadata.tsv',sep='\t',header=None,names=movie_headers,index_col=0)

plot_summaries = pd.read_csv('MovieSummaries/plot_summaries.txt',sep='\t',index_col=0,names=['Wikipedia movie ID','Plot Summary'])

joined = plot_summaries.join(movie_metadata)

plot_summaries_and_revenue = joined[['Plot Summary','Movie box office revenue']].dropna()


revenue_class = pd.qcut(plot_summaries_and_revenue['Movie box office revenue'],num_output_classes)
plot_summaries_and_revenue['Class'] = revenue_class


stop_list = stopwords.words('english')
wordnet_lemmatizer = nltk.WordNetLemmatizer()
plot_summaries_and_revenue['Plot Lemmatized'] = pd.np.empty((len(plot_summaries_and_revenue), 0)).tolist()

number_of_summaries = len(plot_summaries_and_revenue)

for index, row in plot_summaries_and_revenue.iterrows():
    if index %  750 == 0:
        print("another 10 percent done")
    for word,tag in nltk.pos_tag(word_tokenize(row['Plot Summary'])):
        if word.lower() not in stop_list and word.isalpha():
            lemma = wordnet_lemmatizer.lemmatize(word,pos=get_wordnet_pos(tag))
            row['Plot Lemmatized'].append(lemma)

print('Writing to file...')
plot_summaries_and_revenue.to_pickle('ben-preproc.pkl')
