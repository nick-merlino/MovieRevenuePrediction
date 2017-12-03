# Python 3
# Used to tokenize, remove stop words, and lemmatize using POS tagging

from nltk import word_tokenize, pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

# Define stopword list
stop_list = stopwords.words('english')
# Define lematizing method as wornet lemmatizer in order to use part of speech
wnl = WordNetLemmatizer()

# Convert nltk POS tag to basic ones necessary for wordnet lammatizer
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
        return None

def process(summary):
    results = []
    for word,tag in pos_tag(word_tokenize(summary)):
        word = word.lower()
        if word not in stop_list and word.isalpha(): # Needs to be in this order for conjuctions such as "woudln't"
            wn_pos = get_wordnet_pos(tag)
            lemma = wnl.lemmatize(word,pos=wn_pos) if wn_pos is not None else wnl.lemmatize(word)
            results.append(lemma)
    return results
