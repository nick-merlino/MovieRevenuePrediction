# Python 3
# Used to tokenize, remove stop words, and lemmatize using POS tagging

from nltk import word_tokenize, pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from textrank import filter_for_tags, normalize, unique_everseen, build_graph # pip3 install git+git://github.com/davidadamojr/TextRank.git
from networkx import pagerank

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
        if word not in stop_list and word.isalpha(): # Needs to be in this order for conjuctions such as "woudln't"
            wn_pos = get_wordnet_pos(tag)
            lemma = wnl.lemmatize(word,pos=wn_pos) if wn_pos is not None else wnl.lemmatize(word)
            results.append(lemma)
    return results

def extract_key_phrases(text):
    # tokenize the text using nltk
    word_tokens = word_tokenize(text)

    # assign POS tags to the words in the text
    tagged = pos_tag(word_tokens)
    textlist = [x[0] for x in tagged]
    textlist_lem = []
    for i in range(len(textlist)):
        word = textlist[i]
        wn_pos = get_wordnet_pos(tagged[i][1])
        lemma = wnl.lemmatize(word,pos=wn_pos) if wn_pos is not None else wnl.lemmatize(word)
        textlist_lem.append(lemma)

    tagged = filter_for_tags(tagged)
    tagged = normalize(tagged)

    unique_word_set = unique_everseen([x[0] for x in tagged])
    word_set_list = list(unique_word_set)

    # this will be used to determine adjacent words in order to construct
    # keyphrases with two words

    graph = build_graph(word_set_list)

    # pageRank - initial value of 1.0, error tolerance of 0,0001,
    calculated_page_rank = pagerank(graph, weight='weight')

    # most important words in ascending order of importance
    keyphrases = sorted(calculated_page_rank, key=calculated_page_rank.get,
                        reverse=True)

    # the number of keyphrases returned will be relative to the size of the
    # text (a third of the number of vertices)
    one_third = len(word_set_list) // 3
    keyphrases = keyphrases[0:one_third + 1]

    # take keyphrases with multiple words into consideration as done in the
    # paper - if two words are adjacent in the text and are selected as
    # keywords, join them together
    modified_key_phrases = set([])
    # keeps track of individual keywords that have been joined to form a
    # keyphrase
    dealt_with = set([])
    i = 0
    j = 1
    while j < len(textlist):
        first = textlist[i]
        second = textlist[j]
        if first in keyphrases and second in keyphrases:
            keyphrase = textlist_lem[i] + ' ' + textlist_lem[j]
            modified_key_phrases.add(keyphrase)
            dealt_with.add(first)
            dealt_with.add(second)
        else:
            if first in keyphrases and first not in dealt_with:
                modified_key_phrases.add(textlist_lem[i])

            # if this is the last word in the text, and it is a keyword, it
            # definitely has no chance of being a keyphrase at this point
            if j == len(textlist) - 1 and second in keyphrases and \
                    second not in dealt_with:
                modified_key_phrases.add(textlist_lem[j])

        i = i + 1
        j = j + 1

    return modified_key_phrases
