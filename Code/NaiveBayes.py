import pandas as pd
import numpy as np
from nltk import FreqDist
import operator
from utils import get_wordnet_pos
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer


class NBClf:
    # Implementation based off
    # https://web.stanford.edu/~jurafsky/slp3/6.pdf

    def __init__(self,training_df):
        self.df = training_df
        self.stop_list = stopwords.words('english')
        self.wordnet_lemmatizer = WordNetLemmatizer()



    def train(self):
        self.word_counts = FreqDist()

        for index, row in self.df.iterrows():
            for lemma in row['Plot Lemmatized']:
                self.word_counts[lemma] += 1

        self.vocab = set(self.word_counts.keys())

        self.n_tokens = sum(self.word_counts.values())
        self.n_vocab = len(self.vocab)
        self.n_doc = len(self.df)
        self.classes = self.df['Class'].unique()

        print('Number of tokens:', self.n_tokens)
        print('Size of vocab:', self.n_vocab)
        print('Number of documents:', self.n_doc)

        self.log_prior = {}
        bigdoc = {}
        self.log_likelihood = {}
        for c in self.classes:
            print('Working on class:',c)
            N_c = len(self.df[self.df['Class']==c])
            self.log_prior[c] = np.log(N_c/self.n_doc)
            fd = FreqDist()
            for d in self.df[self.df['Class'] == c]['Plot Lemmatized']:
                fd.update(d)
            for word in self.vocab:
                count_w_c = fd[word]
                denominator = sum([(fd[w_prime]+1) for w_prime in self.vocab])
                self.log_likelihood[(word,c)] = np.log((count_w_c+1) / denominator)


    def predict(self,test_summary):
        test_lemmas = []
        for word,tag in pos_tag(test_summary):
            if word.lower() not in self.stop_list and word.isalpha():
                lemma = self.wordnet_lemmatizer.lemmatize(word,pos=get_wordnet_pos(tag))
                test_lemmas.append(lemma)
        total = {}
        for c in self.classes:
            total[c] = self.log_prior[c]
            for word in test_lemmas:
                if word in self.vocab:
                    total[c] += self.log_likelihood[(word,c)]
        #return max(total, key=total.get)
        return max(total.items(), key=operator.itemgetter(1))[0]


df = pd.read_pickle('../Preprocessed/preproc.pkl')
clf = NBClf(df.iloc[2199:2200])
clf.train()
print("Predicted:",clf.predict(df['Plot Summary'].iloc[100]),"Actual:",df['Class'].iloc[100])
