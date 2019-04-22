from data_preprocessing import *
from generate_plot import *
from gensim.models.word2vec import Word2Vec
import pandas as pd

###################################################################
# TODO: Implement the function which returns data from .csv crawled article
###################################################################
df = pd.read_csv("./article_1904.csv", sep=",", encoding='euc-kr')
data = df['article'].tolist()
# take article context list, return the corpus in the total context
corpus = generate_corpus(data)
# print(corpus)

# import pickle
# f = open('corpus.pickle', 'wb')
# pickle.dump(corpus, f)
# f.close

# f = open('corpus.pickle', 'rb')
# corpus = pickle.load(f)
# f.close()
# print('load complete')
###################################################################
# TODO: Implement gensim  Word2Vec model with our written solution
###################################################################
embedding_model = Word2Vec(corpus, size=100, window=2, min_count=2, workers=4, iter=100, sg=1, sample=1e-3)

# def long_word(word):
#      return len(word) > 2 and not word.isdigit()


# print(embedding_model.wv.vocab)
w2c = dict()

# vocab = list(filter(long_word, embedding_model.wv.vocab))
vocab = embedding_model.wv.index2entity[:500]
# print(vocab)
X_w2v = embedding_model[vocab]

###################################################################
# input:
#    vocab : word list
#    X_w2v : corresponding w2v list
# return:
#    draw the plot w.r.t. X_w2v
###################################################################
generate_plot(vocab, X_w2v)
