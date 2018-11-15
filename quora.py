import numpy as np
import pandas as pd
import os
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def get_coefs(word, *arr): 
    return word, np.asarray(arr, dtype='float32')

def load_embedding(method=2):
    if method == 0:
        embedding_path = "../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt"
        embedding_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_path))
    """
    if method == 1:
        embedding_path = "../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec"
    """
    if method == 2:
        embedding_path = "../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"
        embedding_index = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
    if method == 3:
        embedding_path = "../input/embeddings/glove.840B.300d/glove.840B.300d.txt"
        embedding_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_path))
    return embedding_index

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)
    """
    return

def get_submit():
    return

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Train shape : ",train.shape)
print("Test shape : ",test.shape)

train_df, val_df = train_test_split(train, test_size=0.05)