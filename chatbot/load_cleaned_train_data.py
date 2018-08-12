import pandas as pd
import numpy as np
# from IPython.display import display, HTML
import json

import pickle
import re

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, TimeDistributed
import pickle
# import nltk
import itertools
import collections

from keras.preprocessing import sequence
from keras.models import model_from_json
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.optimizers import Adam

import keras.backend as K

import spacy
nlp = spacy.load('en')
# from keras.utils import preprocessing
# from model.textGenModel import TextGenModel


INPUT_DIR = "/Users/ashwins/gitlab_repo/chatbot/"

# EOS_TAG = " EOS "
EOS_TAG = ""
PAD_TAG = " PAD "

dialogue = []
l_no = []
convo_start = []
convo_end = []


# constant token and params for our models
START_TOKEN = "EOS"
END_TOKEN = "EOS"
UNKNOWN_TOKEN = "UNK"
PADDING_TOKEN = "PAD"

# vocabulary_size = 22285
sent_max_len = 20

# hidden_size = 512
hidden_size = 32
embedding_size = 32
batch_size = 32
stateful = False


corpus_tokens_tmp=[]


def get_words_mappings(tokenized_sentences, vocabulary_size):
    # Using NLTK
    #frequence = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    #vocab = frequence.most_common(vocabulary_size)

    # Using basic counter
    counter = collections.Counter(itertools.chain(*tokenized_sentences))
    vocab = counter.most_common(vocabulary_size)
    index_to_word = [x[0] for x in vocab]
    # Add padding for index 0
    index_to_word.insert(1, START_TOKEN)
    index_to_word.insert(0, PADDING_TOKEN)
    # Append unknown token (with index = vocabulary size + 1)
    index_to_word.append(UNKNOWN_TOKEN)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
    
    return index_to_word, word_to_index


def load_data():

    with open ('corpus_token_tmp.dat', 'rb') as p:
        print('\n\nloading pickle...')
        corpus_tokens_tmp = pickle.load(p)
        print('Example tokenized excerpt: {}'.format(corpus_tokens_tmp[10:20]))

        print(len(corpus_tokens_tmp))
        vocabulary_size = 50000
        index_to_word, word_to_index = get_words_mappings(
                                                [corpus_tokens_tmp], #cause a list of sentences is expected
                                                vocabulary_size)
        vocabulary_size = len(index_to_word)
        print("Vocabulary size = " + str(vocabulary_size))

    df_train_inputs = pd.read_csv('./data/df_train_inputs.csv', encoding='utf-8')

    print("\n\ndf_train_inputs.shape : ", df_train_inputs.shape)
    
    # assert df_train_data.shape == df_train_inputs.shape

    print("\n\ndf_train_inputs.head() : ", df_train_inputs.head())

    length = len(sorted(df_train_inputs['input'],key=len, reverse=True)[0])
    max_sent_len = 20
    # encoder_input_data=np.array([xi+[0]*(length-len(xi)) for xi in df_train_inputs['input']])
    # encoder_input_data = encoder_input_data[:38080, :max_sent_len]

    encoder_input_data=np.array([xi+[0]*(length-len(xi)) for xi in df_train_inputs['input']])
    encoder_input_data = encoder_input_data[:38080, :max_sent_len]
    # encoder_input_data = np.reshape(encoder_input_data, (38080, 20, 1))

    print("\n\nencoder_input_data.shape : ", encoder_input_data.shape)
    print("\n\nencoder_input_data[:1] : ", encoder_input_data[:5])

    length = len(sorted(df_train_inputs['output'],key=len, reverse=True)[0])
    # length = 20
    decoder_input_data=np.array([xi+[0]*(length-len(xi)) for xi in df_train_inputs['output']])
    decoder_input_data = decoder_input_data[:38080, :max_sent_len]
    print("\n\ndecoder_input_data.shape : ", decoder_input_data.shape)
    print("\n\ndecoder_input_data[:1] : ", decoder_input_data[:5])

    length = len(sorted(df_train_inputs['output'],key=len, reverse=True)[0])
    print("\n\nlength : ", length)
    # decoder_target_data = df_train_inputs.apply(lambda x: x[2][1:]+[x[2][0]], axis=1).values
    df_tmp=[]
    for i, row in df_train_inputs.iterrows():
        # print(i)
        try:
            df_tmp.append(row['output'][1:]+[row['output'][0]])
        except:
            df_tmp.append([1])
    # df_tmp = df_train_inputs.apply(lambda x: x[2][1:]+[x[2][0]], axis=1).values
    decoder_target_data=np.array([xi+[0]*(length-len(xi)) for xi in df_tmp])

    decoder_target_data = decoder_target_data[:38080, :max_sent_len]
    decoder_target_data = np.expand_dims(decoder_target_data, -1)
    # decoder_target_data = np.reshape(decoder_target_data, (38080,-1,20))

    print("\n\ndecoder_target_data.shape : ", decoder_target_data.shape)
    print("\n\ndecoder_target_data[:1] : ", decoder_target_data[:5])

    print("Example train input sentence: {}".format(encoder_input_data[0]))
    print("and related output = {}".format(decoder_target_data[0]))

    return encoder_input_data, decoder_input_data, decoder_target_data

def main():
    _, _, _ = load_data()

if __name__ == '__main__':
    main()