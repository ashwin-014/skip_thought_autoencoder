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

    with open (INPUT_DIR + 'movie_data/movie_lines.txt', 'rb') as f:
        for l in f:
    #         print(l)
    #         print(json.dumps(l))
            l = l.decode('windows-1252')
            dialogue.append(EOS_TAG + l.split('+++$+++')[-1].lstrip().rstrip() + EOS_TAG)
            l_no.append(l.split('+++$+++')[0].lstrip().rstrip())

    df_lines = pd.DataFrame({'dialogue': dialogue, 'l_no': l_no})
    print("\n\ndf_lines : ", df_lines)


    with open (INPUT_DIR + '/movie_data/movie_conversations.txt') as f:
        for i, l in enumerate(f):
    #         if i==10:
    #             break
    #         print(type(l.split('+++$+++')[-1].lstrip().rstrip()))
    #         print(l.split('+++$+++')[-1].lstrip().rstrip())
    #         print(l.split('+++$+++')[-1].lstrip().rstrip().strip("[]").split(','))
            el = l.split('+++$+++')[-1].lstrip().rstrip().strip("[]").split(',')
            if len(el) == 2 :
    #             print("sdsd")
                for i, ele in enumerate(el):
    #                 print(ele)
                    ele = ele.lstrip().rstrip()
                    ele = ele.replace("'", "")
    #                 print(ele)
                    el[i] = ele
    #             print("------------>", el)
                convo_start.append(el[0])
                convo_end.append(el[1])
    #         print(json.loads(json.dumps(l.split('+++$+++')[-1].lstrip().rstrip()))[0])
    #         l_no.append(l.split('+++$+++')[0].lstrip().rstrip())


    df_convos = pd.DataFrame({'start': convo_start, 'end':convo_end})
    print("\n\ndf_convos : ", df_convos)

    df_lines['l_no_int'] = df_lines['l_no'].str[1:].astype(int)
    df_lines.head()
    # 
    df_lines.sort_values(by =['l_no_int'], ascending=True, inplace=True)
    df_lines.head()

    with open(INPUT_DIR + 'movie_data/movie_conversations_temp.txt', 'w') as f:
    #     for i, row in df_lines.iterrows():
        f.writelines(df_lines['dialogue'])

    str2=''
    with open(INPUT_DIR + 'movie_data/movie_conversations_temp.txt', 'r') as f:
        corpus_text_2 = f.readlines()
    #     print(corpus_text_2[0])
        for c in corpus_text_2:
            str2 = ' '.join([str2, c])

    print(len(str2))
    str2[:200]
    corpus_text3 = str2

    # for i in range(0, 16000000, 100000):
    #     print(i)
    #     tmp_corpus = [token.orth_ for token in nlp(corpus_text3[i:i + 100000])]
    #     corpus_tokens_tmp = corpus_tokens_tmp + tmp_corpus
    


    # with open('corpus_token_tmp.dat', 'wb') as p:
    #     pickle.dump(corpus_tokens_tmp, p)


    with open ('corpus_token_tmp.dat', 'rb') as p:
        corpus_tokens_tmp = pickle.load(p)
    print('Example tokenized excerpt: {}'.format(corpus_tokens_tmp[10:20]))

    print(len(corpus_tokens_tmp))
    vocabulary_size = 50000
    index_to_word, word_to_index = get_words_mappings(
                                            [corpus_tokens_tmp], #cause a list of sentences is expected
                                            vocabulary_size)
    vocabulary_size = len(index_to_word)
    print("Vocabulary size = " + str(vocabulary_size))

    df_train_data = pd.merge(df_lines, df_convos, left_on=['l_no'], right_on=['start'])
    print(df_train_data.head())
    df_train_data = pd.merge(df_train_data, df_lines, left_on=['end'], right_on=['l_no'])
    # df_train_data['end'] = df_train_data[df_convos.loc]
    print(df_train_data.head())

    df_train_data.drop(columns=['l_no_int_x', 'l_no_x', 'l_no_int_y', 'l_no_y'], inplace=True)
    print(df_train_data.shape)
    df_train_data.head()

    cols = df_train_data.columns
    cols = [cols[2], cols[0], cols[1], cols[3]]
    df_train_data = df_train_data[cols]
    df_train_data.rename(index=str, columns = {'dialogue_x' : 'input', 'dialogue_y' : 'output'}, inplace=True)
    df_train_data.head()


    df_train_inputs = pd.DataFrame()

    train_list = []
    for i, row in df_train_data.iterrows() :
        train_inputs = []
        train_outputs = []
        # convert tokens to indexes (and replacing unknown words)
    #     print([w for w in row[1]])
    #     print(type(row[1]))
        train_inputs = [word_to_index.get(w, word_to_index[UNKNOWN_TOKEN]) for w in re.sub("[^\w]", " ", row[1]).split()]
        train_outputs = [word_to_index.get(w, word_to_index[UNKNOWN_TOKEN]) for w in re.sub("[^\w]", " ", row[3]).split()]
        
    #     print("start : ", row[1])
    #     print(train_inputs)
    #     print("end :", row[3])
    #     print(train_outputs)
        train_series = pd.DataFrame(data= {'start' : row[0], 'input' : [train_inputs], 'end' : row[2], 'output' : [train_outputs]})
        train_list.append(train_series)

    df_train_inputs = pd.concat(train_list, ignore_index=True)

    print("\n\ndf_train_inputs.shape : ", df_train_inputs.shape)
    assert df_train_data.shape == df_train_inputs.shape
    print("\n\ndf_train_inputs.head() : ", df_train_inputs.head())

    length = len(sorted(df_train_inputs['input'],key=len, reverse=True)[0])
    max_sent_len = 20
    encoder_input_data=np.array([xi+[0]*(length-len(xi)) for xi in df_train_inputs['input']])
    encoder_input_data = encoder_input_data[:38080, :max_sent_len]
    print("\n\nencoder_input_data.shape : ", encoder_input_data.shape)
    print("\n\nencoder_input_data[:5] : ", encoder_input_data[:5])

    length = len(sorted(df_train_inputs['output'],key=len, reverse=True)[0])
    # length = 20
    decoder_input_data=np.array([xi+[0]*(length-len(xi)) for xi in df_train_inputs['output']])
    decoder_input_data = decoder_input_data[:38080, :max_sent_len]
    print("\n\ndecoder_input_data.shape : ", decoder_input_data.shape)
    print("\n\ndecoder_input_data[:5] : ", decoder_input_data[:5])

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
    # decoder_target_data = np.expand_dims(decoder_target_data, -1)
    decoder_target_data = tf.reshape(decoder_target_data, [38080,-1,20])

    print("\n\ndecoder_target_data.shape : ", decoder_target_data.shape)
    print("\n\ndecoder_target_data[:5] : ", decoder_target_data[:5])

    print("Example train input sentence: {}".format(encoder_input_data[0]))
    print("and related output = {}".format(decoder_target_data[0]))

    return encoder_input_data, decoder_input_data, decoder_target_data

def main():
    _, _, _ = load_data()

if __name__ == '__main__':
    main()