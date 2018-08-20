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

import pandas as pd
import numpy as np

INPUT_DIR = ''

str2=''


START_TOKEN = "EOS"
END_TOKEN = "EOS"
UNKNOWN_TOKEN = "UNK"
PADDING_TOKEN = "PAD"

UNKNOWN_TOKEN = "UNK"
PAD_TOKEN = "PAD"
EOS_TOKEN = "EOS"

vocabulary_size = 50003

max_sent_len = 20

# hidden_size = 512
# hidden_size = 128
# embedding_size = 128
# batch_size = 64

embedding_size = 128
stateful = False


class inference():

    def get_words_mappings(self,tokenized_sentences, vocabulary_size):
        
        counter = collections.Counter(itertools.chain(*tokenized_sentences))
        vocab = counter.most_common(vocabulary_size)
        index_to_word = [x[0] for x in vocab]
        # Add padding for index 0
        index_to_word.insert(0, START_TOKEN)
        index_to_word.insert(1, PADDING_TOKEN)
        # Append unknown token (with index = vocabulary size + 1)
        index_to_word.append(UNKNOWN_TOKEN)
        word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
        
        return index_to_word, word_to_index


    def __init__(self):

        # with open(INPUT_DIR + 'movie_data/movie_conversations_temp.txt', 'r') as f:
        #     corpus_text_2 = f.readlines()
        # #     print(corpus_text_2[0])
        #     for c in corpus_text_2:
        #         str2 = ' '.join([str2, c])

        # print(len(str2))
        # str2[:200]

        # with open ('./data/corpus_token_tmp.dat', 'rb') as p:
        #     corpus_tokens_tmp = pickle.load(p)

        # print(len(corpus_tokens_tmp))
        # vocabulary_size = 5000
        # index_to_word, word_to_index = get_words_mappings(
        #                                         [corpus_tokens_tmp], #cause a list of sentences is expected
        #                                         vocabulary_size)
        # vocabulary_size = len(index_to_word)
        # print("Vocabulary size = " + str(vocabulary_size))

        # df_word_to_index = pd.DataFrame.from_dict(word_to_index, orient='index')
        # index_to_word_dict = dict([(i,w) for w,i in word_to_index.items()])
        # df_index_to_word = pd.DataFrame.from_dict(index_to_word_dict, orient='index')

        # print(df_word_to_index.head())
        # print(df_index_to_word.head())

        # df_word_to_index.to_csv('w2i.csv', encoding='utf-8')
        # df_index_to_word.to_csv('i2w.csv', encoding='utf-8')

        df_word_to_index = pd.read_csv('w2i.csv', encoding='utf-8', index_col =0)
        df_index_to_word = pd.read_csv('i2w.csv', encoding='utf-8', index_col =0)

        self.word_to_index_dict = df_word_to_index.to_dict(orient='index')
        self.index_to_word_dict = df_index_to_word.to_dict(orient='index')

        print({k: self.word_to_index_dict[k] for k in list(self.word_to_index_dict)[:5]})
        print({k: self.index_to_word_dict[k] for k in list(self.index_to_word_dict)[:5]})


    def model_load(self,forcing = True, bi = True):
        if forcing and not bi:
            self.tf_models(epoch=3)
        elif not forcing and not bi:
            self.non_tf_models(epoch=9)
        elif bi :
            self.bi_models(epoch=9)

    def bi_models(self,epoch=9):

        self.autoencoder_model_loaded = load_model('./save/models/full_model_reg_bi_new_epoch_9.h5')
        print(self.autoencoder_model_loaded.summary())

        for l in self.autoencoder_model_loaded.layers:
            print(l)

        # encoder_model = Model(encoder_inputs, encoder_states)
        # encoder_model = load_model('./save/models/enc_model_td_r_new_epoch_9.h5')
        # print(encoder_model.summary)

        #### Encoder :

        enc_inputs = self.autoencoder_model_loaded.get_layer('enc_input').output
        # enc_inputs = Input(shape=(None,), name='enc_input')
        # enc_emb = autoencoder_model_loaded.get_layer('shared_emb').get_output_at(0)
        enc_lstm = self.autoencoder_model_loaded.get_layer('bidirectional_1').output

        self.encoder_model_loaded = Model(enc_inputs, enc_lstm)
        print(self.encoder_model_loaded.summary())

        #### Decoder :

        # latent_dim = 128

        decoder_state_input_h1 = Input(shape=(embedding_size,))
        decoder_state_input_c1 = Input(shape=(embedding_size,))
        decoder_state_input_h2 = Input(shape=(embedding_size,))
        decoder_state_input_c2 = Input(shape=(embedding_size,))
        decoder_input_states = [decoder_state_input_h1, decoder_state_input_c1, decoder_state_input_h2, decoder_state_input_c2]

        dec_inp = self.autoencoder_model_loaded.get_layer('dec_input').output
        dec_emb = self.autoencoder_model_loaded.get_layer('shared_emb')(dec_inp)
        dec_lstm_layer = self.autoencoder_model_loaded.get_layer('bidirectional_2')

        dec_lstm = dec_lstm_layer(dec_emb, initial_state=decoder_input_states)

        decoder_output_states = dec_lstm[1:]

        dec_output = self.autoencoder_model_loaded.get_layer('dec_op_td')(dec_lstm[0])

        self.decoder_model_loaded = Model([dec_inp] + decoder_input_states , [dec_output] + decoder_output_states)

        print(self.decoder_model_loaded.summary())


    def tf_models(self,epoch=3):

        self.autoencoder_model_loaded = load_model('./save/models/full_model_td_r_new_epoch_99.h5')
        print(self.autoencoder_model_loaded.summary())

        for l in self.autoencoder_model_loaded.layers:
            print(l)

        # encoder_model = Model(encoder_inputs, encoder_states)
        # encoder_model = load_model('./save/models/enc_model_td_r_new_epoch_9.h5')
        # print(encoder_model.summary)

        #### Encoder :

        enc_inputs = self.autoencoder_model_loaded.get_layer('enc_input').output
        # enc_inputs = Input(shape=(None,), name='enc_input')
        # enc_emb = autoencoder_model_loaded.get_layer('shared_emb').get_output_at(0)
        enc_lstm = self.autoencoder_model_loaded.get_layer('enc_lstm').output

        self.encoder_model_loaded = Model(enc_inputs, enc_lstm)
        print(self.encoder_model_loaded.summary())

        #### Decoder :

        latent_dim = 32

        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_input_states = [decoder_state_input_h, decoder_state_input_c]

        dec_inp = self.autoencoder_model_loaded.get_layer('dec_input').output
        dec_emb = self.autoencoder_model_loaded.get_layer('shared_emb')(dec_inp)
        dec_lstm_layer = self.autoencoder_model_loaded.get_layer('dec_lstm')

        dec_lstm,  decoder_state_output_h, decoder_state_output_c = dec_lstm_layer(dec_emb, initial_state=decoder_input_states)

        decoder_output_states = [decoder_state_output_h, decoder_state_output_c]

        dec_output = self.autoencoder_model_loaded.get_layer('dec_op_td')(dec_lstm)

        self.decoder_model_loaded = Model([dec_inp] + decoder_input_states , [dec_output] + decoder_output_states)

        print(self.decoder_model_loaded.summary())

        # return autoencoder_model_loaded, encoder_model_loaded, decoder_model_loaded

    def non_tf_models(self,epoch=3):

        self.autoencoder_model_loaded = load_model('./save/models/full_model_td_r_new_epoch_9.h5')
        print(self.autoencoder_model_loaded.summary())

        for l in self.autoencoder_model_loaded.layers:
            print(l)

        # encoder_model = Model(encoder_inputs, encoder_states)
        # encoder_model = load_model('./save/models/enc_model_td_r_new_epoch_9.h5')
        # print(encoder_model.summary)

        #### Encoder :

        enc_inputs = self.autoencoder_model_loaded.get_layer('enc_input').output
        # enc_inputs = Input(shape=(None,), name='enc_input')
        # enc_emb = autoencoder_model_loaded.get_layer('shared_emb').get_output_at(0)
        enc_lstm = self.autoencoder_model_loaded.get_layer('enc_lstm').output

        self.encoder_model_loaded = Model(enc_inputs, enc_lstm)
        print(self.encoder_model_loaded.summary())

        #### Decoder :

        latent_dim = 32

        decoder_input_states = [decoder_state_input_h, decoder_state_input_c]

        dec_lstm_layer = autoencoder_model_loaded.get_layer('dec_lstm')

        dec_lstm,  decoder_state_output_h, decoder_state_output_c = dec_lstm_layer(dec_emb, initial_state=decoder_input_states)

        decoder_output_states = [decoder_state_output_h, decoder_state_output_c]

        dec_output = self.autoencoder_model_loaded.get_layer('dec_op_td')(dec_lstm)

        self.decoder_model_loaded = Model([dec_inp] + decoder_input_states , [dec_output] + decoder_output_states)

        print(self.decoder_model_loaded.summary())
        

    def decode_sequence_td(self,input_seq, bi= True):

        unknown_token_idx = self.word_to_index_dict.get(UNKNOWN_TOKEN).get('0')
        pad_token_idx = self.word_to_index_dict.get(PAD_TOKEN).get('0')
        eos_token_idx = self.word_to_index_dict.get(EOS_TOKEN).get('0')

        print(input_seq.shape)
        # Encode the input as state vectors.
        # if bi:
        op= self.encoder_model_loaded.predict(input_seq)
        states_value = op[1:]
        # else:
            # op , s1,s2= self.encoder_model_loaded.predict(input_seq)
            # states_value = [s1,s2]
        
        print("op shapwe : ",np.array(op[0]).shape)
    #     print("enc op : ", index_to_word_dict.get(np.argmax(op[0,-1,:])).get('0'))
        # print(np.array(op[1]).shape, np.array(op[2]).shape)
    #     print(states_value1)
        

        # Generate empty target sequence of length 1.
        target_seq = np.ones((1, max_sent_len))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0] = self.word_to_index_dict.get("EOS").get('0')

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
    #     while not stop_condition :
        output = self.decoder_model_loaded.predict(
            [target_seq] + states_value)

        output_tokens = output[0]
        print("output_tokens shape: ", output_tokens.shape)
        print("output tokens  : ",output_tokens)
        
        # Sample a token
        for i in range(20):
            sampled_token_index = int(output_tokens[0, i, 0])
            print("predicted word : ", self.index_to_word_dict.get(sampled_token_index,"none").get('0'))
    #     sampled_word = index_to_word_dict[sampled_token_index]
            decoded_sentence += ' ' + self.index_to_word_dict.get(sampled_token_index,"none").get('0')

        # Update the target sequence (of length 1).
    #     target_seq = np.zeros((1, 1, num_decoder_tokens))
    #     target_seq[0, 0, sampled_token_index] = 1.

        # Update states
    #     states_value = [h, c]

        return decoded_sentence


    def decode_sequence(self,input_seq):
    # Encode the input as state vectors.
        op, states_value0, states_value1 = self.encoder_model_loaded.predict(input_seq)
        print(np.array(states_value0).shape, np.array(states_value1).shape)
    #     states_value0 = np.reshape(np.array(states_value0), (20, -1))
    #     states_value1 = np.reshape(np.array(states_value1), (20, -1))
    #     print(states_value0.shape, states_value1.shape)
        
    #     states_value0 = states_value0.tolist()
    #     states_value1 = states_value1.tolist()
    #     print(states_value1)
        states_value = [states_value0, states_value1]
    #     states_value = [states_value0, states_value1]
    #     states_value = [np.argmax(states_value0), np.argmax(states_value1)]

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, max_sent_len))
    #     target_seq = np.zeros(( max_sent_len, 1))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0] = self.word_to_index_dict["PAD"].get('0')
        print(target_seq.shape)
        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        i=1

        while not stop_condition :
            output_tokens, h, c =self. decoder_model_loaded.predict(
                [target_seq] + states_value)
            print("output_tokens shape: ", output_tokens.shape)
            print("output tokens  : ",output_tokens)

            # Sample a token
            sampled_token_index = output_tokens[0, -1, 0]
            print("max index : ", sampled_token_index)
            
            print("predicted word : ", self.index_to_word_dict.get(int(sampled_token_index)).get('0'))
            sampled_word = self.index_to_word_dict.get(int(sampled_token_index)).get('0')
            decoded_sentence += ' ' + sampled_word
            
            if i == max_sent_len-1 or sampled_token_index == unknown_token_idx or sampled_token_index == eos_token_idx:
                stop_condition = True
                print("stpping: ", len(decoded_sentence))

            target_seq[0, i] = sampled_token_index

            # Update states
            states_value = [h, c]
            print("dec state shapes: ", np.array(h).shape, np.array(c).shape)
            
            i+=1
            
        return decoded_sentence


    def infer(self):

        test_str = "Did you change your hair?" #
        # test_str = "Hey there?"
        # text_eval ="Did you change your hair?"
        eval_data = np.array([self.word_to_index_dict.get(str(w), self.word_to_index_dict.get(UNKNOWN_TOKEN).get('0')).get('0') 
                    for w in nlp(test_str)])
        print([str(w) for w in nlp(test_str)])
        print(eval_data.shape)
        print(eval_data)
        eval_data.resize(max_sent_len)
        eval_data = np.reshape(eval_data, (-1, max_sent_len))
        # eval_data = np.reshape(eval_data, (max_sent_len, -1))
        print(eval_data)
        print("eval data shape", eval_data.shape)
        op = self.decode_sequence_td(eval_data)

        print("output is : ", op)


def main():
    inf= inference()
    # word_to_index_dict, index_to_word_dict = load_data()
    inf.model_load(forcing=True, bi=True)
    inf.infer()

if __name__ == '__main__':
    main()