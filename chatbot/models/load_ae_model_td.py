import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, RepeatVector, Bidirectional, Dropout, Embedding, TimeDistributed
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import SGD, RMSprop, Adam
from keras import objectives


vocabulary_size = 50003
max_sent_len = 20
embedding_size = 32
num_encoder_features = 1
# encoder_inputs = Input(shape=(None, ), name='enc_input')
# -------------
def load_model():
    
# To change
    encoder_inputs = Input(shape=(max_sent_len, ), name='enc_input')
    # -------------

    # shared_emb = Embedding(input_dim = vocabulary_size, output_dim = embedding_size, input_length=20, name='shared_emb')
    # -------------
    # To change
    shared_emb = Embedding(input_dim = vocabulary_size, output_dim = embedding_size, name='shared_emb', mask_zero=True) # , input_length = 20
    # -------------

    x = shared_emb(encoder_inputs)
    encoder = LSTM(embedding_size, 
                return_state=True,
                return_sequences=False,
                name='enc_lstm'
                )

    #model.add(BatchNormalization())

    encoder_outputs= encoder(x)
    encoder_states = [encoder_outputs[1], encoder_outputs[2]]

    # Decoder
    decoder_inputs = Input(shape=(max_sent_len, ), name='dec_input')
    x = shared_emb(decoder_inputs)
    decoder_lstm = LSTM(embedding_size, 
                return_sequences=True, 
                return_state = True,
                name='dec_lstm'
                )
    decoder_op, _, _ = decoder_lstm(x, initial_state=encoder_states)
    decoder_outputs = TimeDistributed(Dense(vocabulary_size, activation='softmax', name='dec_op_td'))(decoder_op)
    # decoder_outputs = Dense(vocabulary_size, activation='softmax', name='dec_output')(decoder_op)
    # decoder_outputs = Dense(vocabulary_size, activation='softmax', name='dec_outputs')(decoder_outputs)

    encoder = Model(encoder_inputs, encoder_outputs)
    encoder.summary()

    # decoder = Model(decoder_inputs, decoder_outputs)
    # decoder.summary()

    auto_encoder = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    auto_encoder.summary()

    return auto_encoder, encoder

def main():
    _,_ = load_model()

if __name__ == '__main__':
    main()