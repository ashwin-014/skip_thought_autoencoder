import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, RepeatVector, Bidirectional, Dropout, Embedding
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import SGD, RMSprop, Adam
from keras import objectives

# x = Input(batch_shape=(batch_size, original_dim))
intermediate_dim = 32
latent_dim = 32
timesteps =20
vocabulary_size = 500003
embedding_size = 64
max_sent_len = 20
batch_size = 32
epsilon_std = 1.0

encoder_inputs = Input(shape=(None, ), name='enc_input')

shared_emb = Embedding(vocabulary_size, embedding_size, input_length=20, name='shared_emb', mask_zero=True)
x = shared_emb(encoder_inputs)

h = LSTM(intermediate_dim, 
#                return_state=True,
            return_sequences=False,
            recurrent_dropout = 0.2,
#             merge_mode='concat',
            input_shape = ( max_sent_len, 1),
            name='enc_lstm'
            )(x)

h= Dropout(0.2)(h)
# h = LSTM(intermediate_dim)(x)

# VAE Z layer
z_mean = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_sigma) * epsilon


# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])


input_dim = 1
timesteps = 20
# decoder
# decoder_h = LSTM(intermediate_dim, return_state = False, return_sequences =True, input_shape = ( max_sent_len, 1),name='dec_lstm_h')

# changed : 
decoder_h = Dense(intermediate_dim, name='dec_dense_h') # , return_state = False, return_sequences =True,

decoder_mean = LSTM(input_dim, return_state = False, return_sequences= True,input_shape = ( max_sent_len, 1), name='dec_lstm_mean')
# decoder_mean = Dense(vocabulary_size, name='dec_output', activation='softmax')

decoded_h = RepeatVector(timesteps)(z)
# decoded_h = decoder_h(decoded_h)
decoded_mean = decoder_mean(decoded_h)

# changed :
# decoded_h = decoder_h(z)
# decoded_h = RepeatVector(timesteps)(decoded_h)
# print(decoded_h.shape)
# decoded_mean = decoder_mean(decoded_h)

# end-to-end autoencoder
vae = Model(encoder_inputs, decoded_mean)

# encoder, from inputs to latent space
encoder = Model(encoder_inputs, z_mean)

# generator, from latent space to reconstructed inputs
decoder_input = Input(shape=(latent_dim,))
_rp_decoded_h = RepeatVector(timesteps)(decoder_input)
_decoded_h = decoder_h(_rp_decoded_h)
_decoded_mean = decoder_mean(_decoded_h)

generator = Model(decoder_input, _decoded_mean)

def vae_loss(encoder_inputs, decoded_mean):
#     xent_loss = objectives.sparse_categorical_crossentropy(encoder_inputs, decoded_mean)
    xent_loss = objectives.mse(encoder_inputs, decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    return xent_loss + kl_loss

def load_model():
    opt = Adam(lr=0.01)
    vae.compile(optimizer=opt, loss=vae_loss)

    # for layer in vae.layers:
        # if isinstance(layer.input, list):
            # print(layer.name)
        # else:
            # print(layer.name, layer.input.shape, layer.output.shape)    

    print(vae.summary())
    print(encoder.summary())
    print(generator.summary())

    return vae, encoder, generator

def main():
    load_model()

if __name__ == '__main__':
    main()