from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector, BatchNormalization, Bidirectional, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model
from keras import regularizers
import tensorflow as tf
import numpy as np

# Simple LSTM autoencoder
def simple_autoencoder(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(128, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(64, activation='sigmoid', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(64, activation='sigmoid', return_sequences=True)(L3)
    L5 = LSTM(128, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model

# Bidirectional LSTM autoencoder
def bidirectional_lstm_autoencoder(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = Bidirectional(LSTM(512, activation='tanh', return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(inputs)
    N1 = BatchNormalization()(L1)
    D1 = Dropout(0.2)(N1)
    L2 = LSTM(128, activation='relu', return_sequences=False)(D1)
    L3 = RepeatVector(X.shape[1])(L2)
    D2 = Dropout(0.2)(L3)
    L4 = LSTM(128, activation='relu', return_sequences=True)(D2)
    N2 = BatchNormalization()(L4)
    L5 = Bidirectional(LSTM(512, activation='tanh', return_sequences=True))(N2)
    output = TimeDistributed(Dense(X.shape[2], activation='relu'))(L5)
    model = Model(inputs=inputs, outputs=output)
    return model

# Convolutional autoencoder
def conv_autoencoder(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    x = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    encoded = MaxPooling1D(pool_size=2, padding='same')(x)
    
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(encoded)
    x = UpSampling1D(2)(x)
    x = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(filters=X.shape[2], kernel_size=3, activation='sigmoid', padding='same')(x)
    
    model = Model(inputs=inputs, outputs=decoded)
    return model

# Transformer-based autoencoder
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / np.power(10000., (2 * (i // 2)) / np.float32(d_model))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(position=np.arange(position)[:, np.newaxis], i=np.arange(d_model)[np.newaxis, :], d_model=d_model)
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def transformer_autoencoder(X):
    num_heads = 4
    ff_dim = 64 
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    x = PositionalEncoding(X.shape[1], X.shape[2])(inputs)
    attention_out = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(x, x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_out + x)
    encoder_out = Dense(ff_dim, activation="relu")(x)
    attention_out = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(encoder_out, encoder_out)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_out + encoder_out)
    decoder_out = Dense(X.shape[2], activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=decoder_out)
    return model

def get_autoencoder_model(X, architecture):
    if architecture == 'simple_autoencoder':
        return simple_autoencoder(X)
    elif architecture == 'bidirectional_lstm':
        return bidirectional_lstm_autoencoder(X)
    elif architecture == 'conv_autoencoder':
        return conv_autoencoder(X)
    elif architecture == 'transformer_autoencoder':
        return transformer_autoencoder(X)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
