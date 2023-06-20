from tensorflow.keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

import tensorflow as tf

def autoencoder_model(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(128, activation='relu', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(64, activation='sigmoid', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(64, activation='sigmoid', return_sequences=True)(L3)
    L5 = LSTM(128, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model

# def autoencoder_model(X):
#     inputs = Input(shape=(X.shape[1], X.shape[2]))
#     L1 = LSTM(256, activation='relu', return_sequences=True, 
#               kernel_regularizer=regularizers.l2(0.00))(inputs)
#     L2 = LSTM(128, activation='sigmoid', return_sequences=True)(L1)
#     L3 = LSTM(32, activation='sigmoid', return_sequences=False)(L2)
#     L4 = RepeatVector(X.shape[1])(L3)
#     L5 = LSTM(32, activation='sigmoid', return_sequences=True)(L4)
#     L6 = LSTM(128, activation='sigmoid', return_sequences=True)(L5)
#     L7 = LSTM(256, activation='relu', return_sequences=True)(L6)
#     output = TimeDistributed(Dense(X.shape[2]))(L7)    
#     model = Model(inputs=inputs, outputs=output)
#     return model