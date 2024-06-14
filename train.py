# import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import os
import numpy as np
import pandas as pd
import argparse
import shutil
import tensorflow as tf
from keras.optimizers import adam

from utils.data import get_scaled, load_config, save_scaler, kalman_filter
from utils.smooth import exponential_smoothing, double_exponential_smoothing
from model import autoencoder_model


# функция для среза выборки под размер лага обучения
def get_len_size(LAG, x_size):
    return int(x_size / LAG) * LAG


parser = argparse.ArgumentParser()
parser.add_argument('--station', type=str, default='')
opt = parser.parse_args()
config = load_config(f'config_{opt.station}.yml')


MEAN_NAN = config['MEAN_NAN']
DROP_NAN = config['DROP_NAN']

ROLLING_MEAN = config['ROLLING_MEAN']
EXP_SMOOTH = config['EXP_SMOOTH']
DOUBLE_EXP_SMOOTH = config['DOUBLE_EXP_SMOOTH']

KKS = config['KKS']
NUM_GROUPS = config['NUM_GROUPS']
LAG = config['LAG']
DIR_EXP = config['DIR_EXP'] + str(LAG)
REPORTS_DIR = f'/home/art/InControl/Reports/{DIR_EXP}/'

EPOCHS = config['EPOCHS']
BATCH_SIZE = config['BATCH_SIZE']
POWER_ID = config['POWER_ID']
POWER_LIMIT = config['POWER_LIMIT']
ROLLING_MEAN_WINDOW = config['ROLLING_MEAN_WINDOW']
CSV = True
SQL = False
TRAIN_FILE = f'{REPORTS_DIR}/clear_data/all_data.csv'
# TRAIN_FILE = '/home/art/Downloads/slices(3).csv'


if CSV:
    df = pd.read_csv(TRAIN_FILE)
    # df = df.drop(columns=['timestamp','one_svm_value', 'check_index'])
    df = df.drop(columns='timestamp')

if SQL:
    cnx = sqlite3.connect(TRAIN_FILE)
    # print(cnx)
    df = pd.read_sql_query("SELECT * FROM 'data_train'", cnx)
    df = df.drop(columns=['timestamp', 'index'])
# df = pd.read_sql_query("SELECT * FROM 'data'", cnx)  
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
# df = df[df[POWER_ID] > POWER_LIMIT]
# print(df.columns)
if MEAN_NAN:
    df = df.fillna(df.mean(), inplace=True)
if DROP_NAN:
    df = df.dropna()

KALMAN = False
if KALMAN:
    df = kalman_filter(df)

if ROLLING_MEAN:
    rolling_mean = df.rolling(window=ROLLING_MEAN_WINDOW).mean()
if EXP_SMOOTH:
    for i in df.columns:
        df[str(i)] = exponential_smoothing(df[str(i)].to_numpy(), alpha=0.2)

if DOUBLE_EXP_SMOOTH:
    for i in df.columns:
        df[str(i)] = double_exponential_smoothing(df[str(i)].to_numpy(), alpha=0.02, beta=0.09)

groups = pd.read_csv(KKS, sep=';')
# groups = groups.astype({'group': 'int32'}).dtypes
print(groups['group'])

group_list = []

sum = 0
for i in range(0, NUM_GROUPS):
    group = groups[groups['group'] == i]
    
    if i != 0:
        group = group.append(groups[groups['group'] == 0])
        print(group)
    sum += len(group)
    if len(group) == 0:
        continue
    # print(sum)
    
    group = df[group['kks']]
    #   print(group)
    group_list.append(get_scaled(group))
    save_scaler(group,f'/home/art/InControl/Reports/{DIR_EXP}/scaler_{i}.pkl')
    
    

#   dir_exp = 'lstm_group_predict'

try:
    os.mkdir(f'/home/art/InControl/Reports/{DIR_EXP}')
except:
    print('Directory exit')

try:
    os.mkdir(f'/home/art/InControl/Reports/{DIR_EXP}/train_info/')
except:
    print('Directory exit')

shutil.copy(f'config_{opt.station}.yml', f'/home/art/InControl/Reports/{DIR_EXP}/train_info/')

try:
    os.mkdir(f'/home/art/InControl/Reports/{DIR_EXP}/train_info/model/')
except:
    print('Directory exit')
print(group_list)   

# power =  df['20MBY10CE901_XQ01']

for i in range(0, len(group_list)):
    print(i)
    model_save = f'/home/art/InControl/Reports/{DIR_EXP}/lstm_group_{i}.h5'
    X_train = group_list[i].to_numpy()
    len_size = get_len_size(LAG, X_train.shape[0])
    X_train = X_train[:len_size].reshape(int(X_train.shape[0] / LAG), int(LAG), X_train.shape[1])
    print("Training data shape:", X_train.shape)
    model = autoencoder_model(X_train)
    #   opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer='adam', loss='mae')
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
    mcp_save = tf.keras.callbacks.ModelCheckpoint(model_save, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1,
                                                          min_delta=1e-4, mode='min')
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    #                                                 save_weights_only=True,
    #                                                 verbose=1)
    history = model.fit(X_train, X_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_split=0.1, callbacks=[earlyStopping, mcp_save, reduce_lr_loss]).history

    with open(f'/home/art/InControl/Reports/{DIR_EXP}/train_info/model/modelsummary_{i}.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
