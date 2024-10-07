import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import argparse
import shutil
import json
import clickhouse_driver
import tensorflow as tf
from keras.optimizers import adam
from loguru import logger

from utils.data import get_scaled, load_config, save_scaler, kalman_filter
from utils.smooth import exponential_smoothing, double_exponential_smoothing
from model import get_autoencoder_model

# Поиск "верхней" директории
current_dir = os.path.dirname(__file__)
# Путь к директории на уровень выше
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

def create_windows(data, window_size, step_size):
    """
    Функция для разделения данных на окна.
    
    :param data: Массив данных для разделения.
    :param window_size: Размер каждого окна (количество временных шагов).
    :param step_size: Шаг смещения между окнами.
    :return: Массив данных, разделенных на окна.
    """
    sequences = []
    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        seq = data[start:end]
        sequences.append(seq)
    return np.array(sequences)
# функция для среза выборки под размер лага обучения
def get_len_size(LAG, x_size):
    return int(x_size / LAG) * LAG

# Чтение инициализирующих параметров
parser = argparse.ArgumentParser()
parser.add_argument('--station', type=str, default='')
parser.add_argument('--dir', type=str, default='')
opt = parser.parse_args()
config = load_config(f'{opt.dir}/config/{opt.station}.yml')
model_config = load_config(f'{opt.dir}/config/settings/model.yml')

MEAN_NAN = config['MEAN_NAN']
DROP_NAN = config['DROP_NAN']

ROLLING_MEAN = config['ROLLING_MEAN']
EXP_SMOOTH = config['EXP_SMOOTH']
DOUBLE_EXP_SMOOTH = config['DOUBLE_EXP_SMOOTH']

KKS = f"{parent_dir}/{config['KKS']}"
NUM_GROUPS = config['NUM_GROUPS']
LAG = config['LAG']
DIR_EXP = config['DIR_EXP']
# REPORTS_DIR = f'/home/art/InControl/Reports/{DIR_EXP}/'

EPOCHS = config['EPOCHS']
BATCH_SIZE = config['BATCH_SIZE']
POWER_ID = config['POWER_ID']
POWER_LIMIT = config['POWER_LIMIT']
ROLLING_MEAN_WINDOW = config['ROLLING_MEAN_WINDOW']
USE_ALL_DATA = config['USE_ALL_DATA']

# Чтение и загрузка данных
if USE_ALL_DATA:
    TRAIN_FILE = f"{parent_dir}/{config['TRAIN_FILE']}"
    df = pd.read_csv(TRAIN_FILE, sep=',')
    df = df.drop(columns='timestamp')
else:
    TRAIN_FILE = f'{parent_dir}/Reports/{DIR_EXP}/clear_data/clear_data.csv'
    df = pd.read_csv(TRAIN_FILE)
    df = df.drop(columns=['timestamp','one_svm_value', 'check_index'])
    
# Иницализация GPU
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Обработка данных перед подачей в сеть   
df = df[df[POWER_ID] > POWER_LIMIT]
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
logger.info(f"KKS: \n {groups}")
logger.info(f"Data: \n {df.head()}")

# Создание директроии для проведения экспериметов
try:
    os.mkdir(f'{parent_dir}/Reports/{DIR_EXP}')
except Exception as e:
    logger.error(e)
try:
    os.mkdir(f'{parent_dir}/Reports/{DIR_EXP}/train_info/')
except Exception as e:
    logger.error(e)
shutil.copy(f'{opt.dir}/config/{opt.station}.yml', f'{parent_dir}/Reports/{DIR_EXP}/train_info/')
try:
    os.mkdir(f'{parent_dir}/Reports/{DIR_EXP}/train_info/model/')
except Exception as e:
    logger.error(e)
try:
    os.mkdir(f'{parent_dir}/Reports/{DIR_EXP}/model_pt/')
except Exception as e:
    logger.error(e)
try:
    os.mkdir(f'{parent_dir}/Reports/{DIR_EXP}/scaler_data/')
except Exception as e:
    logger.error(e)

# разбиение данных по группам, скейлинг и сохранение скейлеров 
group_list = []
sum = 0
groups['group'] = groups['group'].astype(int)
logger.debug(f"KKS: \n {groups.dtypes}")
for i in range(0, NUM_GROUPS):
    group = groups[groups['group'] == i]
    logger.debug(group)
    if i != 0:
        group = group.append(groups[groups['group'] == 0])
    sum += len(group)
    if len(group) == 0:
        continue
    group = df[group['kks']]
    group_list.append(get_scaled(group))
    save_scaler(group,f'{parent_dir}/Reports/{DIR_EXP}/scaler_data/scaler_{i}.pkl')
      
# Цикл обучения, c сохранением весов и информации о модели
for i in range(0, len(group_list)):
    logger.info(f"Group {i}")
    model_save = f'{parent_dir}/Reports/{DIR_EXP}/model_pt/lstm_group_{i}.h5'
    X_train = group_list[i].to_numpy()
    # len_size = get_len_size(LAG, X_train.shape[0])
    X_train = create_windows(X_train,10,1)
    # X_train = X_train[:len_size].reshape(int(X_train.shape[0] / LAG), int(LAG), X_train.shape[1])
    print("Training data shape:", X_train.shape)
    model = get_autoencoder_model(X_train, architecture=model_config['architecture'])
    model.compile(optimizer='adam', loss='mae')
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
    mcp_save = tf.keras.callbacks.ModelCheckpoint(model_save, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1,
                                                          min_delta=1e-4, mode='min')
    history = model.fit(X_train, X_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_split=0.1, callbacks=[earlyStopping, mcp_save, reduce_lr_loss]).history

    with open(f'{parent_dir}/Reports/{DIR_EXP}/train_info/model/modelsummary_{i}.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
