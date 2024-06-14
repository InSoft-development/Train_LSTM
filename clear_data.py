import sqlite3
import json
import pandas as pd
import argparse
import os

from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from loguru import logger

from utils.data import load_config

# Поиск "верхней" директории
current_dir = os.path.dirname(__file__)
# Путь к директории на уровень выше
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# Чтение инициализирующих параметров
parser = argparse.ArgumentParser()
parser.add_argument('--station', type=str, default='')
parser.add_argument('--dir', type=str, default='')
opt = parser.parse_args()
config = load_config(f'{opt.dir}/config/{opt.station}.yml')
LAG = config['LAG']
# REPORTS_DIR = f'/home/art/InControl/Reports/{DIR_EXP}/'
DIR_EXP = config['DIR_EXP'] + str(LAG)
REPORTS_DIR = f'{parent_dir}/Reports/{DIR_EXP}/'

POWER_ID = config['POWER_ID']
POWER_LIMIT = config['POWER_LIMIT']
SQL = False
CSV = True
INTERVAL_CLEAR_LEN = config['INTERVAL_CLEAR_LEN']
INTERVAL_ANOMALY_LEN = config['INTERVAL_ANOMALY_LEN']
DISCOUNT_BETWEEN_IDX = config['DISCOUNT_BETWEEN_IDX']
CLEAR_DIR = f'{parent_dir}/Reports/{DIR_EXP}/clear_data/'
JSON_DIR = f'{parent_dir}/Reports/{DIR_EXP}/clear_data/json_interval/'
INTERVAL_REQ = config['INTERVAL_REQ']
TRAIN_FILE = config['TRAIN_FILE']

CHECK_POWER_LIMIT = False

# Функция скалирования данных для подачи в алгоритм очистки данных
def get_scaled(data_features):
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(
        data=scaler.fit_transform(data_features),
        columns=data_features.columns
    )
    return (scaled_data)

# Чтение и загрузка данных
def read_data(train_data):
    if CSV:
        df = pd.read_csv(train_data)
        df = df.dropna()
        # df = df.iloc[:10000]
        print(df)
        time = df['timestamp']
        df = df.drop(columns=['timestamp'])       
    if CHECK_POWER_LIMIT:
        df = df[df[POWER_ID] > POWER_LIMIT]
    return df, time

# Поиск интервалов "чистых" данных
def get_interval(data, len_interval, save_path):
    interval_len = 0
    start_index = 0
    end_index = -1
    count = 0
    dict_list = []
    for idx in data['check_index']:
        if end_index + 1 == idx:
            count += 1
            if count > len_interval:
                dictionary = {
                    "time": (data['timestamp'][start_index], data['timestamp'][end_index]),
                    "len": end_index - start_index,
                    "index": (start_index, end_index)}
                start_index = idx
                dict_list.append(dictionary)
                count = 0


        else:
            if idx - end_index > DISCOUNT_BETWEEN_IDX:
                count = 0
                start_index = idx
            else:
                count += 1

        end_index = idx
    json_object = json.dumps(dict_list, indent=4)
    with open(save_path, "w") as outfile:
        outfile.write(json_object)
    outfile.close()

# Функция формирования csv для найденных интервалов
def create_csv(data, dir, save_path):
    sum_df = pd.DataFrame()
    for json_file in sorted(os.listdir(dir)):
        try:
            f = open(dir + json_file, 'r')
            print(json_file)
            j = json.load(f)
            for interval in j:
                idx = interval['index']
                df = data.iloc[idx[0]:idx[1]]
                sum_df = pd.concat([sum_df, df], ignore_index=True)
        except Exception as e:
            logger.error(e)
        if json_file == 'with.json':
            sum_df.to_csv(f'{save_path}/anomaly.csv', index=False)
        if json_file == 'without.json':
            sum_df.to_csv(f'{save_path}/clear_data.csv', index=False)
        print(sum_df)

# создание директорий необходимых для сохранения выходов скрипта
if __name__ == '__main__':
    try:
        os.mkdir(REPORTS_DIR)
    except Exception as e:
        logger.error(e)
    try:
        os.mkdir(CLEAR_DIR)
    except Exception as e:
        logger.error(e)
    try:
        os.mkdir(JSON_DIR)
    except Exception as e:
        logger.error(e)
# Очистка данных
    data, time = read_data(TRAIN_FILE)
    scaled_data = get_scaled(data)
    one_class_svm = OneClassSVM(nu=0.02, gamma='auto', verbose=True)
    one_class_svm.fit(scaled_data)
    predict = one_class_svm.predict(scaled_data)
    data['timestamp'] = time
    data.to_csv(f'{CLEAR_DIR}/all_data.csv', index=False)
    data['one_svm_value'] = predict
    data['check_index'] = data.index 
    data_without_anomaly = data[data['one_svm_value'] == 1]
    data_with_anomaly = data[data['one_svm_value'] == -1]
    get_interval(data_without_anomaly, INTERVAL_CLEAR_LEN, f'{JSON_DIR}/without.json')
    get_interval(data_with_anomaly, INTERVAL_ANOMALY_LEN, f'{JSON_DIR}/with.json')
    create_csv(data, JSON_DIR, CLEAR_DIR)
