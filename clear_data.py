import sqlite3
import json
import pandas as pd
import argparse
import os

from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from loguru import logger

from utils.data import load_config

parser = argparse.ArgumentParser()
parser.add_argument('--station', type=str, default='')
opt = parser.parse_args()
config = load_config(f'config_{opt.station}.yml')
LAG = config['LAG']
# REPORTS_DIR = f'/home/art/InControl/Reports/{DIR_EXP}/'
DIR_EXP = config['DIR_EXP'] + str(LAG)
REPORTS_DIR = f'/home/art/InControl/Reports/{DIR_EXP}/'
TRAIN_FILE_SQL = config['TRAIN_FILE_SQL']
POWER_ID = config['POWER_ID']
POWER_LIMIT = config['POWER_LIMIT']
SQL = False
CSV = True
INTERVAL_CLEAR_LEN = config['INTERVAL_CLEAR_LEN']
INTERVAL_ANOMALY_LEN = config['INTERVAL_ANOMALY_LEN']
DISCOUNT_BETWEEN_IDX = config['DISCOUNT_BETWEEN_IDX']
CLEAR_DIR = f'/home/art/InControl/Reports/{DIR_EXP}/clear_data/'
JSON_DIR = f'/home/art/InControl/Reports/{DIR_EXP}/clear_data/json_interval/'
INTERVAL_REQ = config['INTERVAL_REQ']

CHECK_POWER_LIMIT = False


def get_scaled(data_features):
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(
        data=scaler.fit_transform(data_features),
        columns=data_features.columns
    )
    return (scaled_data)


def read_data(train_data):
    if CSV:
        df = pd.read_csv(train_data)
        df = df.dropna()
        # df = df.iloc[:10000]
        print(df)
        time = df['timestamp']
        df = df.drop(columns=['timestamp'])
        
    if SQL:
        cnx = sqlite3.connect(train_data)
        df = pd.read_sql_query("SELECT * FROM 'data'", cnx)
        # df.to_csv('/home/art/InControl/data/SOCHI/slices_GT_norm.csv')
        time = df['Measurement Units']
        df = df.drop(columns=['Measurement Units'])
    if CHECK_POWER_LIMIT:
        df = df[df[POWER_ID] > POWER_LIMIT]
    return df, time



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


if __name__ == '__main__':
    try:
        os.mkdir(REPORTS_DIR)
    except:
        print('Directory exit')
    try:
        os.mkdir(CLEAR_DIR)
    except:
        print('Directory exit')
    try:
        os.mkdir(JSON_DIR)

    except:
        print('Directory exit')
    data, time = read_data(TRAIN_FILE_SQL)
   
 
    scaled_data = get_scaled(data)
    one_class_svm = OneClassSVM(nu=0.02, gamma='auto', verbose=True)
    one_class_svm.fit(scaled_data)
    predict = one_class_svm.predict(scaled_data)
    print(predict)
    data['timestamp'] = time
    data.to_csv(f'{CLEAR_DIR}/all_data.csv', index=False)
    data['one_svm_value'] = predict
    data['check_index'] = data.index 
    print(data) 
    # print(data)
    # for interval in INTERVAL_REQ:
    #         data.at[range(interval[0],interval[1]), 'one_svm_value'] = 1
    data_without_anomaly = data[data['one_svm_value'] == 1]
    data_with_anomaly = data[data['one_svm_value'] == -1]
    get_interval(data_without_anomaly, INTERVAL_CLEAR_LEN, f'{JSON_DIR}/without.json')
    get_interval(data_with_anomaly, INTERVAL_ANOMALY_LEN, f'{JSON_DIR}/with.json')
    create_csv(data, JSON_DIR, CLEAR_DIR)
