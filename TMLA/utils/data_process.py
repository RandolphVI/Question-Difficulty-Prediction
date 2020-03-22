# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import os
import json
import math
import logging
import pickle
import numpy as np
from tqdm import tqdm
from scipy import stats


def logger_fn(name, input_file, level=logging.INFO):
    tf_logger = logging.getLogger(name)
    tf_logger.setLevel(level)
    log_dir = os.path.dirname(input_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(input_file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    tf_logger.addHandler(fh)
    return tf_logger


def create_word_dict(input_file, pickle_file):
    with open(input_file, 'r') as fin, open(pickle_file, 'wb') as handle:
        word_dict = dict()
        word_num = 0
        for eachline in fin:
            line = json.loads(eachline)
            words = line['content'] + line['question'] + line['pos_text']
            for word in words:
                if word not in word_dict.keys():
                    word_dict[word] = word_num
                    word_num = word_num + 1
        # Save Word Dict
        pickle.dump(word_dict, handle)


def create_bow_feature(input_file, pickle_file, output_file):
    with open(input_file, 'r') as fin, open(pickle_file, 'rb') as handle, open(output_file, 'w') as fout:
        word_dict = pickle.load(handle)
        word_num = len(word_dict.keys())
        print(word_num)

        for eachline in tqdm(fin):
            line = json.loads(eachline)
            words = line['content'] + line['question'] + line['pos_text']
            feature = [0] * word_num
            for word in words:
                feature[word_dict[word]] += 1
            data_record = {
                'id': line['id'],
                'feature': feature,
                'diff': line['diff']
            }
            fout.write(json.dumps(data_record, ensure_ascii=False) + '\n')


def load_data(data_file):
    x_data, y_data = [], []
    with open(data_file, 'r') as f_train:
        for eachline in f_train:
            line = json.loads(eachline)
            x_data.append(list(map(float, line['feature'])))
            y_data.append(float(line['diff']))

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data


def evaluation(test_y, pred_y):
    # compute pcc
    pcc, _ = stats.pearsonr(pred_y, test_y)
    if math.isnan(pcc):
        print('ERROR: PCC=nan', test_y, pred_y)
    # compute doa
    n = 0
    correct_num = 0
    for i in range(len(test_y) - 1):
        for j in range(i + 1, len(test_y)):
            if (test_y[i] > test_y[j]) and (pred_y[i] > pred_y[j]):
                correct_num += 1
            elif (test_y[i] == test_y[j]) and (pred_y[i] == pred_y[j]):
                continue
            elif (test_y[i] < test_y[j]) and (pred_y[i] < pred_y[j]):
                correct_num += 1
            n += 1
    if n == 0:
        print(test_y)
        return -1, -1
    doa = correct_num / n
    return pcc, doa


if __name__ == '__main__':
    # create_word_dict('../../data/data.json', '../../data/word.pickle')
    # create_bow_feature('../../data/Train_sample.json', '../../data/word.pickle', '../../data/Train_BOW_sample.json')
    # create_bow_feature('../../data/Test_sample.json', '../../data/word.pickle', '../../data/Test_BOW_sample.json')
    pass
