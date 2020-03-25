# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import os
import math
import time
import gensim
import logging
import json
import numpy as np

from collections import OrderedDict
from scipy import stats
from texttable import Texttable
from gensim.models import KeyedVectors
from tflearn.data_utils import pad_sequences


def _option(pattern):
    """
    Get the option according to the pattern.
    (pattern 0: Choose training or restore; pattern 1: Choose best or latest checkpoint.)

    Args:
        pattern: 0 for training step. 1 for testing step.
    Returns:
        The OPTION
    Raises:
        IOError: If the pattern is invalid
    """
    if pattern == 0:
        OPTION = input("[Input] Train or Restore? (T/R): ")
        while not (OPTION.upper() in ['T', 'R']):
            OPTION = input("[Warning] The format of your input is illegal, please re-input: ")
    if pattern == 1:
        OPTION = input("Load Best or Latest Model? (B/L): ")
        while not (OPTION.isalpha() and OPTION.upper() in ['B', 'L']):
            OPTION = input("[Warning] The format of your input is illegal, please re-input: ")
    else:
        raise IOError("[Error] The pattern input is invalid.")
    return OPTION.upper()


def logger_fn(name, input_file, level=logging.INFO):
    """
    The Logger.

    Args:
        name: The name of the logger
        input_file: The logger file path
        level: The logger level
    Returns:
        The logger
    """
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


def tab_printer(args, logger):
    """
    Function to print the logs in a nice tabular format.

    Args:
        args: Parameters used for the model.
        logger: The logger
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    t.add_rows([["Parameter", "Value"]])
    logger.info('\n' + t.draw())


def get_out_dir(option, logger):
    """
    Get the out dir.

    Args:
        option: Train or Restore
        logger: The logger
    Returns:
        The output dir
    Raises:
        IOError: If the option file is invalid
    """
    if option == 'T':
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        logger.info("Writing to {0}\n".format(out_dir))
    if option == 'R':
        MODEL = input("[Input] Please input the checkpoints model you want to restore, "
                      "it should be like (1490175368): ")  # The model you want to restore

        while not (MODEL.isdigit() and len(MODEL) == 10):
            MODEL = input("[Warning] The format of your input is illegal, please re-input: ")
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", MODEL))
        logger.info("Writing to {0}\n".format(out_dir))
    else:
        raise IOError("[Error] The option input is invalid.")
    return out_dir


def get_model_name():
    """
    Get the model name used for test.

    Returns:
        The model name
    """
    MODEL = input("[Input] Please input the model file you want to test, it should be like (1490175368): ")

    while not (MODEL.isdigit() and len(MODEL) == 10):
        MODEL = input("[Warning] The format of your input is illegal, "
                      "it should be like (1490175368), please re-input: ")
    return MODEL


def create_prediction_file(output_file, all_id, all_labels, all_predict_scores):
    """
    Create the prediction file.

    Args:
        output_file: The all classes predicted results provided by network
        all_id: The data record id
        all_labels: The true labels
        all_predict_scores: The predict scores
    Raises:
        IOError: If the prediction file is not a .json file
    """
    if not output_file.endswith('.json'):
        raise IOError("[Error] The prediction file is not a json file."
                      "Please make sure the prediction data is a json file.")
    with open(output_file, 'w') as fout:
        data_size = len(all_predict_scores)
        for i in range(data_size):
            labels = [float(i) for i in all_labels[i]]
            predict_scores = [round(float(i), 4) for i in all_predict_scores[i]]
            data_record = OrderedDict([
                ('id', all_id[i]),
                ('labels', labels),
                ('predict_scores', predict_scores)
            ])
            fout.write(json.dumps(data_record, ensure_ascii=False) + '\n')


def evaluation(true_label, pred_label):
    """
    Calculate the PCC & DOA.

    Args:
        true_label: The true labels
        pred_label: The predicted labels
    Returns:
        The value of PCC & DOA
    """
    test_y = []
    pred_y = []
    for i in true_label:
        for value in i:
            test_y.append(value)
    for j in pred_label:
        for value in j:
            pred_y.append(value)

    # compute pcc
    pcc, _ = stats.pearsonr(pred_y, test_y)
    if math.isnan(pcc):
        print('[Error]: PCC=nan', test_y, pred_y)
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


def create_metadata_file(word2vec_file, output_file):
    """
    Create the metadata file based on the corpus file (Used for the Embedding Visualization later).

    Args:
        word2vec_file: The word2vec file
        output_file: The metadata file path
    Raises:
        IOError: If word2vec model file doesn't exist
    """
    if not os.path.isfile(word2vec_file):
        raise IOError("[Error] The word2vec file doesn't exist.")

    model = KeyedVectors.load_word2vec_format(open(word2vec_file, 'r'), binary=False, unicode_errors='replace')
    word2idx = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    word2idx_sorted = [(k, word2idx[k]) for k in sorted(word2idx, key=word2idx.get, reverse=False)]

    with open(output_file, 'w+') as fout:
        for word in word2idx_sorted:
            if word[0] is None:
                print("[Warning] Empty Line, should replaced by any thing else, or will cause a bug of tensorboard")
                fout.write('<Empty Line>' + '\n')
            else:
                fout.write(word[0] + '\n')


def load_word2vec_matrix(embedding_size, word2vec_file):
    """
    Return the word2vec model matrix.

    Args:
        embedding_size: The embedding size
        word2vec_file: The word2vec file
    Returns:
        The word2vec model matrix
    Raises:
        IOError: If word2vec model file doesn't exist
    """
    if not os.path.isfile(word2vec_file):
        raise IOError("[Error] The word2vec file doesn't exist. ")

    model = KeyedVectors.load_word2vec_format(open(word2vec_file, 'r'), binary=False, unicode_errors='replace')
    vocab_size = len(model.wv.vocab.items())
    vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    vector = np.zeros([vocab_size, embedding_size])
    for key, value in vocab.items():
        if key is not None:
            vector[value] = model[key]
    return vocab_size, vector


def data_word2vec(input_file, word2vec_model):
    """
    Create the research data tokenindex based on the word2vec model file.
    Return the class Data (includes the data tokenindex and data labels).

    Args:
        input_file: The research data
        word2vec_model: The word2vec model file
    Returns:
        The Class _Data() (includes the data tokenindex and data labels)
    Raises:
        IOError: If the input file is not the .json file
    """
    vocab = dict([(k, v.index) for (k, v) in word2vec_model.wv.vocab.items()])

    def _token_to_index(content):
        result = []
        for item in content:
            word2id = vocab.get(item)
            if word2id is None:
                word2id = 0
            result.append(word2id)
        return result

    if not input_file.endswith('.json'):
        raise IOError("[Error] The research data is not a json file. "
                      "Please preprocess the research data into the json file.")
    with open(input_file) as fin:
        id_list = []
        content_index_list = []
        question_index_list = []
        option_index_list = []
        labels_list = []
        total_line = 0

        for eachline in fin:
            data = json.loads(eachline)
            id = data['id']
            content_text = data['content']
            question_text = data['question']
            option_text = data['pos_text']
            labels = data['diff']

            id_list.append(id)
            content_index_list.append(_token_to_index(content_text))
            question_index_list.append(_token_to_index(question_text))
            option_index_list.append(_token_to_index(option_text))
            labels_list.append(labels)
            total_line += 1

    class _Data:
        def __init__(self):
            pass

        @property
        def number(self):
            return total_line

        @property
        def id(self):
            return id_list

        @property
        def content_index(self):
            return content_index_list

        @property
        def question_index(self):
            return question_index_list

        @property
        def option_index(self):
            return option_index_list

        @property
        def labels(self):
            return labels_list

    return _Data()


def data_augmented(data, drop_rate=1.0):
    """
    Data augment.

    Args:
        data: The Class _Data()
        drop_rate: The drop rate
    Returns:
        The Class _AugData()
    """
    aug_num = data.number
    aug_id = data.id
    aug_content_index = data.content_index
    aug_question_index = data.question_index
    aug_option_index = data.option_index
    aug_labels = data.labels

    for i in range(len(data.content_index)):
        data_record = data.content_index[i]
        if len(data_record) == 1:  # 句子长度为 1，则不进行增广
            continue
        elif len(data_record) == 2:  # 句子长度为 2，则交换两个词的顺序
            data_record[0], data_record[1] = data_record[1], data_record[0]
            aug_id.append(data.id[i])
            aug_content_index.append(data_record)
            aug_question_index.append(data.question_index[i])
            aug_option_index.append(data.option_index[i])
            aug_labels.append(data.labels[i])
            aug_num += 1
        else:
            data_record = np.array(data_record)
            for num in range(len(data_record) // 10):  # 打乱词的次数，次数即生成样本的个数；次数根据句子长度而定
                # random shuffle & random drop
                data_shuffled = np.random.permutation(np.arange(int(len(data_record) * drop_rate)))
                new_data_record = data_record[data_shuffled]

                aug_id.append(data.id[i])
                aug_content_index.append(list(new_data_record))
                aug_question_index.append(data.question_index[i])
                aug_option_index.append(data.option_index[i])
                aug_labels.append(data.labels[i])
                aug_num += 1

    class _AugData:
        def __init__(self):
            pass

        @property
        def number(self):
            return aug_num

        @property
        def id(self):
            return aug_id

        @property
        def content_index(self):
            return aug_content_index

        @property
        def question_index(self):
            return aug_question_index

        @property
        def option_index(self):
            return aug_option_index

        @property
        def labels(self):
            return aug_labels

    return _AugData()


def load_data_and_labels(data_file, word2vec_file, data_aug_flag):
    """
    Load research data from files, splits the data into words and generates labels.
    Return split sentences, labels and the max sentence length of the research data.

    Args:
        data_file: The research data
        word2vec_file: The word2vec file
        data_aug_flag: The flag of data augmented
    Returns:
        The class Data
    Raises:
        IOError: If word2vec model file doesn't exist
    """
    # Load word2vec file
    if not os.path.isfile(word2vec_file):
        raise IOError("[Error] The word2vec file doesn't exist. ")

    model = KeyedVectors.load_word2vec_format(open(word2vec_file, 'r'), binary=False, unicode_errors='replace')

    # Load data from files and split by words
    data = data_word2vec(input_file=data_file, word2vec_model=model)
    if data_aug_flag:
        data = data_augmented(data)

    # plot_seq_len(data_file, data)

    return data


def pad_data(data, pad_seq_len):
    """
    Padding each sentence of research data according to the max sentence length.
    Return the padded data and data labels.

    Args:
        data: The research data
        pad_seq_len: The max sentence length of [content, question, option]
    Returns:
        pad_content: The padded data
        pad_question: The padded data
        pad_option: The padded data
        labels: The data labels
    """
    pad_content = pad_sequences(data.content_index, maxlen=pad_seq_len[0], value=0.)
    pad_question = pad_sequences(data.question_index, maxlen=pad_seq_len[1], value=0.)
    pad_option = pad_sequences(data.option_index, maxlen=pad_seq_len[2], value=0.)
    labels = [[float(label)] for label in data.labels]
    return pad_content, pad_question, pad_option, labels


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    含有 yield 说明不是一个普通函数，是一个 Generator.
    函数效果：对 data，一共分成 num_epochs 个阶段（epoch），在每个 epoch 内，如果 shuffle=True，就将 data 重新洗牌，
    批量生成 (yield) 一批一批的重洗过的 data，每批大小是 batch_size，一共生成 int(len(data)/batch_size)+1 批。

    Args:
        data: The data
        batch_size: The size of the data batch
        num_epochs: The number of epochs
        shuffle: Shuffle or not (default: True)
    Returns:
        A batch iterator for data set
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
