# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import os
import gensim
import logging
import json
import numpy as np

from collections import OrderedDict
from gensim.models import KeyedVectors
from tflearn.data_utils import pad_sequences

TEXT_DIR = '../data/content.txt'
METADATA_DIR = '../data/metadata.tsv'


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


def cal_doa(front_labels, behind_labels, front_scores, behind_scores):
    """
    Get the doa value.

    Args:
        front_labels: The front true labels
        behind_labels: The behind true labels
        front_scores: The front predicted scores provided by network
        behind_scores: The behind predicted scores provided by network
    Returns:
        doa: The DOA value
    """
    def sig(front_value, behind_value):
        if front_value > behind_value:
            return 1
        else:
            return 0
    doa, molecule, denominator = 0.0, 0.0, 0.0
    data_size = len(front_scores)
    for index in range(data_size):
        predicted_bool = sig(front_scores[index][0], behind_scores[index][0])
        true_bool = sig(front_labels[index], behind_labels[index])
        molecule += predicted_bool and true_bool
        denominator += true_bool
    doa = molecule / denominator
    return doa


def create_prediction_file(output_file, all_id, all_labels, all_predict_scores):
    """
    Create the prediction file.

    Args:
        output_file: The all classes predicted results provided by network
        all_id: The data record id info provided by class Data
        all_labels: The all origin labels
        all_predict_scores: The all predict scores by threshold
    Raises:
        IOError: If the prediction file is not a <.json> file
    """
    if not output_file.endswith('.json'):
        raise IOError("✘ The prediction file is not a json file."
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


def create_metadata_file(embedding_size, output_file=METADATA_DIR):
    """
    Create the metadata file based on the corpus file(Use for the Embedding Visualization later).

    Args:
        embedding_size: The embedding size
        output_file: The metadata file (default: 'metadata.tsv')
    Raises:
        IOError: If word2vec model file doesn't exist
    """
    word2vec_file = '../data/word2vec_' + str(embedding_size) + '.txt'

    if not os.path.isfile(word2vec_file):
        raise IOError("✘ The word2vec file doesn't exist.")

    model = KeyedVectors.load_word2vec_format(open(word2vec_file, 'r'), binary=False, unicode_errors='replace')
    word2idx = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    word2idx_sorted = [(k, word2idx[k]) for k in sorted(word2idx, key=word2idx.get, reverse=False)]

    with open(output_file, 'w+') as fout:
        for word in word2idx_sorted:
            if word[0] is None:
                print("Empty Line, should replaced by any thing else, or will cause a bug of tensorboard")
                fout.write('<Empty Line>' + '\n')
            else:
                fout.write(word[0] + '\n')


def load_word2vec_matrix(embedding_size):
    """
    Return the word2vec model matrix.

    Args:
        embedding_size: The embedding size
    Returns:
        The word2vec model matrix
    Raises:
        IOError: If word2vec model file doesn't exist
    """
    word2vec_file = '../data/word2vec_' + str(embedding_size) + '.txt'

    if not os.path.isfile(word2vec_file):
        raise IOError("✘ The word2vec file doesn't exist. ")

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
    Return the class Data(includes the data tokenindex and data labels).

    Args:
        input_file: The research data
        word2vec_model: The word2vec model file
    Returns:
        The class Data(includes the data tokenindex and data labels)
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
        raise IOError("✘ The research data is not a json file. "
                      "Please preprocess the research data into the json file.")
    with open(input_file) as fin:
        id_list = [[], []]
        content_index_list = [[], []]
        question_index_list = [[], []]
        option_index_list = [[], []]
        labels_list = [[], []]
        total_line = 0

        for eachline in fin:
            data = json.loads(eachline)
            id_list[0].append(data['front_id'])
            id_list[1].append(data['behind_id'])

            content_index_list[0].append(_token_to_index(data['front_content']))
            content_index_list[1].append(_token_to_index(data['behind_content']))

            question_index_list[0].append(_token_to_index(data['front_question']))
            question_index_list[1].append(_token_to_index(data['behind_question']))

            option_index_list[0].append(_token_to_index(data['front_option']))
            option_index_list[1].append(_token_to_index(data['behind_option']))

            labels_list[0].append(data['front_diff'])
            labels_list[1].append(data['behind_diff'])
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


def load_data_and_labels(data_file, embedding_size, data_aug_flag):
    """
    Load research data from files, splits the data into words and generates labels.
    Return split sentences, labels and the max sentence length of the research data.

    Args:
        data_file: The research data
        embedding_size: The embedding size
        data_aug_flag: The flag of data augmented
    Returns:
        The class Data
    Raises:
        IOError: If word2vec model file doesn't exist
    """
    word2vec_file = '../data/word2vec_' + str(embedding_size) + '.txt'

    # Load word2vec model file
    if not os.path.isfile(word2vec_file):
        raise IOError("✘ The word2vec file doesn't exist. ")

    model = KeyedVectors.load_word2vec_format(open(word2vec_file, 'r'), binary=False, unicode_errors='replace')

    # Load data from files and split by words
    data = data_word2vec(input_file=data_file, word2vec_model=model)

    # plot_seq_len(data_file, data)

    return data


def pad_data(data, pad_seq_len):
    """
    Padding each sentence of research data according to the max sentence length.
    Return the padded data and data labels.

    Args:
        data: The research data
        pad_seq_len: The max sentence length of [content, question, option] text
    Returns:
        pad_content: The padded data
        pad_question: The padded data
        pad_option: The padded data
        labels: The data labels
    """
    pad_seq_len_list = list(map(int, pad_seq_len.split(',')))
    pad_content_list, pad_question_list, pad_option_list, labels_list = [], [], [], []
    for i in data.content_index:
        pad_content_list.append(pad_sequences(i, maxlen=pad_seq_len_list[0], value=0.))
    for j in data.question_index:
        pad_question_list.append(pad_sequences(j, maxlen=pad_seq_len_list[1], value=0.))
    for k in data.option_index:
        pad_option_list.append(pad_sequences(k, maxlen=pad_seq_len_list[2], value=0.))
    for l in data.labels:
        labels_list.append([[float(label)] for label in l])
    return pad_content_list, pad_question_list, pad_option_list, labels_list


def batch_iter(data_tuples, batch_size, num_epochs, shuffle=True):
    """
    含有 yield 说明不是一个普通函数，是一个 Generator.
    函数效果：对 data，一共分成 num_epochs 个阶段（epoch），在每个 epoch 内，如果 shuffle=True，就将 data 重新洗牌，
    批量生成 (yield) 一批一批的重洗过的 data，每批大小是 batch_size，一共生成 int(len(data)/batch_size)+1 批。

    Args:
        data_tuples: The data tuples
        batch_size: The size of the data batch
        num_epochs: The number of epochs
        shuffle: Shuffle or not (default: True)
    Returns:
        A batch iterator for data set
    """
    (x_content, x_question, x_option, y) = data_tuples
    x_content_front, x_content_behind = x_content[0], x_content[1]
    x_question_front, x_question_behind = x_question[0], x_question[1]
    x_option_front, x_option_behind = x_option[0], x_option[1]
    y_front, y_behind = y[0], y[1]

    data = list(zip(x_content_front, x_content_behind,
                    x_question_front, x_question_behind,
                    x_option_front, x_option_behind,
                    y_front, y_behind))

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
