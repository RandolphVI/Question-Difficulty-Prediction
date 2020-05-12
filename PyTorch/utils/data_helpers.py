# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import os
import math
import gensim
import logging
import json
import torch
import numpy as np
import pandas as pd

from scipy import stats
from texttable import Texttable
from gensim.models import KeyedVectors


def option():
    """
    Choose training or restore pattern.

    Returns:
        The OPTION
    """
    OPTION = input("[Input] Train or Restore? (T/R): ")
    while not (OPTION.upper() in ['T', 'R']):
        OPTION = input("[Warning] The format of your input is illegal, please re-input: ")
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
    logger = logging.getLogger(name)
    logger.setLevel(level)
    log_dir = os.path.dirname(input_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File Handler
    fh = logging.FileHandler(input_file, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # stream Handler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.WARNING)
    logger.addHandler(sh)
    return logger


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


def create_prediction_file(save_dir, identifiers, predictions):
    """
    Create the prediction file.

    Args:
        save_dir: The all classes predicted results provided by network
        identifiers: The data record id
        predictions: The predict scores
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    preds_file = os.path.abspath(os.path.join(save_dir, 'predictions.csv'))
    out = pd.DataFrame()
    out["id"] = identifiers
    out["predictions"] = [round(float(i), 4) for i in predictions]
    out.to_csv(preds_file, index=None)


def evaluation(true_label, pred_label):
    """
    Calculate the PCC & DOA.

    Args:
        true_label: The true labels
        pred_label: The predicted labels
    Returns:
        The value of PCC & DOA
    """
    # compute pcc
    pcc, _ = stats.pearsonr(pred_label, true_label)
    if math.isnan(pcc):
        print('[Error]: PCC=nan', true_label, pred_label)
    # compute doa
    n = 0
    correct_num = 0
    for i in range(len(true_label) - 1):
        for j in range(i + 1, len(true_label)):
            if (true_label[i] > true_label[j]) and (pred_label[i] > pred_label[j]):
                correct_num += 1
            elif (true_label[i] == true_label[j]) and (pred_label[i] == pred_label[j]):
                continue
            elif (true_label[i] < true_label[j]) and (pred_label[i] < pred_label[j]):
                correct_num += 1
            n += 1
    if n == 0:
        print(true_label)
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


def load_word2vec_matrix(word2vec_file):
    """
    Return the word2vec model matrix.

    Args:
        word2vec_file: The word2vec file
    Returns:
        The word2vec model matrix
    Raises:
        IOError: If word2vec model file doesn't exist
    """
    if not os.path.isfile(word2vec_file):
        raise IOError("[Error] The word2vec file doesn't exist. ")

    model = KeyedVectors.load_word2vec_format(open(word2vec_file, 'r'), binary=False, unicode_errors='replace')
    vocab_size = model.wv.vectors.shape[0]
    embedding_size = model.vector_size
    vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    embedding_matrix = np.zeros([vocab_size, embedding_size])
    for key, value in vocab.items():
        if key is not None:
            embedding_matrix[value] = model[key]
    return vocab_size, embedding_size, embedding_matrix


def data_word2vec(input_file, word2vec_model):
    """
    Create the research data tokenindex based on the word2vec model file.
    Return the class _Data() (includes the data tokenindex and data labels).

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
        f_id_list = []
        b_id_list = []
        f_content_index_list = []
        b_content_index_list = []
        f_question_index_list = []
        b_question_index_list = []
        f_option_index_list = []
        b_option_index_list = []
        f_labels_list = []
        b_labels_list = []

        for eachline in fin:
            data = json.loads(eachline)
            f_id = data['front_id']
            b_id = data['behind_id']
            f_content_text = data['front_content']
            b_content_text = data['behind_content']
            f_question_text = data['front_question']
            b_question_text = data['behind_question']
            f_option_text = data['front_option']
            b_option_text = data['behind_option']
            f_labels = data['front_diff']
            b_labels = data['behind_diff']

            f_id_list.append(f_id)
            b_id_list.append(b_id)
            f_content_index_list.append(_token_to_index(f_content_text))
            b_content_index_list.append(_token_to_index(b_content_text))
            f_question_index_list.append(_token_to_index(f_question_text))
            b_question_index_list.append(_token_to_index(b_question_text))
            f_option_index_list.append(_token_to_index(f_option_text))
            b_option_index_list.append(_token_to_index(b_option_text))
            f_labels_list.append(f_labels)
            b_labels_list.append(b_labels)

    class _Data:
        def __init__(self):
            pass

        @property
        def f_id(self):
            return f_id_list

        @property
        def b_id(self):
            return b_id_list

        @property
        def f_content_index(self):
            return f_content_index_list

        @property
        def b_content_index(self):
            return b_content_index_list

        @property
        def f_question_index(self):
            return f_question_index_list

        @property
        def b_question_index(self):
            return b_question_index_list

        @property
        def f_option_index(self):
            return f_option_index_list

        @property
        def b_option_index(self):
            return b_option_index_list

        @property
        def f_labels(self):
            return f_labels_list

        @property
        def b_labels(self):
            return b_labels_list

    return _Data()


def load_data_and_labels(data_file, word2vec_file):
    """
    Load research data from files, splits the data into words and generates labels.
    Return split sentences, labels and the max sentence length of the research data.

    Args:
        data_file: The research data
        word2vec_file: The word2vec model file
    Returns:
        The class _Data()
    Raises:
        IOError: If word2vec model file doesn't exist
    """
    # Load word2vec file
    if not os.path.isfile(word2vec_file):
        raise IOError("[Error] The word2vec file doesn't exist. ")

    model = KeyedVectors.load_word2vec_format(open(word2vec_file, 'r'), binary=False, unicode_errors='replace')

    # Load data from files and split by words
    data = data_word2vec(input_file=data_file, word2vec_model=model)
    return data


def pad_sequence_with_maxlen(sequences, batch_first=False, padding_value=0, maxlen_arg=None):
    r"""
    Change from the raw code in torch.nn.utils.rnn for the need to pad with a assigned length
    Pad a list of variable length Tensors with ``padding_value``
    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length. For example, if the input is list of
    sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
    otherwise.
    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.
    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])
    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.
    Arguments:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements. Default: 0.
        maxlen:the the max length you want to pad
    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    # if maxlen_arg != None and maxlen_arg < max_len:
    #   max_len = max_len_arg
    if maxlen_arg == None:
        max_len = max([s.size(0) for s in sequences])
    else:
        max_len = maxlen_arg
    #

    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = min(max_len, tensor.size(0))
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor[:length]
        else:
            out_tensor[:length, i, ...] = tensor[:length]

    return out_tensor


def pad_data(data, pad_seq_len):
    """
    Version for PyTorch

    Args:
        data: The research data
        pad_seq_len: The max sentence length of [content, question, option] text
    Returns:
        pad_content: The padded data
        pad_question: The padded data
        pad_option: The padded data
        labels: The data labels
    """
    f_pad_content = pad_sequence_with_maxlen([torch.tensor(item) for item in data.f_content_index],
                                             batch_first=True, padding_value=0., maxlen_arg=pad_seq_len[0])
    b_pad_content = pad_sequence_with_maxlen([torch.tensor(item) for item in data.b_content_index],
                                             batch_first=True, padding_value=0., maxlen_arg=pad_seq_len[0])
    f_pad_question = pad_sequence_with_maxlen([torch.tensor(item) for item in data.f_question_index],
                                              batch_first=True, padding_value=0., maxlen_arg=pad_seq_len[1])
    b_pad_question = pad_sequence_with_maxlen([torch.tensor(item) for item in data.b_question_index],
                                              batch_first=True, padding_value=0., maxlen_arg=pad_seq_len[1])
    f_pad_option = pad_sequence_with_maxlen([torch.tensor(item) for item in data.f_option_index],
                                            batch_first=True, padding_value=0., maxlen_arg=pad_seq_len[2])
    b_pad_option = pad_sequence_with_maxlen([torch.tensor(item) for item in data.b_option_index],
                                            batch_first=True, padding_value=0., maxlen_arg=pad_seq_len[2])
    f_labels = torch.tensor(data.f_labels)
    b_labels = torch.tensor(data.b_labels)

    fb_pad_content = (f_pad_content, b_pad_content)
    fb_pad_question = (f_pad_question, b_pad_question)
    fb_pad_option = (f_pad_option, b_pad_option)
    fb_labels = (f_labels, b_labels)

    return fb_pad_content, fb_pad_question, fb_pad_option, fb_labels

