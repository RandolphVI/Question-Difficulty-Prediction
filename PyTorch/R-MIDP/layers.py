# -*- coding:utf-8 -*-
__author__ = 'randolph'

"""RMIDP layers."""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BiRNNLayer(nn.Module):
    def __init__(self, input_units, rnn_type, rnn_layers, rnn_hidden_size, dropout_keep_prob):
        super(BiRNNLayer, self).__init__()
        if rnn_type == 'LSTM':
            self.bi_rnn = nn.LSTM(input_size=input_units, hidden_size=rnn_hidden_size, num_layers=rnn_layers,
                                  batch_first=True, bidirectional=True, dropout=dropout_keep_prob)
        if rnn_type == 'GRU':
            self.bi_rnn = nn.GRU(input_size=input_units, hidden_size=rnn_hidden_size, num_layers=rnn_layers,
                                 batch_first=True, bidirectional=True, dropout=dropout_keep_prob)

    def forward(self, input_x):
        rnn_out, _ = self.bi_rnn(input_x)
        rnn_avg = torch.mean(rnn_out, dim=1)
        return rnn_out, rnn_avg


class RMIDP(nn.Module):
    """An implementation of RMIDP"""

    def __init__(self, args, vocab_size, embedding_size, pretrained_embedding=None):
        super(RMIDP, self).__init__()
        """
        :param args: Arguments object.
        """
        self.args = args
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.pretrained_embedding = pretrained_embedding
        self._setup_layers()

    def _setup_embedding_layer(self):
        """
        Creating Embedding layers.
        """
        if self.pretrained_embedding is None:
            embedding_weight = torch.FloatTensor(np.random.uniform(-1, 1, size=(self.vocab_size, self.embedding_size)))
            embedding_weight = Variable(embedding_weight, requires_grad=True)
        else:
            if self.args.embedding_type == 0:
                embedding_weight = torch.from_numpy(self.pretrained_embedding).float()
            if self.args.embedding_type == 1:
                embedding_weight = Variable(torch.from_numpy(self.pretrained_embedding).float(), requires_grad=True)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, _weight=embedding_weight)

    def _setup_bi_rnn_layer(self):
        """
        Creating Bi-RNN Layer.
        """
        self.bi_rnn = BiRNNLayer(input_units=self.embedding_size, rnn_type=self.args.rnn_type,
                                 rnn_layers=self.args.rnn_layers, rnn_hidden_size=self.args.rnn_dim,
                                 dropout_keep_prob=self.args.dropout_rate)

    def _setup_fc_layer(self):
        """
         Creating FC Layer.
         """
        self.fc = nn.Linear(in_features=self.args.rnn_dim * 2, out_features=self.args.fc_dim, bias=True)
        self.out = nn.Linear(in_features=self.args.fc_dim, out_features=1, bias=True)

    def _setup_layers(self):
        """
        Creating layers of model.
        1. Embedding Layer.
        2. Bi-RNN Layer.
        3. FC Layer.
        """
        self._setup_embedding_layer()
        self._setup_bi_rnn_layer()
        self._setup_fc_layer()

    def _sub_network(self, x_content, x_question, x_option):
        embedded_sentence_content = self.embedding(x_content)
        embedded_sentence_question = self.embedding(x_question)
        embedded_sentence_option = self.embedding(x_option)

        # Concat Vectors
        # [batch_size, sequence_length_all, embedding_size]
        embedded_sentence_all = torch.cat((embedded_sentence_content, embedded_sentence_question,
                                           embedded_sentence_option), dim=1)

        # Bi-RNN Layer
        rnn_out, rnn_pooled = self.bi_rnn(embedded_sentence_all)

        # Fully Connected Layer
        fc_out = self.fc(rnn_pooled)

        # Final scores
        logits = self.out(fc_out).squeeze()
        scores = torch.sigmoid(logits)

        return logits, scores

    def forward(self, x_fb_content, x_fb_question, x_fb_option):
        """
        Forward propagation pass.
        :param x_fb_content: Front & Behind Content tensors with features. <list>
        :param x_fb_question: Front & Behind Question tensors with features. <list>
        :param x_fb_option: Front & Behind Option tensors  with features. <list>
        :return logits: The predicted logistic values.
        :return scores: The predicted scores.
        """
        f_logits, f_scores = self._sub_network(x_fb_content[0], x_fb_question[0], x_fb_option[0])
        b_logits, b_scores = self._sub_network(x_fb_content[1], x_fb_question[1], x_fb_option[1])

        logits = (f_logits, b_logits)
        scores = (f_scores, b_scores)
        return logits, scores


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, predict_y, input_y):
        # Loss
        value = (predict_y[0] - predict_y[1]) - (input_y[0] - input_y[1])
        losses = torch.mean(torch.pow(value, 2))
        return losses
