# -*- coding:utf-8 -*-
__author__ = 'randolph'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import utils.data_helpers as dh


def truncated_normal_(tensor, mean=0, std=0.1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


class _BiLSTMLayer(nn.Module):
    def __init__(self, embedding_size, lstm_hidden_size, dropout_keep_prob):
        super(_BiLSTMLayer, self).__init__()
        self.bi_lstm = nn.LSTM(input_size=embedding_size, hidden_size=lstm_hidden_size, num_layers=1,
                               batch_first=True, bidirectional=True, dropout=dropout_keep_prob)

    def forward(self, input_x):
        lstm_out, _ = self.bi_lstm(input_x)
        lstm_pooled = torch.max(lstm_out, dim=1)[0]
        return lstm_out, lstm_pooled


class TextRMIDP(nn.Module):
    """A RMIDP for text classification."""

    def __init__(
            self, seq_len_list, vocab_size, lstm_hidden_size, fc_hidden_size, embedding_size,
            embedding_type, pretrained_embedding=None, dropout_keep_prob=None):
        super(TextRMIDP, self).__init__()
        # Embedding Layer
        # Use random generated the word vector by default
        # Can also be obtained through our own word vectors trained by our corpus
        if pretrained_embedding is None:
            embedding_weight = torch.FloatTensor(np.random.uniform(-1, 1, size=(vocab_size, embedding_size)))
            embedding_weight = Variable(embedding_weight, requires_grad=True)
        else:
            if embedding_type == 0:
                embedding_weight = torch.from_numpy(pretrained_embedding).float()
            if embedding_type == 1:
                embedding_weight = Variable(torch.from_numpy(pretrained_embedding).float(), requires_grad=True)
        self.embedding = nn.Embedding(vocab_size, embedding_size, _weight=embedding_weight)

        # Bi-LSTM Layer
        self._bi_lstm = _BiLSTMLayer(embedding_size, lstm_hidden_size, dropout_keep_prob)

        # Fully Connected Layer
        self.fc = nn.Linear(lstm_hidden_size * 2, fc_hidden_size, bias=True)

        # Add dropout
        self.dropout = nn.Dropout(dropout_keep_prob)

        # Final scores
        self.out = nn.Linear(fc_hidden_size, 1, bias=True)

        # for name, param in self.named_parameters():
        #     if 'embedding' not in name and 'weight' in name:
        #         truncated_normal_(param.data, mean=0, std=0.1)
        #     else:
        #         nn.init.constant_(param.data, 0.1)

    def forward(self, x_content, x_question, x_option):
        # Embedding Layer
        embedded_sentence_content = self.embedding(x_content)
        embedded_sentence_question = self.embedding(x_question)
        embedded_sentence_option = self.embedding(x_option)

        # Concat Vectors
        # [batch_size, sequence_length_all, embedding_size]
        embedded_sentence_all = torch.cat((embedded_sentence_content, embedded_sentence_question,
                                           embedded_sentence_option), dim=1)
        print("embedded_sentence_all: {0}".format(embedded_sentence_all.size()))

        # Bi-LSTM Layer
        lstm_out, lstm_pooled = self._bi_lstm(embedded_sentence_all)
        print("lstm_out: {0}".format(lstm_pooled.size()))

        # Fully Connected Layer
        fc_out = self.fc(lstm_pooled)
        print("fc: {0}".format(fc_out.size()))

        # Dropout
        fc_drop = self.dropout(fc_out)

        # Final scores
        logits = self.out(fc_drop).squeeze()
        scores = torch.sigmoid(logits)

        return logits, scores


