"""TARNN layers."""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BiRNNLayer(nn.Module):
    def __init__(self, embedding_size, rnn_type, rnn_layers, rnn_hidden_size, dropout_keep_prob):
        super(BiRNNLayer, self).__init__()
        if rnn_type == 'LSTM':
            self.bi_rnn = nn.LSTM(input_size=embedding_size, hidden_size=rnn_hidden_size, num_layers=rnn_layers,
                                  batch_first=True, bidirectional=True, dropout=dropout_keep_prob)
        if rnn_type == 'GRU':
            self.bi_rnn = nn.GRU(input_size=embedding_size, hidden_size=rnn_hidden_size, num_layers=rnn_layers,
                                 batch_first=True, bidirectional=True, dropout=dropout_keep_prob)

    def forward(self, input_x):
        rnn_out, _ = self.bi_rnn(input_x)
        rnn_avg = torch.mean(rnn_out, dim=1)
        return rnn_out, rnn_avg


class AttentionLayer(nn.Module):
    # TODO
    def __init__(self, num_units, att_unit_size, att_type):
        super(AttentionLayer, self).__init__()
        self.att_type = att_type

    def forward(self, input_x, input_y):
        if self.att_type == 'normal':
            attention_matrix = torch.matmul(input_y, input_x.transpose(1, 2))
            attention_weight = torch.softmax(attention_matrix, dim=2)
            attention_visual = torch.mean(attention_matrix, dim=1)
            attention_out = torch.matmul(attention_weight, input_x)
            # TODO
            attention_out = torch.mean(attention_out, dim=1)
        if self.att_type == 'cosine':
            # cos_matrix = []
            # normalized_x = F.normalize(input_x, p=2, dim=2)
            pass
        if self.att_type == 'mlp':
            pass
        if self.att_type == 'islet':
            alpha_matrix = []
            seq_len = list(input_y.size())[-2]
            for t in range(seq_len):
                u_t = torch.matmul(torch.unsqueeze(input_y[:, t, :], dim=1), input_x.transpose(1, 2))
                u_t = torch.tanh(u_t)
                alpha_matrix.append(u_t)
            attention_matrix = torch.cat(alpha_matrix, dim=1)
            attention_matrix = torch.squeeze(attention_matrix, dim=2)
            attention_weight = F.softmax(attention_matrix, dim=1)
            attention_visual = torch.mean(attention_weight, dim=1)
            attention_out = torch.mul(torch.unsqueeze(attention_visual, dim=-1), input_x)
            attention_out = torch.mean(attention_out, dim=1)
        return attention_visual, attention_out


class HighwayLayer(nn.Module):
    def __init__(self, in_units, out_units):
        super(HighwayLayer, self).__init__()
        self.highway_linear = nn.Linear(in_features=in_units, out_features=out_units, bias=True)
        self.highway_gate = nn.Linear(in_features=in_units, out_features=out_units, bias=True)

    def forward(self, input_x):
        highway_g = torch.relu(self.highway_linear(input_x))
        highway_t = torch.sigmoid(self.highway_gate(input_x))
        highway_out = torch.mul(highway_g, highway_t) + torch.mul((1 - highway_t), input_x)
        return highway_out


class TARNN(nn.Module):
    """An implementation of TARNN"""
    def __init__(self, args, vocab_size, embedding_size, pretrained_embedding=None, dropout_rate=None):
        super(TARNN, self).__init__()
        """
        :param args: Arguments object.
        """
        self.args = args
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate
        self.pretrained_embedding = pretrained_embedding
        self._setup_layers()

    def _setup_embedding_layer(self):
        """
        Creating TARNN layers.
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
        self.bi_rnn_content = BiRNNLayer(embedding_size=self.embedding_size, rnn_type=self.args.rnn_type,
                                         rnn_layers=self.args.rnn_layers, rnn_hidden_size=self.args.rnn_dim,
                                         dropout_keep_prob=self.dropout_rate)
        self.bi_rnn_question = BiRNNLayer(embedding_size=self.embedding_size, rnn_type=self.args.rnn_type,
                                          rnn_layers=self.args.rnn_layers, rnn_hidden_size=self.args.rnn_dim,
                                          dropout_keep_prob=self.dropout_rate)
        self.bi_rnn_option = BiRNNLayer(embedding_size=self.embedding_size, rnn_type=self.args.rnn_type,
                                        rnn_layers=self.args.rnn_layers, rnn_hidden_size=self.args.rnn_dim,
                                        dropout_keep_prob=self.dropout_rate)

    def _setup_attention(self):
        """
        Creating attention Layer.
        """
        self.att_cq = AttentionLayer(num_units=self.args.attention_dim,
                                     att_unit_size=self.args.attention_dim,
                                     att_type=self.args.attention_type)
        self.att_oq = AttentionLayer(num_units=self.args.attention_dim,
                                     att_unit_size=self.args.attention_dim,
                                     att_type=self.args.attention_type)

    def _setup_highway_layer(self):
        """
         Creating Highway Layer.
         """
        self.highway = HighwayLayer(in_units=self.args.fc_dim, out_units=self.args.fc_dim)

    def _setup_fc_layer(self):
        """
         Creating FC Layer.
         """
        self.fc = nn.Linear(in_features=self.args.rnn_dim * 2 * 3, out_features=self.args.fc_dim, bias=True)
        self.out = nn.Linear(in_features=self.args.fc_dim, out_features=1, bias=True)

    def _setup_dropout(self):
        """
         Creating Dropout.
         """
        self.dropout = nn.Dropout(self.dropout_rate)

    def _setup_layers(self):
        """
        Creating layers of model.
        1. Embedding Layer.
        2. Bi-RNN Layer.
        3. Attention Layer.
        4. Highway Layer.
        4. FC Layer.
        """
        self._setup_embedding_layer()
        self._setup_bi_rnn_layer()
        self._setup_attention()
        self._setup_highway_layer()
        self._setup_fc_layer()
        self._setup_dropout()

    def _sub_network(self, x_content, x_question, x_option):
        embedded_sentence_content = self.embedding(x_content)
        embedded_sentence_question = self.embedding(x_question)
        embedded_sentence_option = self.embedding(x_option)

        # Average Vectors
        # [batch_size, embedding_size]
        embedded_content_average = torch.mean(embedded_sentence_content, dim=1)
        embedded_question_average = torch.mean(embedded_sentence_question, dim=1)
        embedded_option_average = torch.mean(embedded_sentence_option, dim=1)

        # Bi-LSTM Layer
        lstm_out_content, lstm_avg_content = self.bi_rnn_content(embedded_sentence_content)
        lstm_out_question, lstm_avg_question = self.bi_rnn_question(embedded_sentence_question)
        lstm_out_option, lstm_avg_option = self.bi_rnn_option(embedded_sentence_option)

        # Attention Layer
        attention_cq_visual, attention_cq = self.att_cq(lstm_out_content, lstm_out_question)
        attention_oq_visual, attention_oq = self.att_oq(lstm_out_option, lstm_out_question)

        # Concat
        # shape of att_out: [batch_size, lstm_hidden_size * 2 * 3]
        att_out = torch.cat((attention_cq, lstm_avg_question, attention_oq), dim=1)

        # Fully Connected Layer
        fc_out = self.fc(att_out)

        # Highway Layer
        highway_out = self.highway(fc_out)

        # Dropout
        h_drop = self.dropout(highway_out)

        return h_drop

    def forward(self, x_fb_content, x_fb_question, x_fb_option):
        """
        Forward propagation pass.
        :param x_fb_content: Front & Behind Content tensors with features. <list>
        :param x_fb_question: Front & Behind Question tensors with features. <list>
        :param x_fb_option: Front & Behind Option tensors  with features. <list>
        # TODO
        :return class_capsule_output: Class capsule outputs.
        """
        self.f_out = self._sub_network(x_fb_content[0], x_fb_question[0], x_fb_option[0])
        self.b_out = self._sub_network(x_fb_content[1], x_fb_question[1], x_fb_option[1])

        # Final scores
        f_logits = self.out(self.f_out).squeeze()
        b_logits = self.out(self.b_out).squeeze()
        f_scores = torch.sigmoid(f_logits)
        b_scores = torch.sigmoid(b_logits)

        logits = (f_logits, b_logits)
        scores = (f_scores, b_scores)
        return logits, scores


class MSE_Loss(nn.Module):
    def __init__(self):
        super(MSE_Loss, self).__init__()
        self.MSELoss = nn.MSELoss(reduce=True, size_average=True)

    def forward(self, predict_y, input_y):
        # Loss
        f_loss = self.MSELoss(predict_y[0], input_y[0])
        b_loss = self.MSELoss(predict_y[1], input_y[1])
        losses = f_loss + b_loss
        return losses
