# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import numpy as np
import tensorflow as tf


class TextRMIDP(object):
    """A RMIDP for text classification."""

    def __init__(
            self, sequence_length, vocab_size, embedding_type, embedding_size, rnn_hidden_size, rnn_type,
            rnn_layers, fc_hidden_size, l2_reg_lambda=0.0, pretrained_embedding=None):

        # Placeholders for input, output, dropout_prob and training_tag
        self.input_x_content = tf.placeholder(tf.int32, [None, sequence_length[0]], name="input_x_content")
        self.input_x_question = tf.placeholder(tf.int32, [None, sequence_length[1]], name="input_x_question")
        self.input_x_option = tf.placeholder(tf.int32, [None, sequence_length[2]], name="input_x_option")
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")

        def _get_rnn_cell(rnn_hidden_size, rnn_type):
            if rnn_type == 'RNN':
                return tf.nn.rnn_cell.BasicRNNCell(rnn_hidden_size)
            if rnn_type == 'LSTM':
                return tf.nn.rnn_cell.BasicLSTMCell(rnn_hidden_size)
            if rnn_type == 'GRU':
                return tf.nn.rnn_cell.GRUCell(rnn_hidden_size)

        def _bi_rnn_layer(input_x, name=""):
            # Bi-RNN Layer
            with tf.variable_scope(name + "Bi_rnn", reuse=tf.AUTO_REUSE):
                fw_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([_get_rnn_cell(rnn_hidden_size, rnn_type)
                                                           for _ in range(rnn_layers)])
                bw_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([_get_rnn_cell(rnn_hidden_size, rnn_type)
                                                           for _ in range(rnn_layers)])
                if self.dropout_keep_prob is not None:
                    fw_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(fw_rnn_cell, output_keep_prob=self.dropout_keep_prob)
                    bw_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(bw_rnn_cell, output_keep_prob=self.dropout_keep_prob)

                # Creates a dynamic bidirectional recurrent neural network
                # shape of `outputs`: tuple -> (outputs_fw, outputs_bw)
                # shape of `outputs_fw`: [batch_size, sequence_length, rnn_hidden_size]

                # shape of `state`: tuple -> (outputs_state_fw, output_state_bw)
                # shape of `outputs_state_fw`: tuple -> (c, h) c: memory cell; h: hidden state
                outputs, state = tf.nn.bidirectional_dynamic_rnn(fw_rnn_cell, bw_rnn_cell, input_x, dtype=tf.float32)

            # Concat output
            # [batch_size, sequence_length, rnn_hidden_size * 2]
            rnn_out = tf.concat(outputs, axis=2, name=name + "rnn_out")

            # [batch_size, rnn_hidden_size * 2]
            rnn_pooled = tf.reduce_max(rnn_out, axis=1, name=name + "rnn_pooled")

            return rnn_pooled

        def _fc_layer(input_x, name=""):
            """
            Fully Connected Layer.
            Args:
                input_x:
                name: Scope name
            Returns:
                [batch_size, fc_hidden_size]
            """
            with tf.name_scope(name + "fc"):
                num_units = input_x.get_shape().as_list()[-1]
                W = tf.Variable(tf.truncated_normal(shape=[num_units, fc_hidden_size],
                                                    stddev=0.1, dtype=tf.float32), name="W")
                b = tf.Variable(tf.constant(value=0.1, shape=[fc_hidden_size], dtype=tf.float32), name="b")
                fc = tf.nn.xw_plus_b(input_x, W, b)
                fc_out = tf.nn.relu(fc)
            return fc_out

        # Embedding Layer
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            # Use random generated the word vector by default
            # Can also be obtained through our own word vectors trained by our corpus
            if pretrained_embedding is None:
                self.embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], minval=-1.0, maxval=1.0,
                                                               dtype=tf.float32), trainable=True, name="embedding")
            else:
                if embedding_type == 0:
                    self.embedding = tf.constant(pretrained_embedding, dtype=tf.float32, name="embedding")
                if embedding_type == 1:
                    self.embedding = tf.Variable(pretrained_embedding, trainable=True,
                                                 dtype=tf.float32, name="embedding")
            # [batch_size, sequence_length, embedding_size]
            self.embedded_sentence_content = tf.nn.embedding_lookup(self.embedding, self.input_x_content)
            self.embedded_sentence_question = tf.nn.embedding_lookup(self.embedding, self.input_x_question)
            self.embedded_sentence_option = tf.nn.embedding_lookup(self.embedding, self.input_x_option)

        # Concat -> embedded_sentence_all: [batch_size, sequence_length_all, embedding_size]
        self.embedded_sentence_all = tf.concat([self.embedded_sentence_content, self.embedded_sentence_question,
                                               self.embedded_sentence_option], axis=1)

        # Bi-RNN Layer
        # bi_rnn_out: [batch_size, rnn_hidden_size * 2]
        self.bi_rnn_out = _bi_rnn_layer(self.embedded_sentence_all, name="total_")

        # Fully Connected Layer
        self.fc_out = _fc_layer(self.bi_rnn_out)

        # Add dropout
        with tf.name_scope("dropout"):
            self.fc_drop = tf.nn.dropout(self.fc_out, self.dropout_keep_prob)

        # Final scores
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal(shape=[fc_hidden_size, 1],
                                                stddev=0.1, dtype=tf.float32), name="W")
            b = tf.Variable(tf.constant(value=0.1, shape=[1], dtype=tf.float32), name="b")
            self.logits = tf.nn.xw_plus_b(self.fc_drop, W, b, name="logits")
            self.scores = tf.sigmoid(self.logits, name="scores")

        # Calculate mean cross-entropy loss, L2 loss
        with tf.name_scope("loss"):
            losses = tf.reduce_mean(tf.square(self.input_y - self.scores), name="losses")
            l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()],
                                 name="l2_losses") * l2_reg_lambda
            self.loss = tf.add(losses, l2_losses, name="loss")