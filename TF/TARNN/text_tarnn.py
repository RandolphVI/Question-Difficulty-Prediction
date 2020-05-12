# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import numpy as np
import tensorflow as tf


class TextTARNN(object):
    """A TARNN for text classification."""

    def __init__(
            self, sequence_length, vocab_size, embedding_type, embedding_size, rnn_hidden_size, rnn_type, rnn_layers,
            attention_type, fc_hidden_size, l2_reg_lambda=0.0, pretrained_embedding=None):

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
                # shape of `outputs_state_fw`: [batch_size, rnn_hidden_size]
                outputs, state = tf.nn.bidirectional_dynamic_rnn(fw_rnn_cell, bw_rnn_cell, input_x, dtype=tf.float32)

            # Concat output
            # [batch_size, sequence_length, rnn_hidden_size * 2]
            rnn_out = tf.concat(outputs, axis=2, name=name + "rnn_out")

            # [batch_size, rnn_hidden_size * 2]
            rnn_avg = tf.reduce_mean(rnn_out, axis=1, name=name + "rnn_avg")

            return rnn_out, rnn_avg

        def _attention(input_x, input_y, name=""):
            """
            Attention Layer.
            Args:
                input_x: [batch_size, sequence_length_1, rnn_hidden_size * 2]
                input_y: [batch_size, sequence_length_2, rnn_hidden_size * 2]
                name: Scope name
            Returns:
                'normal':
                    attention_matrix: [batch_size, seq_len_2, seq_len_1]
                    attention_weight: [batch_size, seq_len_2, seq_len_1]
                    attention_visual: [batch_size, seq_len_1]
                    attention_out: [batch_size, rnn_hidden_size * 2]
                'cosine':
                    attention_matrix: [batch_size, seq_len_2, seq_len_1]
                    attention_weight: [batch_size, seq_len_2, seq_len_1]
                    attention_visual: [batch_size, seq_len_1]
                    attention_out: [batch_size, rnn_hidden_size * 2]
                'mlp':
                    attention_matrix: [batch_size, seq_len_1 + seq_len_2, seq_len_1]
                    attention_weight: [batch_size, seq_len_1 + seq_len_2, seq_len_1]
                    attention_visual: [batch_size, seq_len_1]
                    attention_out: [batch_size, rnn_hidden_size * 2]
            """
            if attention_type == 'normal':
                with tf.name_scope(name + "attention"):
                    attention_matrix = tf.matmul(input_y, tf.transpose(input_x, perm=[0, 2, 1]))
                    attention_weight = tf.nn.softmax(attention_matrix, axis=2, name="attention_matrix")
                    attention_visual = tf.reduce_mean(attention_matrix, axis=1, name="visual")
                    attention_out = tf.matmul(attention_weight, input_x)
                    attention_out = tf.reduce_mean(attention_out, axis=1)
            if attention_type == 'cosine':
                with tf.name_scope(name + "cos_attention"):
                    cos_matrix = []
                    seq_len = input_y.get_shape().as_list()[-2]
                    normalize_x = tf.nn.l2_normalize(input_x, 2)
                    y = tf.unstack(input_y, axis=1)
                    for i in range(seq_len):
                        normalize_y = tf.nn.l2_normalize(tf.expand_dims(y[i], axis=1), 2)
                        cos_similarity = tf.reduce_sum(tf.multiply(normalize_y, normalize_x), axis=2)
                        # cos_similarity: [batch_size, seq_len_1]
                        cos_matrix.append(cos_similarity)
                    # attention_matrix: [batch_size, seq_len_2, seq_len_1]
                    attention_matrix = tf.stack(cos_matrix, axis=1, name="attention_matrix")
                    attention_visual = tf.reduce_mean(attention_matrix, axis=1, name="visual")
                    attention_out = tf.multiply(tf.expand_dims(attention_visual, axis=-1), input_x)
                    attention_out = tf.reduce_mean(attention_out, axis=1)
            if attention_type == 'mlp':
                with tf.name_scope(name + "mlp_attention"):
                    x = tf.concat([input_x, input_y], axis=1)
                    num_units = x.get_shape().as_list()[-1]
                    seq_len = input_x.get_shape().as_list()[-2]
                    W_s1 = tf.Variable(tf.truncated_normal(shape=[num_units, seq_len],
                                                           stddev=0.1, dtype=tf.float32), name="W_s1")
                    W_s2 = tf.Variable(tf.truncated_normal(shape=[seq_len, seq_len],
                                                           stddev=0.1, dtype=tf.float32), name="W_s2")
                    attention_matrix = tf.map_fn(
                        fn=lambda x: tf.matmul(x, W_s2),
                        elems=tf.tanh(
                            tf.map_fn(
                                fn=lambda x: tf.matmul(x, W_s1),
                                elems=x,
                                dtype=tf.float32
                            )
                        )
                    )
                    attention_weight = tf.nn.softmax(attention_matrix, axis=2, name="attention_matrix")
                    attention_visual = tf.reduce_mean(attention_matrix, axis=1, name="visual")
                    attention_out = tf.matmul(attention_weight, input_x)
                    attention_out = tf.reduce_mean(attention_out, axis=1)
            if attention_type == 'islet':
                with tf.name_scope(name + 'islet_attention'):
                    alpha_matrix = []
                    seq_len = input_y.get_shape().as_list()[-2]
                    y = tf.unstack(input_y, axis=1)
                    for t in range(seq_len):
                        u_t = tf.matmul(tf.expand_dims(y[t], axis=1), tf.transpose(input_x, perm=[0, 2, 1]))
                        # u_t: [batch_size, 1, seq_len_1]
                        u_t = tf.tanh(u_t)
                        alpha_matrix.append(u_t)
                    attention_matrix = tf.stack(alpha_matrix, axis=1)
                    # attention_matrix: [batch_size, seq_len_2, seq_len_1] (after squeeze)
                    attention_matrix = tf.squeeze(attention_matrix, axis=2)
                    attention_weight = tf.nn.softmax(attention_matrix, name="attention_matrix")
                    attention_visual = tf.reduce_mean(attention_weight, axis=1, name="visual")
                    attention_out = tf.multiply(tf.expand_dims(attention_visual, axis=-1), input_x)
                    attention_out = tf.reduce_sum(attention_out, axis=1)
            return attention_visual, attention_out

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

        def _linear(input_, output_size, scope="SimpleLinear"):
            """
            Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]
            Args:
                input_: a tensor or a list of 2D, batch x n, Tensors.
                output_size: int, second dimension of W[i].
                scope: VariableScope for the created subgraph; defaults to "SimpleLinear".
            Returns:
                A 2D Tensor with shape [batch x output_size] equal to
                sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
            Raises:
                ValueError: if some of the arguments has unspecified or wrong shape.
            """

            shape = input_.get_shape().as_list()
            if len(shape) != 2:
                raise ValueError("Linear is expecting 2D arguments: {0}".format(str(shape)))
            if not shape[1]:
                raise ValueError("Linear expects shape[1] of arguments: {0}".format(str(shape)))
            input_size = shape[1]

            # Now the computation.
            with tf.variable_scope(scope):
                W = tf.get_variable("W", [input_size, output_size], dtype=input_.dtype)
                b = tf.get_variable("b", [output_size], dtype=input_.dtype)

            return tf.nn.xw_plus_b(input_, W, b)

        def _highway_layer(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu):
            """
            Highway Network (cf. http://arxiv.org/abs/1505.00387).
            t = sigmoid(Wy + b)
            z = t * g(Wy + b) + (1 - t) * y
            where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
            """

            for idx in range(num_layers):
                g = f(_linear(input_, size, scope=("highway_lin_{0}".format(idx))))
                t = tf.sigmoid(_linear(input_, size, scope=("highway_gate_{0}".format(idx))) + bias)
                output = t * g + (1. - t) * input_
                input_ = output

            return output

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

        self.rnn_out_content, self.rnn_avg_content = _bi_rnn_layer(self.embedded_sentence_content, name="content_")
        self.rnn_out_question, self.rnn_avg_question = _bi_rnn_layer(self.embedded_sentence_question, name="question_")
        self.rnn_out_option, self.rnn_avg_option = _bi_rnn_layer(self.embedded_sentence_option, name="option_")

        # Attention Layer
        self.visual_cq, self.attention_cq = _attention(self.rnn_out_content, self.rnn_out_question, name="cq_")
        self.visual_oq, self.attention_oq = _attention(self.rnn_out_option, self.rnn_out_question, name="oq_")

        # attention_out: [batch_size, hidden_size * 2 * 3]
        self.attention_out = tf.concat([self.attention_cq, self.rnn_avg_question, self.attention_oq], axis=1)

        # Fully Connected Layer
        self.fc_out = _fc_layer(self.attention_out)

        # Highway Layer
        with tf.name_scope("highway"):
            self.highway = _highway_layer(self.fc_out, self.fc_out.get_shape()[1], num_layers=1, bias=0)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.highway, self.dropout_keep_prob)

        # Final scores
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal(shape=[fc_hidden_size, 1],
                                                stddev=0.1, dtype=tf.float32), name="W")
            b = tf.Variable(tf.constant(value=0.1, shape=[1], dtype=tf.float32), name="b")
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")
            self.scores = tf.sigmoid(self.logits, name="scores")

        # Calculate mean cross-entropy loss, L2 loss
        with tf.name_scope("loss"):
            losses = tf.reduce_mean(tf.square(self.input_y - self.scores), name="losses")
            l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()],
                                 name="l2_losses") * l2_reg_lambda
            self.loss = tf.add(losses, l2_losses, name="loss")
