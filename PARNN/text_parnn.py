# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import numpy as np
import tensorflow as tf

from tensorflow import tanh
from tensorflow import sigmoid
from tensorflow.contrib import rnn
from tensorflow.python.ops import array_ops
from tensorflow.contrib.layers import batch_norm


class BatchNormLSTMCell(rnn.RNNCell):
    """Batch normalized LSTM (cf. http://arxiv.org/abs/1603.09025)"""

    def __init__(self, num_units, is_training=False, forget_bias=1.0,
                 activation=tanh, reuse=None):
        """Initialize the BNLSTM cell.

        Args:
          num_units: int, The number of units in the BNLSTM cell.
          forget_bias: float, The bias added to forget gates (see above).
            Must set to `0.0` manually when restoring from CudnnLSTM-trained
            checkpoints.
          activation: Activation function of the inner states.  Default: `tanh`.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.
        """
        self._num_units = num_units
        self._is_training = is_training
        self._forget_bias = forget_bias
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return rnn.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__, reuse=self._reuse):
            c, h = state
            input_size = inputs.get_shape().as_list()[1]
            W_xh = tf.get_variable('W_xh',
                                   [input_size, 4 * self._num_units],
                                   initializer=orthogonal_initializer())
            W_hh = tf.get_variable('W_hh',
                                   [self._num_units, 4 * self._num_units],
                                   initializer=bn_lstm_identity_initializer(0.95))
            bias = tf.get_variable('bias', [4 * self._num_units])

            xh = tf.matmul(inputs, W_xh)
            hh = tf.matmul(h, W_hh)

            bn_xh = batch_norm(xh, self._is_training)
            bn_hh = batch_norm(hh, self._is_training)

            hidden = bn_xh + bn_hh + bias

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = array_ops.split(value=hidden, num_or_size_splits=4, axis=1)

            new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
            bn_new_c = batch_norm(new_c, 'c', self._is_training)
            new_h = self._activation(bn_new_c) * sigmoid(o)
            new_state = rnn.LSTMStateTuple(new_c, new_h)

            return new_h, new_state


def orthogonal(shape):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    return q.reshape(shape)


def bn_lstm_identity_initializer(scale):

    def _initializer(shape, dtype=tf.float32, partition_info=None):
        """
        Ugly cause LSTM params calculated in one matrix multiply
        """

        size = shape[0]
        # gate (j) is identity
        t = np.zeros(shape)
        t[:, size:size * 2] = np.identity(size) * scale
        t[:, :size] = orthogonal([size, size])
        t[:, size * 2:size * 3] = orthogonal([size, size])
        t[:, size * 3:] = orthogonal([size, size])
        return tf.constant(t, dtype=dtype)

    return _initializer


def orthogonal_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        return tf.constant(orthogonal(shape), dtype)
    return _initializer


class TextPARNN(object):
    """A PARNN for text classification."""

    def __init__(
            self, pair_size, sequence_length, vocab_size, lstm_hidden_size, fc_hidden_size, attention_type,
            embedding_size, embedding_type, l2_reg_lambda=0.0, pretrained_embedding=None):

        # Placeholders for input, output, dropout_prob and training_tag
        self.input_x_content = tf.placeholder(tf.int32, [pair_size, None, sequence_length[0]], name="input_x_content")
        self.input_x_question = tf.placeholder(tf.int32, [pair_size, None, sequence_length[1]], name="input_x_question")
        self.input_x_option = tf.placeholder(tf.int32, [pair_size, None, sequence_length[2]], name="input_x_option")
        self.input_y = tf.placeholder(tf.float32, [pair_size, None, 1], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")

        def _embedding_layer(input_x_content, input_x_question, input_x_option, embedding_type, pretrained_embedding):
            """
            Embedding Layer.
            Args:
                input_x_content: [batch_size, sequence_length[0]]
                input_x_question: [batch_size, sequence_length[1]]
                input_x_option: [batch_size, sequence_length[2]]
                embedding_type: The embedding type
                pretrained_embedding: None or [vocab_size, embedding_size]

            Returns:
                embedded_sentence_content: [batch_size, sequence_length[0], embedding_size]
                embedded_sentence_question: [batch_size, sequence_length[1], embedding_size]
                embedded_sentence_option: [batch_size, sequence_length[2], embedding_size]
            """
            if pretrained_embedding is None:
                embedding = tf.get_variable("embedding", shape=[vocab_size, embedding_size],
                                            initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0),
                                            trainable=True)
            else:
                if embedding_type == 0:
                    embedding = tf.constant(pretrained_embedding, dtype=tf.float32, name="embedding")
                if embedding_type == 1:
                    embedding = tf.get_variable("embedding", shape=[vocab_size, embedding_size],
                                                initializer=tf.constant_initializer(pretrained_embedding),
                                                trainable=True)

            # [batch_size, sequence_length, embedding_size]
            embedded_sentence_content = tf.nn.embedding_lookup(embedding, input_x_content)
            embedded_sentence_question = tf.nn.embedding_lookup(embedding, input_x_question)
            embedded_sentence_option = tf.nn.embedding_lookup(embedding, input_x_option)

            return embedded_sentence_content, embedded_sentence_question, embedded_sentence_option

        def _bi_lstm_layer(input_x, name=""):
            # Bi-LSTM Layer
            with tf.variable_scope(name + "Bi_lstm", reuse=tf.AUTO_REUSE):
                lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size)  # forward direction cell
                lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size)  # backward direction cell
                if self.dropout_keep_prob is not None:
                    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
                    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)

                # Creates a dynamic bidirectional recurrent neural network
                # shape of `outputs`: tuple -> (outputs_fw, outputs_bw)
                # shape of `outputs_fw`: [batch_size, sequence_length, lstm_hidden_size]

                # shape of `state`: tuple -> (outputs_state_fw, output_state_bw)
                # shape of `outputs_state_fw`: tuple -> (c, h) c: memory cell; h: hidden state
                outputs, state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, input_x, dtype=tf.float32)

            # Concat output
            # [batch_size, sequence_length, lstm_hidden_size * 2]
            lstm_out = tf.concat(outputs, axis=2, name=name + "lstm_out")

            # [batch_size, lstm_hidden_size * 2]
            lstm_avg = tf.reduce_mean(lstm_out, axis=1, name=name + "lstm_avg")

            return lstm_out, lstm_avg

        def _attention(input_x, input_y, name=""):
            """
            Attention Layer.
            Args:
                input_x: [batch_size, sequence_length_1, lstm_hidden_size * 2]
                input_y: [batch_size, sequence_length_2, lstm_hidden_size * 2]
                name: Scope name
            Returns:
                attention_matrix: [batch_size, sequence_length_2, sequence_length_1]
                attention_weight: [batch_size, num_classes, sequence_length]
                attention_visual: [batch_size, sequence_length_1]
                attention_out: [batch_size, lstm_hidden_size * 2]
            """
            if attention_type == 'normal':
                with tf.variable_scope(name + "attention"):
                    attention_matrix = tf.matmul(input_y, tf.transpose(input_x, perm=[0, 2, 1]))
                    attention_weight = tf.nn.softmax(attention_matrix, name="attention_matrix")
                    attention_visual = tf.reduce_mean(attention_matrix, axis=1, name="visual")
                    attention_out = tf.matmul(attention_weight, input_x)
                    attention_out = tf.reduce_mean(attention_out, axis=1)
            if attention_type == 'cosine':
                with tf.variable_scope(name + "cos_attention"):
                    cos_matrix = []
                    seq_len = input_y.get_shape().as_list()[-2]
                    normalize_x = tf.nn.l2_normalize(input_x, 2)
                    y = tf.unstack(input_y, axis=1)
                    for i in range(seq_len):
                        normalize_y = tf.nn.l2_normalize(tf.expand_dims(y[i], axis=1), 2)
                        cos_similarity = tf.reduce_sum(tf.multiply(normalize_y, normalize_x), axis=2)
                        cos_matrix.append(cos_similarity)
                    attention_matrix = tf.stack(cos_matrix, axis=1, name="attention_matrix")
                    attention_visual = tf.reduce_mean(attention_matrix, axis=1, name="visual")
                    attention_out = tf.multiply(tf.expand_dims(attention_visual, axis=-1), input_x)
                    attention_out = tf.reduce_mean(attention_out, axis=1)
            if attention_type == 'mlp':
                with tf.variable_scope(name + "mlp_attention", reuse=tf.AUTO_REUSE):
                    x = tf.concat([input_x, input_y], axis=1)
                    num_units = x.get_shape().as_list()[-1]
                    W = tf.get_variable("W", shape=[num_units, num_units],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
                    b = tf.get_variable("b", shape=[num_units], initializer=tf.constant_initializer(value=0.1))
                    u = tf.get_variable("u", shape=[num_units, 1],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
                    attention_matrix = tf.map_fn(
                        fn=lambda x: tf.matmul(x, u),
                        elems=tf.tanh(
                            tf.map_fn(
                                fn=lambda x: tf.nn.xw_plus_b(x, W, b),
                                elems=x,
                                dtype=tf.float32
                            )
                        )
                    )
                    attention_weight = tf.nn.softmax(attention_matrix, name="attention_matrix")
                    attention_visual = tf.reduce_mean(attention_matrix, name="visual")
                    attention_out = tf.multiply(attention_weight, x)
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
            with tf.variable_scope(name + "fc", reuse=tf.AUTO_REUSE):
                num_units = input_x.get_shape().as_list()[-1]
                W = tf.get_variable("W", shape=[num_units, fc_hidden_size],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
                b = tf.get_variable("b", shape=[fc_hidden_size], initializer=tf.constant_initializer(value=0.1))
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

        def _sub_network(input_x_content, input_x_question, input_x_option, dropout_keep_prob, train_flag):
            # Embedding Layer
            with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
                embedded_sentence_content, embedded_sentence_question, embedded_sentence_option = \
                    _embedding_layer(input_x_content, input_x_question, input_x_option,
                                     embedding_type, pretrained_embedding)

            # Bi-lstm Layer
            lstm_out_content, lstm_avg_content = _bi_lstm_layer(embedded_sentence_content, name="content_")
            lstm_out_question, lstm_avg_question = _bi_lstm_layer(embedded_sentence_question, name="question_")
            lstm_out_option, lstm_avg_option = _bi_lstm_layer(embedded_sentence_option, name="option_")

            # Attention Layer
            visual_cq, attention_cq = _attention(lstm_out_content, lstm_out_question, name="cq_")
            visual_oq, attention_oq = _attention(lstm_out_option, lstm_out_question, name="oq_")

            # attention_out: [batch_size, hidden_size * 2 * 3]
            attention_out = tf.concat([attention_cq, lstm_avg_question, attention_oq], axis=1)

            # Fully Connected Layer
            fc_out = _fc_layer(attention_out)

            # Highway Layer
            with tf.variable_scope("highway", reuse=tf.AUTO_REUSE):
                highway = _highway_layer(fc_out, fc_out.get_shape()[1], num_layers=1, bias=0)
                h_drop = tf.nn.dropout(highway, dropout_keep_prob)

            # Final scores
            with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
                W = tf.get_variable("W", shape=[fc_hidden_size, 1],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
                b = tf.get_variable("b", shape=[1], initializer=tf.constant_initializer(value=0.1))
                logits = tf.nn.xw_plus_b(h_drop, W, b, name="logits")
                scores = tf.sigmoid(logits, name="scores")

            return visual_cq, visual_oq, scores

        self.visual_cq = []
        self.visual_oq = []
        self.scores_outputs = []

        for index in range(pair_size):
            visual_cq, visual_oq, scores = _sub_network(
                tf.unstack(self.input_x_content, axis=0)[index],
                tf.unstack(self.input_x_question, axis=0)[index],
                tf.unstack(self.input_x_option, axis=0)[index],
                self.dropout_keep_prob, self.is_training)
            self.visual_cq.append(visual_cq)
            self.visual_oq.append(visual_oq)
            self.scores_outputs.append(scores)

        print(self.scores_outputs)

        # Calculate mean cross-entropy loss, L2 loss
        with tf.name_scope("loss"):
            front_loss = self.scores_outputs[0] - tf.unstack(self.input_y, axis=0)[0]
            behind_loss = self.scores_outputs[1] - tf.unstack(self.input_y, axis=0)[1]
            losses = tf.reduce_mean(tf.square(front_loss - behind_loss), name="losses")
            l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()],
                                 name="l2_losses") * l2_reg_lambda
            self.loss = tf.add(losses, l2_losses, name="loss")

        for v in tf.trainable_variables():
            print(v.name)
