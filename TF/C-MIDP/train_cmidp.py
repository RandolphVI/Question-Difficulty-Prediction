# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import os
import sys
import time
import logging
import tensorflow as tf

from text_cmidp import TextCMIDP
from TF.utils import checkmate as cm
from TF.utils import data_helpers as dh
from tensorboard.plugins import projector
from sklearn.metrics import mean_squared_error, r2_score

# Parameters
# ==================================================

TRAIN_OR_RESTORE = input("[Input] Train or Restore? (T/R): ")

while not (TRAIN_OR_RESTORE.isalpha() and TRAIN_OR_RESTORE.upper() in ['T', 'R']):
    TRAIN_OR_RESTORE = input("[Error] The format of your input is illegal, please re-input: ")
logging.info("The format of your input is legal, now loading to next step...")

TRAIN_OR_RESTORE = TRAIN_OR_RESTORE.upper()

if TRAIN_OR_RESTORE == 'T':
    logger = dh.logger_fn("tflog", "logs/training-{0}.log".format(time.asctime()))
if TRAIN_OR_RESTORE == 'R':
    logger = dh.logger_fn("tflog", "logs/restore-{0}.log".format(time.asctime()))

TRAININGSET_DIR = '../../data/Train_sample.json'
VALIDATIONSET_DIR = '../../data/Validation_sample.json'
METADATA_DIR = '../../data/metadata.tsv'

# Data Parameters
tf.flags.DEFINE_string("training_data_file", TRAININGSET_DIR, "Data source for the training data.")
tf.flags.DEFINE_string("validation_data_file", VALIDATIONSET_DIR, "Data source for the validation data.")
tf.flags.DEFINE_string("metadata_file", METADATA_DIR, "Metadata file for embedding visualization"
                                                      "(Each line is a word segment in metadata_file).")
tf.flags.DEFINE_string("train_or_restore", TRAIN_OR_RESTORE, "Train or Restore.")

# Model Hyperparameters
tf.flags.DEFINE_float("learning_rate", 0.001, "The learning rate (default: 0.001)")
tf.flags.DEFINE_string("pad_seq_len", "350,15,10", "Recommended padding Sequence length of data (depends on the data)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("embedding_type", 1, "The embedding type (default: 1)")
tf.flags.DEFINE_integer("fc_hidden_size", 400, "Hidden size for fully connected layer (default: 1024)")
tf.flags.DEFINE_string("filter_sizes", "3,3", "Comma-separated filter sizes (default: '3')")
tf.flags.DEFINE_string("num_filters", "200,400", "Comma-separated number of filters per filter size (default: 128)")
tf.flags.DEFINE_integer("pooling_size", 3, "Pooling sizes (default: '3')")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training Parameters
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 256)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 100)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 5000)")
tf.flags.DEFINE_float("norm_ratio", 2, "The ratio of the sum of gradients norms of trainable variable (default: 1.25)")
tf.flags.DEFINE_integer("decay_steps", 5000, "how many steps before decay learning rate. (default: 500)")
tf.flags.DEFINE_float("decay_rate", 0.95, "Rate of decay for learning rate. (default: 0.95)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 1000)")
tf.flags.DEFINE_integer("num_checkpoints", 10, "Number of checkpoints to store (default: 50)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
dilim = '-' * 100
logger.info('\n'.join([dilim, *['{0:>50}|{1:<50}'.format(attr.upper(), FLAGS.__getattr__(attr))
                                for attr in sorted(FLAGS.__dict__['__wrapped'])], dilim]))


def train_cmidp():
    """Training cmdip model."""

    # Load sentences, labels, and training parameters
    logger.info("Loading data...")

    logger.info("Training data processing...")
    train_data = dh.load_data_and_labels(FLAGS.training_data_file, FLAGS.embedding_dim, data_aug_flag=False)

    logger.info("Validation data processing...")
    val_data = dh.load_data_and_labels(FLAGS.validation_data_file, FLAGS.embedding_dim, data_aug_flag=False)

    logger.info("Training data padding...")
    x_train_content, x_train_question, x_train_option, y_train = dh.pad_data(train_data, FLAGS.pad_seq_len)

    logger.info("Validation data padding...")
    x_val_content, x_val_question, x_val_option, y_val = dh.pad_data(val_data, FLAGS.pad_seq_len)

    # Build vocabulary
    VOCAB_SIZE, pretrained_word2vec_matrix = dh.load_word2vec_matrix(FLAGS.embedding_dim)

    # Build a graph and cmidp object
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cmidp = TextCMIDP(
                sequence_length=list(map(int, FLAGS.pad_seq_len.split(','))),
                vocab_size=VOCAB_SIZE,
                fc_hidden_size=FLAGS.fc_hidden_size,
                embedding_size=FLAGS.embedding_dim,
                embedding_type=FLAGS.embedding_type,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(','))),
                num_filters=list(map(int, FLAGS.num_filters.split(','))),
                pooling_size=FLAGS.pooling_size,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                pretrained_embedding=pretrained_word2vec_matrix)

            # Define training procedure
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                learning_rate = tf.train.exponential_decay(learning_rate=FLAGS.learning_rate,
                                                           global_step=cmidp.global_step, decay_steps=FLAGS.decay_steps,
                                                           decay_rate=FLAGS.decay_rate, staircase=True)
                optimizer = tf.train.AdamOptimizer(learning_rate)
                grads, vars = zip(*optimizer.compute_gradients(cmidp.loss))
                grads, _ = tf.clip_by_global_norm(grads, clip_norm=FLAGS.norm_ratio)
                train_op = optimizer.apply_gradients(zip(grads, vars), global_step=cmidp.global_step, name="train_op")

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in zip(grads, vars):
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{0}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{0}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            if FLAGS.train_or_restore == 'R':
                MODEL = input("[Input] Please input the checkpoints model you want to restore, "
                              "it should be like(1490175368): ")  # The model you want to restore

                while not (MODEL.isdigit() and len(MODEL) == 10):
                    MODEL = input("[Error] The format of your input is illegal, please re-input: ")
                logger.info("The format of your input is legal, now loading to next step...")
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", MODEL))
                logger.info("Writing to {0}\n".format(out_dir))
            else:
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                logger.info("Writing to {0}\n".format(out_dir))

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            best_checkpoint_dir = os.path.abspath(os.path.join(out_dir, "bestcheckpoints"))

            # Summaries for loss
            loss_summary = tf.summary.scalar("loss", cmidp.loss)

            # Train summaries
            train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Validation summaries
            validation_summary_op = tf.summary.merge([loss_summary])
            validation_summary_dir = os.path.join(out_dir, "summaries", "validation")
            validation_summary_writer = tf.summary.FileWriter(validation_summary_dir, sess.graph)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            best_saver = cm.BestCheckpointSaver(save_dir=best_checkpoint_dir, num_to_keep=3, maximize=False)

            if FLAGS.train_or_restore == 'R':
                # Load cmidp model
                logger.info("Loading model...")
                checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
                logger.info(checkpoint_file)

                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{0}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
            else:
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                # Embedding visualization config
                config = projector.ProjectorConfig()
                embedding_conf = config.embeddings.add()
                embedding_conf.tensor_name = "embedding"
                embedding_conf.metadata_path = FLAGS.metadata_file

                projector.visualize_embeddings(train_summary_writer, config)
                projector.visualize_embeddings(validation_summary_writer, config)

                # Save the embedding visualization
                saver.save(sess, os.path.join(out_dir, "embedding", "embedding.ckpt"))

            current_step = sess.run(cmidp.global_step)

            def train_step(x_batch_content, x_batch_question, x_batch_option, y_batch):
                """A single training step"""
                feed_dict = {
                    cmidp.input_x_content: x_batch_content,
                    cmidp.input_x_question: x_batch_question,
                    cmidp.input_x_option: x_batch_option,
                    cmidp.input_y: y_batch,
                    cmidp.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    cmidp.is_training: True
                }
                _, step, summaries, loss = sess.run(
                    [train_op, cmidp.global_step, train_summary_op, cmidp.loss], feed_dict)
                logger.info("step {0}: loss {1:g}".format(step, loss))
                train_summary_writer.add_summary(summaries, step)

            def validation_step(x_val_content, x_val_question, x_val_option, y_val, writer=None):
                """Evaluates model on a validation set"""
                batches_validation = dh.batch_iter(list(zip(x_val_content, x_val_question, x_val_option, y_val)),
                                                   FLAGS.batch_size, 1)

                eval_counter, eval_loss = 0, 0.0

                true_labels = []
                predicted_scores = []

                for batch_validation in batches_validation:
                    x_batch_content, x_batch_question, x_batch_option, y_batch = zip(*batch_validation)
                    feed_dict = {
                        cmidp.input_x_content: x_batch_content,
                        cmidp.input_x_question: x_batch_question,
                        cmidp.input_x_option: x_batch_option,
                        cmidp.input_y: y_batch,
                        cmidp.dropout_keep_prob: 1.0,
                        cmidp.is_training: False
                    }
                    step, summaries, scores, cur_loss = sess.run(
                        [cmidp.global_step, validation_summary_op, cmidp.scores, cmidp.loss], feed_dict)

                    # Prepare for calculating metrics
                    for i in y_batch:
                        true_labels.append(i)
                    for j in scores:
                        predicted_scores.append(j)

                    eval_loss = eval_loss + cur_loss
                    eval_counter = eval_counter + 1

                    if writer:
                        writer.add_summary(summaries, step)

                eval_loss = float(eval_loss / eval_counter)

                # Calculate PCC & DOA
                pcc, doa = dh.evaluation(true_labels, predicted_scores)
                # Calculate RMSE
                rmse = mean_squared_error(true_labels, predicted_scores) ** 0.5
                r2 = r2_score(true_labels, predicted_scores)

                return eval_loss, pcc, doa, rmse, r2

            # Generate batches
            batches_train = dh.batch_iter(list(zip(x_train_content, x_train_question, x_train_option, y_train)),
                                          FLAGS.batch_size, FLAGS.num_epochs)

            num_batches_per_epoch = int((len(y_train) - 1) / FLAGS.batch_size) + 1

            # Training loop. For each batch...
            for batch_train in batches_train:
                x_batch_train_content, x_batch_train_question, x_batch_train_option, y_batch_train = zip(*batch_train)
                train_step(x_batch_train_content, x_batch_train_question, x_batch_train_option, y_batch_train)
                current_step = tf.train.global_step(sess, cmidp.global_step)

                if current_step % FLAGS.evaluate_every == 0:
                    logger.info("\nEvaluation:")
                    eval_loss, pcc, doa, rmse, r2 = validation_step(x_val_content, x_val_question, x_val_option, y_val,
                                                                    writer=validation_summary_writer)
                    logger.info("All Validation set: Loss {0:g} | PCC {1:g} | DOA {2:g} | RMSE {3:g} | R2 {4:g}"
                                .format(eval_loss, pcc, doa, rmse, r2))
                    best_saver.handle(rmse, sess, current_step)
                if current_step % FLAGS.checkpoint_every == 0:
                    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    logger.info("Saved model checkpoint to {0}\n".format(path))
                if current_step % num_batches_per_epoch == 0:
                    current_epoch = current_step // num_batches_per_epoch
                    logger.info("Epoch {0} has finished!".format(current_epoch))

    logger.info("✔︎ Done.")


if __name__ == '__main__':
    train_cmidp()