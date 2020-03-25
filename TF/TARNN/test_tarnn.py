# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import os
import sys
import time
import logging

sys.path.append('../')
logging.getLogger('tensorflow').disabled = True

import tensorflow as tf
from utils import checkmate as cm
from utils import data_helpers as dh
from param_parser import parameter_parser
from sklearn.metrics import mean_squared_error, r2_score

args = parameter_parser()
MODEL = dh.get_model_name()
logger = dh.logger_fn("tflog", "logs/Test-{0}.log".format(time.asctime()))

CPT_DIR = 'runs/' + MODEL + '/checkpoints/'
BEST_CPT_DIR = 'runs/' + MODEL + '/bestcheckpoints/'
SAVE_DIR = 'output/' + MODEL


def test_tarnn():
    """Test TARNN model."""
    # Print parameters used for the model
    dh.tab_printer(args, logger)

    # Load data
    logger.info("Loading data...")
    logger.info("Data processing...")
    test_data = dh.load_data_and_labels(args.test_file, args.word2vec_file, data_aug_flag=False)

    logger.info("Data padding...")
    x_test_content, x_test_question, x_test_option, y_test = dh.pad_data(test_data, args.pad_seq_len)

    # Load tarnn model
    OPTION = dh._option(pattern=1)
    if OPTION == 'B':
        logger.info("Loading best model...")
        checkpoint_file = cm.get_best_checkpoint(BEST_CPT_DIR, select_maximum_value=True)
    else:
        logger.info("Loading latest model...")
        checkpoint_file = tf.train.latest_checkpoint(CPT_DIR)
    logger.info(checkpoint_file)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=args.allow_soft_placement,
            log_device_placement=args.log_device_placement)
        session_conf.gpu_options.allow_growth = args.gpu_options_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{0}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x_content = graph.get_operation_by_name("input_x_content").outputs[0]
            input_x_question = graph.get_operation_by_name("input_x_question").outputs[0]
            input_x_option = graph.get_operation_by_name("input_x_option").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]

            # Tensors we want to evaluate
            scores = graph.get_operation_by_name("output/scores").outputs[0]
            loss = graph.get_operation_by_name("loss/loss").outputs[0]

            # Split the output nodes name by '|' if you have several output nodes
            output_node_names = "output/scores"

            # Save the .pb model file
            output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                            output_node_names.split("|"))
            tf.train.write_graph(output_graph_def, "graph", "graph-tarnn-{0}.pb".format(MODEL), as_text=False)

            # Generate batches for one epoch
            batches = dh.batch_iter(list(zip(x_test_content, x_test_question, x_test_option, y_test)),
                                    args.batch_size, 1, shuffle=False)

            test_counter, test_loss = 0, 0.0

            # Collect the predictions here
            true_labels = []
            predicted_scores = []

            for batch_test in batches:
                x_batch_content, x_batch_question, x_batch_option, y_batch = zip(*batch_test)
                feed_dict = {
                    input_x_content: x_batch_content,
                    input_x_question: x_batch_question,
                    input_x_option: x_batch_option,
                    input_y: y_batch,
                    dropout_keep_prob: 1.0,
                    is_training: False
                }
                batch_scores, cur_loss = sess.run([scores, loss], feed_dict)

                # Prepare for calculating metrics
                for i in y_batch:
                    true_labels.append(i)
                for j in batch_scores:
                    predicted_scores.append(j)

                test_loss = test_loss + cur_loss
                test_counter = test_counter + 1

            # Calculate PCC & DOA
            pcc, doa = dh.evaluation(true_labels, predicted_scores)
            # Calculate RMSE
            rmse = mean_squared_error(true_labels, predicted_scores) ** 0.5
            r2 = r2_score(true_labels, predicted_scores)

            test_loss = float(test_loss / test_counter)

            logger.info("All Test Dataset: Loss {0:g} | PCC {1:g} | DOA {2:g} | RMSE {3:g} | R2 {4:g}"
                        .format(test_loss, pcc, doa, rmse, r2))

            # Save the prediction result
            if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)
            dh.create_prediction_file(output_file=SAVE_DIR + "/predictions.json", all_id=test_data.id,
                                      all_labels=true_labels, all_predict_scores=predicted_scores)

    logger.info("All Done.")


if __name__ == '__main__':
    test_tarnn()
