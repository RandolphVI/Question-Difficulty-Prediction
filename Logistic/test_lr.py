# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import sys
import time
import data_process as dp
import tensorflow as tf
from sklearn.externals import joblib

logger = dp.logger_fn("logistic-log", "logistic/test-{0}.log".format(time.asctime()))

TRAININGSET_DIR = '../data/Train_BOW.json'
TEST_DIR = '../data/Test_BOW.json'
MODEL_DIR = 'logistic_model.m'

# Data Parameters
tf.flags.DEFINE_string("training_data_file", TRAININGSET_DIR, "Data source for the training data.")
tf.flags.DEFINE_string("test_data_file", TEST_DIR, "Data source for the test data.")
tf.flags.DEFINE_string("model_file", MODEL_DIR, "Model file.")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
dilim = '-' * 100
logger.info('\n'.join([dilim, *['{0:>50}|{1:<50}'.format(attr.upper(), FLAGS.__getattr__(attr))
                                for attr in sorted(FLAGS.__dict__['__wrapped'])], dilim]))


def test():
    logger.info("✔︎ Loading data...")

    x_train, y_train, x_test, y_test = dp.get_data_logistic(FLAGS.training_data_file, FLAGS.test_data_file)

    logger.info("✔︎ Loading model...")
    model = joblib.load(FLAGS.model_file)

    logger.info("✔︎ Predicting...")
    y_pred = model.predict(x_test)

    logger.info("✔︎ Calculate Metrics...")
    pcc, doa = dp.evaluation(y_test, y_pred)

    logger.info("☛ Logistic: PCC {0:g}, DOA {1:g}".format(pcc, doa))

    logger.info("✔︎ Done.")


if __name__ == '__main__':
    test()
