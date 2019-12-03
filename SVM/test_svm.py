# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import sys
import time
import data_process as dp
import tensorflow as tf
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error

logger = dp.logger_fn("svm-log", "svm/test-{0}.log".format(time.asctime()))

TEST_DIR = '../data/Test_BOW.json'
MODEL_DIR = 'svm_model.m'

# Data Parameters
tf.flags.DEFINE_string("test_data_file", TEST_DIR, "Data source for the test data.")
tf.flags.DEFINE_string("model_file", MODEL_DIR, "Model file.")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
dilim = '-' * 100
logger.info('\n'.join([dilim, *['{0:>50}|{1:<50}'.format(attr.upper(), FLAGS.__getattr__(attr))
                                for attr in sorted(FLAGS.__dict__['__wrapped'])], dilim]))


def test():
    logger.info("✔︎ Loading data...")

    x_test, y_test = dp.load_data(FLAGS.test_data_file)

    logger.info("✔︎ Loading model...")
    model = joblib.load(FLAGS.model_file)

    logger.info("✔︎ Predicting...")
    y_pred = model.predict(x_test)

    logger.info("✔︎ Calculate Metrics...")
    pcc, doa = dp.evaluation(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    logger.info("☛ Logistic: PCC {0:g}, DOA {1:g}, RSME {2:g}".format(pcc, doa, rmse))

    logger.info("✔︎ Done.")


if __name__ == '__main__':
    test()
