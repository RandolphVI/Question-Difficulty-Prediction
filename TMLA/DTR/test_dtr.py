# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import sys
import time
import tensorflow as tf

from TMLA.utils import data_process as dp
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error, r2_score

logger = dp.logger_fn("dtr-log", "dtr/test-{0}.log".format(time.asctime()))

TEST_DIR = '../../data/Test_BOW_sample.json'
MODEL_DIR = 'dtr_model.m'

# Data Parameters
tf.flags.DEFINE_string("test_data_file", TEST_DIR, "Data source for the test data.")
tf.flags.DEFINE_string("model_file", MODEL_DIR, "Model file.")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
dilim = '-' * 100
logger.info('\n'.join([dilim, *['{0:>50}|{1:<50}'.format(attr.upper(), FLAGS.__getattr__(attr))
                                for attr in sorted(FLAGS.__dict__['__wrapped'])], dilim]))


def test():
    logger.info("Loading data...")

    x_test, y_test = dp.load_data(FLAGS.test_data_file)

    logger.info("Loading model...")
    model = joblib.load(FLAGS.model_file)

    logger.info("Predicting...")
    y_pred = model.predict(x_test)

    logger.info("Calculate Metrics...")
    pcc, doa = dp.evaluation(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    logger.info("DTR: PCC {0:g} | DOA {1:g} | RMSE {2:g} | R2 {3:g}".format(pcc, doa, rmse, r2))

    logger.info("All Done.")


if __name__ == '__main__':
    test()
