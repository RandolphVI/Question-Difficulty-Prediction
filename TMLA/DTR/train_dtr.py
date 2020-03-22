# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import sys
import time
import tensorflow as tf

from TMLA.utils import data_process as dp
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals import joblib

logger = dp.logger_fn("dtr-log", "dtr/train-{0}.log".format(time.asctime()))

TRAININGSET_DIR = '../../data/Train_BOW_sample.json'
MODEL_DIR = 'dtr_model.m'

# Data Parameters
tf.flags.DEFINE_string("training_data_file", TRAININGSET_DIR, "Data source for the training data.")
tf.flags.DEFINE_string("model_file", MODEL_DIR, "Model file.")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
dilim = '-' * 100
logger.info('\n'.join([dilim, *['{0:>50}|{1:<50}'.format(attr.upper(), FLAGS.__getattr__(attr))
                                for attr in sorted(FLAGS.__dict__['__wrapped'])], dilim]))


def train():
    # Load data
    logger.info("Loading data...")

    x_train, y_train = dp.load_data(FLAGS.training_data_file)

    logger.info("Finish building BOW.")

    model = DecisionTreeRegressor(criterion="mse", splitter="best")

    logger.info("Training model...")
    model.fit(x_train, y_train)

    logger.info("Finish training. Saving model...")
    joblib.dump(model, FLAGS.model_file)


if __name__ == '__main__':
    train()
