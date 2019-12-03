# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import sys
import time
import data_process as dp
import tensorflow as tf
from sklearn.svm import SVR
from sklearn.externals import joblib

logger = dp.logger_fn("svm-log", "svm/train-{0}.log".format(time.asctime()))

TRAININGSET_DIR = '../data/Train_BOW.json'
MODEL_DIR = 'svm_model.m'

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
    logger.info("✔︎ Loading data...")

    x_train, y_train = dp.load_data(FLAGS.training_data_file)

    logger.info("✔︎ Finish building BOW.")

    model = SVR()

    logger.info("✔︎ Training model...")
    model.fit(x_train, y_train)

    logger.info("✔︎ Finish training. Saving model...")
    joblib.dump(model, FLAGS.model_file)


if __name__ == '__main__':
    train()
