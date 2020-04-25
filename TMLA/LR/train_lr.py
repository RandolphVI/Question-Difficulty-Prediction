# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import sys
import time

sys.path.append('../')

from utils import data_process as dp
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib

logger = dp.logger_fn("lr-log", "lr/train-{0}.log".format(time.asctime()))

# Data Parameters
TRAININGSET_DIR = '../../data/Train_BOW_sample.json'
MODEL_DIR = 'lr_model.m'


def train():
    # Load data
    logger.info("Loading data...")

    x_train, y_train = dp.load_data(TRAININGSET_DIR)

    logger.info("Finish building BOW.")

    model = LinearRegression()

    logger.info("Training model...")
    model.fit(x_train, y_train)

    logger.info("Finish training. Saving model...")
    joblib.dump(model, MODEL_DIR)


if __name__ == '__main__':
    train()
