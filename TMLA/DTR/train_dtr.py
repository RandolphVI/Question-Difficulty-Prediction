# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import time

from TMLA.utils import data_process as dp
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals import joblib

logger = dp.logger_fn("dtr-log", "dtr/train-{0}.log".format(time.asctime()))

# Data Parameters
TRAININGSET_DIR = '../../data/Train_BOW_sample.json'
MODEL_DIR = 'dtr_model.m'


def train():
    # Load data
    logger.info("Loading data...")

    x_train, y_train = dp.load_data(TRAININGSET_DIR)

    logger.info("Finish building BOW.")

    model = DecisionTreeRegressor(criterion="mse", splitter="best")

    logger.info("Training model...")
    model.fit(x_train, y_train)

    logger.info("Finish training. Saving model...")
    joblib.dump(model, MODEL_DIR)


if __name__ == '__main__':
    train()
