# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import sys
import time

sys.path.append('../')

import xgboost as xgb
from utils import data_process as dp
from sklearn.externals import joblib

logger = dp.logger_fn("xgb-log", "xgb/train-{0}.log".format(time.asctime()))

# Data Parameters
TRAININGSET_DIR = '../../data/Train_BOW_sample.json'
VALIDATION_DIR = '../../data/Validation_BOW_sample.json'
MODEL_DIR = 'xgb_model.m'


def train():
    # Load data
    logger.info("Loading data...")

    x_train, y_train = dp.load_data(TRAININGSET_DIR)
    x_val, y_val = dp.load_data(VALIDATION_DIR)

    d_train = xgb.DMatrix(x_train, label=y_train)
    d_val = xgb.DMatrix(x_val, label=y_val)
    watchlist = [(d_train, 'train'), (d_val, 'valid')]
    logger.info("Finish building BOW.")

    params_xgb = {
        'objective': 'reg:linear',
        'eta': 0.001,
        'max_depth': 10,
        'eval_metric': 'rmse'
    }
    # TODO
    model = xgb.train(params_xgb, d_train, 10000, evals=watchlist, early_stopping_rounds=20, verbose_eval=10)
    logger.info("Training model...")

    logger.info("Finish training. Saving model...")
    joblib.dump(model, MODEL_DIR)


if __name__ == '__main__':
    train()
