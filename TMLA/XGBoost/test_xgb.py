# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import sys
import time

sys.path.append('../')

import xgboost as xgb
from utils import data_process as dp
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error, r2_score

logger = dp.logger_fn("xgb-log", "xgb/test-{0}.log".format(time.asctime()))

# Data Parameters
TEST_DIR = '../../data/Test_BOW_sample.json'
MODEL_DIR = 'xgb_model.m'


def test():
    logger.info("Loading data...")

    x_test, y_test = dp.load_data(TEST_DIR)
    d_test = xgb.DMatrix(x_test, label=y_test)

    logger.info("Loading model...")
    model = joblib.load(MODEL_DIR)

    logger.info("Predicting...")
    y_pred = model.predict(d_test)

    logger.info("Calculate Metrics...")
    pcc, doa = dp.evaluation(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    logger.info("XGBoost: PCC {0:.4f} | DOA {1:.4f} | RMSE {2:.4f} | R2 {3:.4f}".format(pcc, doa, rmse, r2))

    logger.info("All Done.")


if __name__ == '__main__':
    test()
