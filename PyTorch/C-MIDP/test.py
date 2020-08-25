# -*- coding:utf-8 -*-
__author__ = 'randolph'

import os
import sys
import time
import torch

sys.path.append('../')

from layers import CMIDP, Loss
from utils import checkmate as cm
from utils import data_helpers as dh
from utils import param_parser as parser
from tqdm import trange
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score

args = parser.parameter_parser()
MODEL = dh.get_model_name()
logger = dh.logger_fn("ptlog", "logs/Test-{0}.log".format(time.asctime()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CPT_DIR = os.path.abspath(os.path.join(os.path.curdir, "runs", MODEL))
SAVE_DIR = os.path.abspath(os.path.join(os.path.curdir, "outputs", MODEL))


def test():
    logger.info("Loading Data...")
    logger.info("Data processing...")
    test_data = dh.load_data_and_labels(args.test_file, args.word2vec_file)
    logger.info("Data padding...")
    test_dataset = dh.MyData(test_data, args.pad_seq_len, device)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    VOCAB_SIZE, EMBEDDING_SIZE, pretrained_word2vec_matrix = dh.load_word2vec_matrix(args.word2vec_file)

    criterion = Loss()
    net = CMIDP(args, VOCAB_SIZE, EMBEDDING_SIZE, pretrained_word2vec_matrix).to(device)
    checkpoint_file = cm.get_best_checkpoint(CPT_DIR, select_maximum_value=False)
    checkpoint = torch.load(checkpoint_file)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    logger.info("Scoring...")
    true_labels, predicted_scores = [], []
    batches = trange(len(test_loader), desc="Batches", leave=True)
    for batch_cnt, batch in zip(batches, test_loader):
        x_test_fb_content, x_test_fb_question, x_test_fb_option, \
        x_test_fb_clens, x_test_fb_qlens, x_test_fb_olens, y_test_fb = batch
        logits, scores = net(x_test_fb_content, x_test_fb_question, x_test_fb_option)
        for i in y_test_fb[0].tolist():
            true_labels.append(i)
        for j in scores[0].tolist():
            predicted_scores.append(j)

    # Calculate the Metrics
    test_rmse = mean_squared_error(true_labels, predicted_scores) ** 0.5
    test_r2 = r2_score(true_labels, predicted_scores)
    test_pcc, test_doa = dh.evaluation(true_labels, predicted_scores)
    logger.info("All Test set: PCC {0:.4f} | DOA {1:.4f} | RMSE {2:.4f} | R2 {3:.4f}"
                .format(test_pcc, test_doa, test_rmse, test_r2))
    logger.info('Test Finished.')

    logger.info('Creating the prediction file...')
    dh.create_prediction_file(save_dir=SAVE_DIR, identifiers=test_data['f_id'], predictions=predicted_scores)

    logger.info('All Finished.')


if __name__ == "__main__":
    test()

