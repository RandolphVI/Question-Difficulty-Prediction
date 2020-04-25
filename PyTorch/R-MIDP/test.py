# -*- coding:utf-8 -*-
__author__ = 'randolph'

import os
import sys
import time
import torch

sys.path.append('../')

from layers import RMIDP, Loss
from utils import checkmate as cm
from utils import data_helpers as dh
from utils import param_parser as parser
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score


args = parser.parameter_parser()
MODEL = dh.get_model_name()
logger = dh.logger_fn("ptlog", "logs/Test-{0}.log".format(time.asctime()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


CPT_DIR = os.path.abspath(os.path.join(os.path.curdir, "runs", MODEL))
SAVE_DIR = os.path.abspath(os.path.join(os.path.curdir, "outputs", MODEL))


def create_input_data(record):
    """
    Creating features and targets with Torch tensors.
    """
    x_f_content, x_b_content, x_f_question, x_b_question, x_f_option, x_b_option, y_f, y_b = record
    x_fb_content = (x_f_content.to(device), x_b_content.to(device))
    x_fb_question = (x_f_question.to(device), x_b_question.to(device))
    x_fb_option = (x_f_option.to(device), x_b_option.to(device))
    y_fb = (y_f.to(device), y_b.to(device))
    return x_fb_content, x_fb_question, x_fb_option, y_fb


def test():
    logger.info("Loading Data...")
    logger.info("Data processing...")
    test_data = dh.load_data_and_labels(args.test_file, args.word2vec_file)
    logger.info("Data padding...")
    x_test_fb_content, x_test_fb_question, x_test_fb_option, y_test_fb = dh.pad_data(test_data, args.pad_seq_len)

    test_dataset = TensorDataset(x_test_fb_content[0], x_test_fb_content[1],
                                 x_test_fb_question[0], x_test_fb_question[1],
                                 x_test_fb_option[0], x_test_fb_option[1],
                                 y_test_fb[0], y_test_fb[1])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    VOCAB_SIZE, EMBEDDING_SIZE, pretrained_word2vec_matrix = dh.load_word2vec_matrix(args.word2vec_file)

    criterion = Loss()
    model = RMIDP(args, VOCAB_SIZE, EMBEDDING_SIZE, pretrained_word2vec_matrix).to(device)
    checkpoint_file = cm.get_best_checkpoint(CPT_DIR, select_maximum_value=False)
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info("Scoring...")
    true_labels, predicted_scores = [], []
    for batch in test_loader:
        x_test_fb_content, x_test_fb_question, x_test_fb_option, y_test_fb = create_input_data(batch)
        logits, scores = model(x_test_fb_content, x_test_fb_question, x_test_fb_option)
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
    dh.create_prediction_file(save_dir=SAVE_DIR, identifiers=test_data.f_id, predictions=predicted_scores)

    logger.info('All Finished.')


if __name__ == "__main__":
    test()

