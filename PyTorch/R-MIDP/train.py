# -*- coding:utf-8 -*-
__author__ = 'randolph'

import os
import sys
import time
import torch
import torch.nn as nn

sys.path.append('../')

from layers import RMIDP, Loss
from utils import checkmate as cm
from utils import data_helpers as dh
from utils import param_parser as parser
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score


args = parser.parameter_parser()
OPTION = dh.option()
logger = dh.logger_fn("ptlog", "logs/{0}-{1}.log".format('Train' if OPTION == 'T' else 'Restore', time.asctime()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def train():
    """Training RMIDP model."""
    dh.tab_printer(args, logger)

    # Load sentences, labels, and training parameters
    logger.info("Loading data...")
    logger.info("Data processing...")
    train_data = dh.load_data_and_labels(args.train_file, args.word2vec_file)
    val_data = dh.load_data_and_labels(args.validation_file, args.word2vec_file)

    logger.info("Data padding...")
    x_train_fb_content, x_train_fb_question, x_train_fb_option, y_train_fb = dh.pad_data(train_data, args.pad_seq_len)
    x_val_fb_content, x_val_fb_question, x_val_fb_option, y_val_fb = dh.pad_data(val_data, args.pad_seq_len)

    train_dataset = TensorDataset(x_train_fb_content[0], x_train_fb_content[1],
                                  x_train_fb_question[0], x_train_fb_question[1],
                                  x_train_fb_option[0], x_train_fb_option[1],
                                  y_train_fb[0], y_train_fb[1])
    val_dataset = TensorDataset(x_val_fb_content[0], x_val_fb_content[1],
                                x_val_fb_question[0], x_val_fb_question[1],
                                x_val_fb_option[0], x_val_fb_option[1],
                                y_val_fb[0], y_val_fb[1])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Load word2vec model
    VOCAB_SIZE, EMBEDDING_SIZE, pretrained_word2vec_matrix = dh.load_word2vec_matrix(args.word2vec_file)

    # Init network
    logger.info("Init nn...")
    net = RMIDP(args, VOCAB_SIZE, EMBEDDING_SIZE, pretrained_word2vec_matrix).to(device)

    print("Model's state_dict:")
    for param_tensor in net.state_dict():
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    criterion = Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.l2_lambda)

    if OPTION == 'T':
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        saver = cm.BestCheckpointSaver(save_dir=out_dir, num_to_keep=args.num_checkpoints, maximize=False)
        logger.info("Writing to {0}\n".format(out_dir))
    elif OPTION == 'R':
        timestamp = input("[Input] Please input the checkpoints model you want to restore: ")
        while not (timestamp.isdigit() and len(timestamp) == 10):
            timestamp = input("[Warning] The format of your input is illegal, please re-input: ")
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        saver = cm.BestCheckpointSaver(save_dir=out_dir, num_to_keep=args.num_checkpoints, maximize=False)
        logger.info("Writing to {0}\n".format(out_dir))
        checkpoint = torch.load(out_dir)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    logger.info("Training...")
    writer = SummaryWriter('summary')

    def eval_model(val_loader, epoch):
        """
        Evaluate on the validation set.
        """
        net.eval()
        eval_loss = 0.0
        true_labels, predicted_scores = [], []
        for batch in val_loader:
            x_val_fb_content, x_val_fb_question, x_val_fb_option, y_val_fb = create_input_data(batch)
            logits, scores = net(x_val_fb_content, x_val_fb_question, x_val_fb_option)
            avg_batch_loss = criterion(scores, y_val_fb)
            eval_loss = eval_loss + avg_batch_loss.item()
            for i in y_val_fb[0].tolist():
                true_labels.append(i)
            for j in scores[0].tolist():
                predicted_scores.append(j)

        # Calculate the Metrics
        eval_rmse = mean_squared_error(true_labels, predicted_scores) ** 0.5
        eval_r2 = r2_score(true_labels, predicted_scores)
        eval_pcc, eval_doa = dh.evaluation(true_labels, predicted_scores)
        eval_loss = eval_loss / len(val_loader)
        cur_value = eval_rmse
        logger.info("All Validation set: Loss {0:g} | PCC {1:.4f} | DOA {2:.4f} | RMSE {3:.4f} | R2 {4:.4f}"
                    .format(eval_loss, eval_pcc, eval_doa, eval_rmse, eval_r2))
        writer.add_scalar('validation loss', eval_loss, epoch)
        writer.add_scalar('validation PCC', eval_pcc, epoch)
        writer.add_scalar('validation DOA', eval_doa, epoch)
        writer.add_scalar('validation RMSE', eval_rmse, epoch)
        writer.add_scalar('validation R2', eval_r2, epoch)
        return cur_value

    for epoch in tqdm(range(args.epochs), desc="Epochs:", leave=True):
        # Training step
        batches = trange(len(train_loader), desc="Batches", leave=True)
        for batch_cnt, batch in zip(batches, train_loader):
            net.train()
            x_train_fb_content, x_train_fb_question, x_train_fb_option, y_train_fb = create_input_data(batch)
            optimizer.zero_grad()   # 如果不置零，Variable 的梯度在每次 backward 的时候都会累加
            logits, scores = net(x_train_fb_content, x_train_fb_question, x_train_fb_option)
            avg_batch_loss = criterion(scores, y_train_fb)
            avg_batch_loss.backward()
            optimizer.step()    # Parameter updating
            batches.set_description("Batches (Loss={:.4f})".format(avg_batch_loss.item()))
            logger.info('[epoch {0}, batch {1}] loss: {2:.4f}'.format(epoch + 1, batch_cnt, avg_batch_loss.item()))
            writer.add_scalar('training loss', avg_batch_loss, batch_cnt)
        # Evaluation step
        cur_value = eval_model(val_loader, epoch)
        saver.handle(cur_value, net, optimizer, epoch)
    writer.close()

    logger.info('Training Finished.')


if __name__ == "__main__":
    train()

