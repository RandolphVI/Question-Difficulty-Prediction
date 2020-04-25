# -*- coding:utf-8 -*-
__author__ = 'randolph'

import os
import time
import shutil
import torch
import torch.optim
import torch.nn as nn

# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from text_rmidp import TextRMIDP
from PyTorch.utils import data_helpers as dh

# Parameters
# Data Parameters
training_data_file = '../../data/train_sample.json'
validation_data_file = '../../data/validation_sample.json'

# Hyper parameters
learning_rate = 0.001
pad_seq_len_list = "350, 150, 10"
embedding_dim = 300
embedding_type = 1
lstm_hidden_size = 256
fc_hidden_size = 512
dropout_keep_prob = 0.5
l2_reg_lambda = 0

# Training Parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 256
num_epochs = 20
evaluate_every = 1
checkpoint_every = 1
best_rmse = 0.0


def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.1)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')


def print_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            print("weight", m.weight.data)
            print("bias:", m.bias.data)
            print("next...")


def save_checkpoint(state, is_best, filename):
    checkpoint_dir = os.path.dirname(filename)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.MSELoss = nn.MSELoss(reduce=True, size_average=True)

    def forward(self, predict_y, input_y):
        # Loss
        loss = self.MSELoss(predict_y, input_y)
        return loss


def train_rmidp():
    global best_rmse
    train_or_restore = input("Train or Restore?(T/R): ")
    while not (train_or_restore.isalpha() and train_or_restore.upper() in ['T', 'R']):
        train_or_restore = input("[ Warning ] The format of your input is illegal, please re-input: ")
    train_or_restore = train_or_restore.upper()
    if train_or_restore == 'T':
        logger = dh.logger_fn("training", "log/training-{0}.log".format(str(int(time.time()))))
    else:
        logger = dh.logger_fn("training", "log/restore-{0}.log".format(str(int(time.time()))))

    # Load sentences, labels, and training parameters
    logger.info("Loading Data...")
    train_data = dh.load_data_and_labels(training_data_file, embedding_dim, data_aug_flag=False)
    val_data = dh.load_data_and_labels(validation_data_file, embedding_dim, data_aug_flag=False)

    logger.info("Data padding...")
    x_train_content, x_train_question, x_train_option, y_train = dh.pad_data(train_data, pad_seq_len_list)
    x_val_content, x_val_question, x_val_option, y_val = dh.pad_data(val_data, pad_seq_len_list)

    train_dataset = TensorDataset(x_train_content, x_train_question, x_train_option, y_train)
    val_dataset = TensorDataset(x_val_content, x_val_question, x_val_option, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load word2vec model
    vocab_size, pretrained_word2vec_matrix = dh.load_word2vec_matrix(embedding_dim)

    # Init network
    logger.info("Init nn...")
    net = TextRMIDP(
        seq_len_list=list(map(int, pad_seq_len_list.split(','))),
        vocab_size=vocab_size,
        lstm_hidden_size=lstm_hidden_size,
        fc_hidden_size=fc_hidden_size,
        embedding_size=embedding_dim,
        embedding_type=embedding_type,
        pretrained_embedding=pretrained_word2vec_matrix,
        dropout_keep_prob=dropout_keep_prob).to(device)

    # weights_init(model=net)
    # print_weight(model=net)

    print("Model's state_dict:")
    for param_tensor in net.state_dict():
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    criterion = Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=l2_reg_lambda)
    if train_or_restore == 'R':
        model = input("Please input the checkpoints model you want to restore: ")
        while not (model.isdigit() and len(model) == 10):
            model = input("[ Warning ] The format of your input is illegal, please re-input: ")
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", model))
        logger.info("Writing to {0}\n".format(out_dir))
        checkpoint = torch.load(out_dir)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        net.train()
        best_rmse = checkpoint['best_rmse'].to(device)
    else:
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        logger.info("Writing to {0}\n".format(out_dir))

    logger.info("Training...")
    # writer = SummaryWriter('summary')
    step_cnt = 0
    for epoch in range(num_epochs):
        for x_train_content, x_train_question, x_train_option, y_train in train_loader:
            x_train_content, x_train_question, x_train_option, y_train = \
                [i.to(device) for i in [x_train_content, x_train_question, x_train_option, y_train]]
            # TODO
            optimizer.zero_grad()  # 如果不置零，Variable 的梯度在每次 backward 的时候都会累加
            logits, scores = net(x_train_content, x_train_question, x_train_option)
            print(scores)
            loss = criterion(scores, y_train)
            loss.backward()

            # Parameter updating
            optimizer.step()
            step_cnt += 1
            logger.info('[epoch %d, step %5d] loss: %.3f' % (epoch + 1, step_cnt, loss))
            # TODO
            # writer.add_scalar('training loss', train_loss / train_cnt, epoch * len(train_dataset) + train_cnt)
        if epoch % evaluate_every == 0:
            val_loss = 0.0
            val_batch_cnt = 0
            true_labels = []
            predicted_scores = []
            for x_val_content, x_val_question, x_val_option, y_val in val_loader:
                x_val_content, x_val_question, x_val_option, y_val = \
                    [i.to(device) for i in [x_val_content, x_val_question, x_val_option, y_val]]
                logits, scores = net(x_val_content, x_val_question, x_val_option)
                loss = criterion(scores,  y_val)
                val_loss += loss
                val_batch_cnt += 1
                for i in y_val.tolist():
                    true_labels.append(i)
                for j in scores.tolist():
                    predicted_scores.append(j)

            # Calculate the Metrics
            eval_rmse = mean_squared_error(true_labels, predicted_scores) ** 0.5
            eval_r2 = r2_score(true_labels, predicted_scores)
            eval_pcc, eval_doa = dh.evaluation(true_labels, predicted_scores)
            logger.info("best rmse: {0}".format(best_rmse))
            is_best = eval_rmse > best_rmse
            best_rmse = max(eval_rmse, best_rmse)

            logger.info("All Validation set: Loss {0:g} | PCC {1:g} | DOA {2:g} | RMSE {3:g} | R2 {4:g}"
                        .format(val_loss / val_batch_cnt, eval_pcc, eval_doa, eval_rmse, eval_r2))
            # writer.add_scalar('validation loss', val_loss / val_cnt, epoch)
            # writer.add_scalar('validation AUC', eval_auc, epoch)
            # writer.add_scalar('validation AUPRC', eval_prc, epoch)

        if epoch % checkpoint_every == 0:
            filename = os.path.abspath(os.path.join(out_dir, "epoch{0}.pth".format(epoch)))
            save_checkpoint({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_rmse': best_rmse,
            }, is_best, filename=filename)
    # writer.add_graph(net, x_train)
    # writer.close()

    logger.info('Finished Training.')


if __name__ == "__main__":
    train_rmidp()

