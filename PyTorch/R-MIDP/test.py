# -*- coding:utf-8 -*-
__author__ = 'randolph'

import os
import time
import torch
import torch.optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
from text_tarnn import TextTARNN
from train_tarnn import Loss
from PyTorch.utils import data_helpers as dh

# Parameters
best_or_latest = input("☛ Load Best or Latest Model?(B/L): ")
while not (best_or_latest.isalpha() and best_or_latest.upper() in ['B', 'L']):
    best_or_latest = input("✘ The format of your input is illegal, please re-input: ")
best_or_latest = best_or_latest.upper()

# Data Parameters
test_data_file = '../../data/test_sample.json'

# Hyper parameters
pad_seq_len_list = "350, 150, 10"
att_type = 'normal'  # ['normal', 'cosine', 'mlp']
embedding_dim = 300
embedding_type = 1
lstm_hidden_size = 256
attention_unit_size = 200
attention_penalization = True
fc_hidden_size = 1024
dropout_keep_prob = 0.5
l2_reg_lambda = 0
threshold = 0.5

# Test Parameters
batch_size = 1


def test_harnn():
    print("Loading Data")
    test_data = dh.load_data_and_labels(test_data_file, embedding_dim, data_aug_flag=False)
    x_test, y_test, y_test_0, y_test_1, y_test_2, y_test_3 = dh.pad_data(test_data, pad_seq_len_list)
    test_dataset = TensorDataset(x_test, y_test, y_test_0, y_test_1, y_test_2, y_test_3)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    vocab_size, pretrained_word2vec_matrix = dh.load_word2vec_matrix(embedding_dim)

    print("Init nn")
    net = TextTARNN(
        seq_len_list=list(map(int, pad_seq_len_list.split(','))),
        vocab_size=vocab_size,
        lstm_hidden_size=lstm_hidden_size,
        fc_hidden_size=fc_hidden_size,
        att_type=att_type,
        embedding_size=embedding_dim,
        embedding_type=embedding_type,
        pretrained_embedding=pretrained_word2vec_matrix,
        dropout_keep_prob=dropout_keep_prob).to(device)
    criterion = Loss()
    if best_or_latest == 'L':
        models = os.listdir("./model")
        models.sort(key=lambda x: x[x.find('.'):])
        out_dir = "./model/" + models[-1]
    else:
        out_dir = "./model_best.pth"
    checkpoint = torch.load(out_dir)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    print("Testing")
    # Collection
    test_counter, test_loss = 0, 0.0
    true_labels = []
    predicted_labels = []
    predicted_scores = []

    # Collect for calculating metrics
    true_onehot_labels = []
    predicted_onehot_scores = []
    predicted_onehot_labels_ts = []
    predicted_onehot_labels_tk = [[] for _ in range(top_num)]
    for x_test, y_test, y_test_0, y_test_1, y_test_2, y_test_3 in test_loader:
        (scores, first_attention, first_visual, second_visual, third_visual, fourth_visual), outputs = net(x_test)
        test_loss += criterion(outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6],
                               y_test_0, y_test_1, y_test_2, y_test_3, y_test)
        test_counter += 1
        f = open('attention.html', 'w')
        f.write('<html style="margin:0;padding:0;"><body style="margin:0;padding:0;">\n')
        f.write('<div style="margin:25px;">\n')
        for k in range(len(first_attention[0])):
            f.write('<p style="margin:10px;">\n')
            for i in range(len(first_attention[0][0])):
                attention = "{:.2f}".format(first_attention[0][k][i])
                word = x_test[0][i]
                f.write(f'\t<span style="margin-left:3px;background-color:rgba(255,0,0,{attention})">{word}</span>\n')
            f.write('</p>\n')
        f.write('</div>\n')
        f.write('</body></html>')
        f.close()

        # Prepare for calculating metrics
        for onehot_labels in y_test:
            true_onehot_labels.append(onehot_labels.tolist())
        for onehot_scores in scores:
            predicted_onehot_scores.append(onehot_scores.tolist())

        # Get the predicted labels by threshold
        batch_predicted_labels_ts, batch_predicted_scores_ts = \
            dh.get_label_threshold(scores=scores.detach().numpy(), threshold=threshold)

        # Add results to collection
        # for labels in y_batch_test_labels:
        #     true_labels.append(labels)
        for labels in batch_predicted_labels_ts:
            predicted_labels.append(labels)
        for values in batch_predicted_scores_ts:
            predicted_scores.append(values)

        # Get one-hot prediction by threshold
        batch_predicted_onehot_labels_ts = \
            dh.get_onehot_label_threshold(scores=scores.detach().numpy(), threshold=threshold)

        for onehot_labels in batch_predicted_onehot_labels_ts:
            predicted_onehot_labels_ts.append(onehot_labels)

        # Get one-hot prediction by topK
        for i in range(top_num):
            batch_predicted_onehot_labels_tk = dh.get_onehot_label_topk(scores=scores.detach().numpy(), top_num=i+1)
            for onehot_labels in batch_predicted_onehot_labels_tk:
                predicted_onehot_labels_tk[i].append(onehot_labels)

    # Calculate Precision & Recall & F1
    test_pre_ts = precision_score(y_true=np.array(true_onehot_labels),
                                  y_pred=np.array(predicted_onehot_labels_ts), average='micro')
    test_rec_ts = recall_score(y_true=np.array(true_onehot_labels),
                               y_pred=np.array(predicted_onehot_labels_ts), average='micro')

    test_F_ts = f1_score(y_true=np.array(true_onehot_labels),
                         y_pred=np.array(predicted_onehot_labels_ts), average='micro')

    # Calculate the average AUC
    test_auc = roc_auc_score(y_true=np.array(true_onehot_labels),
                             y_score=np.array(predicted_onehot_scores), average='micro')

    # Calculate the average PR
    test_prc = average_precision_score(y_true=np.array(true_onehot_labels),
                                       y_score=np.array(predicted_onehot_scores), average="micro")

    test_loss = float(test_loss / test_counter)

    print("☛ All Test Dataset: Loss {0:g} | AUC {1:g} | AUPRC {2:g}" .format(test_loss, test_auc, test_prc))
    print("☛ Predict by threshold: Precision {0:g}, Recall {1:g}, F1 {2:g}" .format(test_pre_ts, test_rec_ts, test_F_ts))
    print('Finished Test')


if __name__ == "__main__":
    test_harnn()

