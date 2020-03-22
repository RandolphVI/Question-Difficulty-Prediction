from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
import numpy as np
from bintrees import FastRBTree
import json
import csv
import math
from scipy import stats


# 词袋模型作为特征
def extract_feature():
    print('extracting feature...')
    # 获取词典
    dict_rbtree = FastRBTree()
    word_num = 0
    with open('exercises/exercise.doc.modified3', 'r', encoding='utf8') as input_file:
        line = input_file.readline()
        while line != '':
            line_split = line.split()
            for seq in line_split[1:]:
                for char in seq:
                    index = dict_rbtree.get(char)
                    if index is None:
                        dict_rbtree.insert(char, word_num)
                        word_num += 1
            line = input_file.readline()
    print('word_num=', word_num)
    # 抽取词袋特征
    with open('exercises/exercise.doc.modified3', 'r', encoding='utf8') as input_file, open('exercises/baseline_feature.csv', 'w', encoding='utf8', newline='') as output_file:
        csv_output = csv.writer(output_file)
        line = input_file.readline()
        while line != '':
            line_split = line.split()
            exercise_id = line_split[0]
            feature = [0] * word_num
            for seq in line_split[1:]:
                for char in seq:
                    feature[dict_rbtree[char]] += 1
            # feature_3 = 0
            # for i in [1, -1]:
            #     for char in line_split[i]:
            #         if ('\u4e00' > char) or (char > '\u9fff'):
            #             feature_3 += 1
            csv_output.writerow([exercise_id] + feature)
            line = input_file.readline()
    print('finished')


# 准备训练和测试数据，logistic()和random_forest()使用
def prepare_data():
    print('preparing data...')
    # load exercise features
    feature_rbtree = FastRBTree()
    with open('exercises/baseline_feature.csv', 'r', encoding='utf8') as input_file:
        csv_input = csv.reader(input_file)
        for row in csv_input:
            # print(row)
            feature_rbtree.insert(row[0], row[1:])
    # load score rates
    score_rate_rbtree = FastRBTree()
    with open('records/group_by_exercise', 'r', encoding='utf8') as input_file:
        exercises = json.load(input_file)
    for exercise in exercises:
        score_rate_rbtree.insert(exercise['exercise_id'], exercise['avg_score_rate'])
    # divide data
    with open('records/group_by_mission', 'r', encoding='utf8') as input_file:
        missions = json.load(input_file)
    # 先把所有试题得分率相同的mission去掉，与文章模型的实验保持一致
    missions_copy = missions[:]
    for mission in missions_copy:
        error_mission = True
        score_rate = mission['exercises'][0]['avg_score_rate']
        for exercise in mission['exercises']:
            if exercise['avg_score_rate'] != score_rate:
                error_mission = False
                break
        if error_mission:
            missions.remove(mission)
    test_mission_nums = [500, 1000, 1500, 2000]
    for num in test_mission_nums:
        train_fname = 'aggregate/baseline_train_' + str(num / 5000) + '.csv'
        test_fname = 'aggregate/baseline_test_' + str(num / 5000)
        train_rbtree, test_mission_list = FastRBTree(), []
        for mission in missions[:-num]:
            for exercise in mission['exercises']:
                exercise_id = exercise['exercise_id']
                feature = feature_rbtree.get(exercise_id)
                if feature is not None:
                    train_rbtree.insert(exercise_id, [exercise_id] + feature + [score_rate_rbtree.get(exercise_id)])
        for mission in missions[-num:]:
            exer_i = 0
            while exer_i < len(mission['exercises']):
                exercise_id = mission['exercises'][exer_i]['exercise_id']
                feature = feature_rbtree.get(exercise_id)
                if (train_rbtree.get(exercise_id) is not None) or (feature is None):
                    del mission['exercises'][exer_i]
                    continue
                mission['exercises'][exer_i]['feature'] = feature
                exer_i += 1
            if len(mission['exercises']) > 0:
                test_mission_list.append(mission)
        # write data
        with open(train_fname, 'w', encoding='utf8', newline='') as output_file:
            csv_output = csv.writer(output_file)
            for exercise in train_rbtree.items():
                csv_output.writerow(exercise[1])
        with open(test_fname, 'w', encoding='utf8') as output_file:
            json.dump(test_mission_list, output_file, ensure_ascii=False, indent=4)
    print('finished')


# 整合试题特征和考试得分率，logistic_mission()和random_forest_mission()使用
def aggregate_data():
    print('aggregating data...')
    # load exercise features
    feature_rbtree = FastRBTree()
    with open('exercises/baseline_feature.csv', 'r', encoding='utf8') as input_file:
        csv_input = csv.reader(input_file)
        for row in csv_input:
            # print(row)
            feature_rbtree.insert(row[0], row[1:])
    # load missions，mission顺序要与mission_feature_data_modified1一致
    with open('aggregate/mission_feature_data_modified1', 'r', encoding='utf8') as input_file:
        missions = json.load(input_file)
    # aggregate
    aggregated_data = []
    total_exercise_num = 0
    for mission in missions:
        if len(mission['exercises']) < 1:
            continue
        # 先检查是不是所有题的avg_score_rate都相同（有这种情况），如果是，跳过该mission
        error_mission = True
        score_rate = float(mission['exercises'][0]['avg_score_rate'])
        for exer in mission['exercises']:
            if float(exer['avg_score_rate']) != score_rate:
                error_mission = False
                break
        if error_mission:
            continue

        aggregated_mission = {'paper_id': mission['paper_id'], 'school_id': mission['school_id'],
                              'test_date': mission['test_date']}
        aggregated_exercises = []
        for exercise in mission['exercises']:
            feature = feature_rbtree.get(exercise['exercise_id'])
            if feature is not None:
                aggregated_exercise = exercise
                aggregated_exercise['feature'] = feature
                aggregated_exercises.append(aggregated_exercise)
        aggregated_exercise_num = len(aggregated_exercises)
        aggregated_mission['exercise_num'] = aggregated_exercise_num
        aggregated_mission['exercises'] = aggregated_exercises
        aggregated_data.append(aggregated_mission)

        total_exercise_num += aggregated_exercise_num

    with open('aggregate/baseline_data', 'w', encoding='utf8') as output_file:
        json.dump(aggregated_data, fp=output_file, indent=4, ensure_ascii=False)
    print('total_exercise_num= %d, mission_num= %d' % (total_exercise_num, len(aggregated_data)))
    print('finished')


# 训练集中得分率是所有记录的平均得分率，测试集中每个mission中的得分率是该mission中的得分率（为了在评估时与文章模型保持一致）
def logistic():
    print('running logistic regression')
    test_ratios = [0.1, 0.2, 0.3, 0.4]
    for test_ratio in test_ratios:
        train_fname = 'aggregate/baseline_train_' + str(test_ratio) + '.csv'
        test_fname = 'aggregate/baseline_test_' + str(test_ratio)
        # load data
        train_x, train_y = [], []
        with open(train_fname, 'r', encoding='utf8') as input_file:
            csv_input = csv.reader(input_file)
            for row in csv_input:
                train_x.append(list(map(float, row[1:-1])))
                train_y.append(float(row[-1]))
        with open(test_fname, 'r', encoding='utf8') as input_file:
            test_missions = json.load(input_file)
        train_x = np.array(train_x)
        train_y = np.array(list(map(int, np.array(train_y) * 1000)))
        model = LogisticRegression()
        model.fit(train_x, train_y)
        # test and evaluation
        pcc_accu, doa_accu, mission_num = 0, 0, 0
        for mission in test_missions:
            test_x, test_y = [], []
            for exercise in mission['exercises']:
                test_x.append(list(map(float, exercise['feature'])))
                test_y.append(float(exercise['avg_score_rate']))
            if len(test_y) > 1:
                test_y = np.array(list(map(int, np.array(test_y) * 1000)))
                pred_y = model.predict(test_x)
                pcc, doa = evaluation(test_y, pred_y)
                if doa == -1:
                    continue
                mission_num += 1
                pcc_accu += pcc
                doa_accu += doa
        print('test ratio=%f, pcc=%f, doa=%f' % (test_ratio, pcc_accu / mission_num, doa_accu / mission_num))


# 与文章模型一样保留mission信息，每个mission中的试题得分率为该mission中
def logistic_mission():
    print('running logistic_mission()...')
    # load data
    with open('aggregate/baseline_data', 'r', encoding='utf8') as input_file:
        missions = json.load(input_file)
    # testset_sizes = [500, 1000, 1500, 2000]
    testset_sizes = [1500, 2000]
    for testset_size in testset_sizes:
        train_x, train_y = [], []
        train_exer_ids = FastRBTree()
        # divide data for training
        for mission in missions[:-testset_size]:
            for exercise in mission['exercises']:
                train_x.append(list(map(float, exercise['feature'])))
                train_y.append(float(exercise['avg_score_rate']))
                train_exer_ids.insert(exercise['exercise_id'], '')
        train_x = np.array(train_x)
        train_y = np.array(list(map(int, np.array(train_y) * 1000)))
        # train
        model = LogisticRegression()
        model.fit(train_x, train_y)
        joblib.dump(model, 'logistic_m_' + str(testset_size / 5000))
        # test
        # model = joblib.load('logistic_m_500')
        doa_accu, pcc_accu = 0, 0
        mission_num = 0
        for mission in missions[-testset_size:]:
            test_x, test_y = [], []
            for exercise in mission['exercises']:
                if train_exer_ids.get(exercise['exercise_id']) is None:
                    test_x.append(exercise['feature'])
                    test_y.append(exercise['avg_score_rate'])
            if len(test_y) < 2:
                continue
            test_x = np.array(test_x, dtype=np.uint32)
            test_y = np.array(list(map(int, np.array(test_y) * 1000)))
            pred_y = model.predict(test_x)
            pcc, doa = evaluation(test_y, pred_y)
            if not math.isnan(pcc):
                pcc_accu += pcc
                doa_accu += doa
                mission_num += 1
        print('test ratio=%f, pcc=%f, doa=%f' % (testset_size / 5000, pcc_accu / mission_num, doa_accu / mission_num))


# 训练集中得分率是所有记录的平均得分率，测试集中每个mission中的得分率是该mission中的得分率（为了在评估时与文章模型保持一致）
def random_forest():
    print('running random forest regression')
    test_ratios = [0.1, 0.2, 0.3, 0.4]
    for test_ratio in test_ratios:
        train_fname = 'aggregate/baseline_train_' + str(test_ratio) + '.csv'
        test_fname = 'aggregate/baseline_test_' + str(test_ratio)
        # load data
        train_x, train_y = [], []
        with open(train_fname, 'r', encoding='utf8') as input_file:
            csv_input = csv.reader(input_file)
            for row in csv_input:
                train_x.append(list(map(float, row[1:-1])))
                train_y.append(float(row[-1]))
        with open(test_fname, 'r', encoding='utf8') as input_file:
            test_missions = json.load(input_file)
        train_x = np.array(train_x)
        # train_y = np.array(list(map(int, np.array(train_y) * 1000)))
        train_y = np.array(train_y)
        model = RandomForestRegressor()
        model.fit(train_x, train_y)
        # test and evaluation
        pcc_accu, doa_accu, mission_num = 0, 0, 0
        for mission in test_missions:
            test_x, test_y = [], []
            for exercise in mission['exercises']:
                test_x.append(list(map(float, exercise['feature'])))
                test_y.append(float(exercise['avg_score_rate']))
            if len(test_y) > 1:
                # test_y = np.array(list(map(int, np.array(test_y) * 1000)))
                test_y = np.array(test_y)
                pred_y = model.predict(test_x)
                # print(test_y, pred_y)
                pcc, doa = evaluation(test_y, pred_y)
                if doa == -1:
                    continue
                mission_num += 1
                pcc_accu += pcc
                doa_accu += doa
        print('test ratio=%f, pcc=%f, doa=%f' % (test_ratio, pcc_accu / mission_num, doa_accu / mission_num))


def random_forest_mission():
    print('running random_forest_mission()...')
    # load data
    with open('aggregate/baseline_data', 'r', encoding='utf8') as input_file:
        missions = json.load(input_file)
    test_ratios = np.array([0.4, 0.3, 0.2, 0.1])
    testset_sizes = list(map(int, test_ratios * len(missions)))
    for testset_size in testset_sizes:
        train_x, train_y = [], []
        train_exer_ids = FastRBTree()
        # divide data for training
        for mission in missions[:-testset_size]:
            for exercise in mission['exercises']:
                train_x.append(list(map(float, exercise['feature'])))
                train_y.append(float(exercise['avg_score_rate']))
                train_exer_ids.insert(exercise['exercise_id'], '')
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        # train
        model = RandomForestRegressor()
        model.fit(train_x, train_y)
        joblib.dump(model, 'rf_m_' + str(testset_size / 5000))
        # test
        doa_accu, pcc_accu = 0, 0
        mission_num = 0
        for mission in missions[-testset_size:]:
            test_x, test_y = [], []
            for exercise in mission['exercises']:
                if train_exer_ids.get(exercise['exercise_id']) is None:
                    test_x.append(exercise['feature'])
                    test_y.append(exercise['avg_score_rate'])
            if len(test_y) < 2:
                continue
            test_x = np.array(test_x)
            test_y = np.array(test_y)
            pred_y = model.predict(test_x)
            pcc, doa = evaluation(test_y, pred_y)
            if not math.isnan(pcc):
                pcc_accu += pcc
                doa_accu += doa
                mission_num += 1
        print('test ratio=%f, pcc=%f, doa=%f' % (testset_size / 5000, pcc_accu / mission_num, doa_accu / mission_num))


def evaluation(test_y, pred_y):
    # compute pcc
    pcc, _ = stats.pearsonr(pred_y, test_y)
    if math.isnan(pcc):
        print('ERROR: PCC=nan', test_y, pred_y)
    # compute doa
    n = 0
    correct_num = 0
    for i in range(len(test_y) - 1):
        for j in range(i + 1, len(test_y)):
            if (test_y[i] > test_y[j]) and (pred_y[i] > pred_y[j]):
                correct_num += 1
            elif (test_y[i] == test_y[j]) and (pred_y[i] == pred_y[j]):
                continue
            elif (test_y[i] < test_y[j]) and (pred_y[i] < pred_y[j]):
                correct_num += 1
            n += 1
    if n == 0:
        print(test_y)
        return -1, -1
    doa = correct_num / n
    return pcc, doa


if __name__ == '__main__':
    # extract_feature()
    # prepare_data()
    # aggregate_data()
    # logistic()
    # random_forest()
    random_forest_mission()
    # logistic_mission()
