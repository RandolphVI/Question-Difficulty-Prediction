import random
from tqdm import tqdm


def random_pick(input_file, train_file, val_file, test_file, val_num, test_num):
    def get_lines_num(input_file):
        with open(input_file, 'r') as f:
            lines = f.readlines()
            return len(lines)

    total_line = get_lines_num(input_file)
    total_list = [i for i in range(total_line)]

    print('First Step Done.')

    with open(input_file, 'r') as fin, open(train_file, 'w') as f_train, \
            open(val_file, 'w') as f_val, open(test_file, 'w') as f_test:
        test_list = random.sample(total_list, test_num)
        val_list = random.sample(list(set(total_list) - set(test_list)), val_num)
        for index, eachline in tqdm(enumerate(fin)):
            if index not in val_list and index not in test_list:
                f_train.write(eachline)
            if index in val_list:
                f_val.write(eachline)
            if index in test_list:
                f_test.write(eachline)


random_pick('data.json', 'Train.json', 'Validation.json', 'Test.json', val_num=500, test_num=6000)