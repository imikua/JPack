import numpy as np

def get_discrete_deli_dataset():
    item_size_set = [
                        [24.5, 35.5, 39],
                        [27.5, 35.5, 20],
                        [39, 52.5, 29.5],
                        [32.55, 42.68, 31.28],
                        [26.5, 35, 19.5],
                        [23.5, 33.4, 20.7],
                        [30, 32.5, 16.5],
                        [27, 31.5, 27.5],
                        [23, 62.5, 32.5],
                        [44, 46, 31],
                        [23, 28, 25],
                        [32.5, 47, 11.5],
                        [30.5, 38, 27.5],
                        [27.5, 39.5, 15],
                        [30, 31.5, 35.5],
                        [41, 76.5, 50],
                        [35.5, 64.5, 30],
                        [23.1, 31.3, 20.5],
                        [32, 47, 32],
                        [53, 76, 36],
                        [40.5, 44.5, 57.5],
                        [44, 79, 34.5],
                        [30, 39, 55],
                        [48.5, 78, 52],
                        [32, 78.5, 48.5],
                        [34.3, 42.5, 39.5],
                        [23.5, 33.4, 20.7],
                        [51.8, 54, 53.2],
                        [47, 49.8, 40],
                        [40, 43.5, 55.7],
                        [36, 47, 26.3],
                        [60, 72.5, 32.5],
                        [60, 61, 30],
                        [23.5, 24.5, 30.5],
                        [44.5, 63, 27.5]]
    item_size_set = np.array(item_size_set) * 2
    item_size_set = item_size_set.astype('int')
    return item_size_set


def get_deli_prob_dataset():
    item_size_set = [
        (24.5, 35.5, 39, 3371),
        (27.5, 35.5, 20, 2309),
        (39, 52.5, 29.5, 2021),
        (32.55, 42.68, 31.28, 1721),
        (26.5, 35, 19.5, 1098),
        (23.5, 33.4, 20.7, 890),
        (30, 32.5, 16.5, 841),
        (27, 31.5, 27.5, 806),
        (23, 62.5, 32.5, 609),
        (44, 46, 31, 511),
        (23, 28, 25, 435),
        (32.5, 47, 11.5, 420),
        (30.5, 38, 27.5, 391),
        (27.5, 39.5, 15, 363),
        (30, 31.5, 35.5, 346),
        (41, 76.5, 50, 346),
        (35.5, 64.5, 30, 326),
        (23.1, 31.3, 20.5, 314),
        (32, 47, 32, 285),
        (53, 76, 36, 266),
        (40.5, 44.5, 57.5, 256),
        (44, 79, 34.5, 243),
        (30, 39, 55, 242),
        (48.5, 78, 52, 222),
        (32, 78.5, 48.5, 213),
        (34.3, 42.5, 39.5, 203),
        (23.5, 33.4, 20.7, 188),
        (51.8, 54, 53.2, 187),
        (47, 49.8, 40, 181),
        (40, 43.5, 55.7, 176),
        (36, 47, 26.3, 159),
        (60, 72.5, 32.5, 143),
        (60, 61, 30, 110)
    ]
    item_size_set = np.array(item_size_set)
    item_size_set[:, 0:3] = np.ceil(item_size_set[:, 0:3])
    item_size_set[:, -1] = item_size_set[:, -1] / item_size_set[:, -1].sum()
    item_size_set = item_size_set.tolist()
    return item_size_set

def parse_txt_to_list(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # 去除换行符并分割每一行的数字
            values = line.strip().split()
            # 将每个字符串的数字转换为浮点数或整数
            row = [float(value) if '.' in value else int(value) for value in values]
            data.append(row)
    return data

def get_opai_dataset():
    file_path = './datasets/opai.txt'
    data_list = parse_txt_to_list(file_path)
    data_list = np.array(data_list)
    data_list = np.ceil(data_list).astype('int')
    data_list = data_list.tolist()
    new_data_list = []
    for data in data_list:
        if max(data) > 200:
            continue
        new_data_list.append(data)
    print(new_data_list)
    return new_data_list

def get_real_opai_dataset():
    file_path = './datasets/opai.txt'
    data_list = parse_txt_to_list(file_path)
    data_list = np.array(data_list)
    data_list = np.ceil(data_list).astype('int')
    data_list = data_list.tolist()
    new_data_list = []

    for data in data_list:
        if max(data) > 250:
            continue
        new_data_list.append(data)

    return new_data_list

def get_discrete_dataset():
    lower = 1
    higher = 5
    resolution = 1
    item_size_set = []
    for i in range(lower, higher + 1):
        for j in range(lower, higher + 1):
            for k in range(lower, higher + 1):
                item_size_set.append((i * resolution, j * resolution, k * resolution))

    return item_size_set

# get_real_opai_dataset()