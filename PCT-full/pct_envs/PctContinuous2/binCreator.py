import numpy as np
import copy
import pickle
from scipy.stats import truncnorm, norm

class BoxCreator(object):
    def __init__(self):
        self.box_list = []

    def reset(self):
        self.box_list.clear()
        return self.change_distribution()

    def generate_box_size(self, **kwargs):
        pass

    def preview(self, length):
        while len(self.box_list) < length:
            self.generate_box_size()
        return copy.deepcopy(self.box_list[:length])

    def drop_box(self, idx=0):
        assert len(self.box_list) >= 0
        self.box_list.pop(idx)

    def change_distribution(self):
        return 0

class RandomBoxCreator(BoxCreator):
    default_box_set = []
    for i in range(5):
        for j in range(5):
            for k in range(5):
                default_box_set.append((2+i, 2+j, 2+k))

    def __init__(self, box_size_set=None):
        super().__init__()
        self.box_set = box_size_set
        print(self.box_set)
        if self.box_set is None:
            self.box_set = RandomBoxCreator.default_box_set

    def generate_box_size(self, **kwargs):
        idx = np.random.randint(0, len(self.box_set))
        self.box_list.append(self.box_set[idx])

class ContinuousBoxCreator(BoxCreator):
    def __init__(self, setting, sample_left_bound, sample_right_bound, normal=False, normal_mean = 0.5, normal_std = 0.5):
        super().__init__()
        self.setting = setting
        self.sample_left_bound = sample_left_bound
        self.sample_right_bound = sample_right_bound
        self.normal = normal
        self.normal_mean = normal_mean
        self.normal_std = normal_std

    def generate_box_size(self,  **kwargs):
        if self.setting == 2:
            if self.normal:
                next_box = (round(truncated_normal(self.normal_mean, self.normal_std, self.sample_left_bound, self.sample_right_bound)[0],3),
                            round(truncated_normal(self.normal_mean, self.normal_std, self.sample_left_bound, self.sample_right_bound)[0],3),
                            round(truncated_normal(self.normal_mean, self.normal_std, self.sample_left_bound, self.sample_right_bound)[0],3))
            else:
                next_box = (round(np.random.uniform(self.sample_left_bound, self.sample_right_bound), 3),
                            round(np.random.uniform(self.sample_left_bound, self.sample_right_bound), 3),
                            round(np.random.uniform(self.sample_left_bound, self.sample_right_bound), 3))
        else:
            if self.normal:
                next_box = (round(truncated_normal(self.normal_mean, self.normal_std, self.sample_left_bound, self.sample_right_bound)[0],3),
                            round(truncated_normal(self.normal_mean, self.normal_std, self.sample_left_bound, self.sample_right_bound)[0],3),
                            round(truncated_normal(self.normal_mean, self.normal_std, self.sample_left_bound, self.sample_right_bound)[0],3)
                            # np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5])
                            )
            else:
                next_box = (round(np.random.uniform(self.sample_left_bound, self.sample_right_bound), 3),
                        round(np.random.uniform(self.sample_left_bound, self.sample_right_bound), 3),
                        round(np.random.uniform(self.sample_left_bound, self.sample_right_bound), 3)
                        # np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5])
                            )  # may create bug here
        self.box_list.append(next_box)


def pdf(x, means, std_devs, weights = None):
    if weights is None:
        weights = np.ones(len(means)) / len(means)
    return np.mean([weights[i] * norm.pdf(x, means[i], std_devs[i]) for i in range(len(means))], axis=0)


def rejection_sampling(means, std_devs, sample_left_bound, sample_right_bound, weights = None):
    # 定义均匀分布的上界（大于概率密度函数的上界）
    upper_bound = 0.6

    # 进行拒绝采样
    while True:
        # 从均匀分布中随机采样
        # x_sample = np.random.uniform(0.001, 1)
        x_sample = np.random.uniform(sample_left_bound, sample_right_bound)

        # 从均匀分布中采样一个上界
        y_sample = np.random.uniform(0, upper_bound)

        # 如果采样点在概率密度函数的范围内，接受采样
        if y_sample < pdf(x_sample, means, std_devs, weights):
            return x_sample

class MixContinuousBoxCreator(BoxCreator):
    def __init__(self, setting, sample_left_bound, sample_right_bound):
        super().__init__()
        self.setting = setting

        self.deviation = 0.2
        # self.means = np.array([0.2, 0.5, 0.8])
        # self.std_devs = np.array([self.deviation, self.deviation, self.deviation])
        self.means    = np.array([0.1, 0.3, 0.5])
        self.std_devs = np.array([0.2, 0.1, 0.2])
        self.sample_left_bound  = sample_left_bound
        self.sample_right_bound = sample_right_bound
        self.num_distributions = int(2 ** len(self.means))
        self.change_distribution()

    def generate_box_size(self,  **kwargs):
        if self.setting == 2:
            next_box = (rejection_sampling(self.this_means, self.this_std_devs, self.sample_left_bound, self.sample_right_bound),
                        rejection_sampling(self.this_means, self.this_std_devs, self.sample_left_bound, self.sample_right_bound),
                        rejection_sampling(self.this_means, self.this_std_devs, self.sample_left_bound, self.sample_right_bound))
        else:
            next_box = (rejection_sampling(self.this_means, self.this_std_devs, self.sample_left_bound, self.sample_right_bound),
                        rejection_sampling(self.this_means, self.this_std_devs, self.sample_left_bound, self.sample_right_bound),
                        rejection_sampling(self.this_means, self.this_std_devs, self.sample_left_bound, self.sample_right_bound)
                        # np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5])
            )
        next_box = (round(next_box[0], 3), round(next_box[1], 3), round(next_box[2], 3))
        self.box_list.append(next_box)

    def change_distribution(self):
        while True:
            binary_index = np.random.randint(0, 2, len(self.means))
            real_index = np.where(binary_index == 1)[0]
            # todo we start from simple mixture distribution, and change this part later
            if np.sum(binary_index) != 1:
                continue
            if len(real_index) != 0:
                self.this_means = self.means[real_index]
                self.this_std_devs = self.std_devs[real_index]
                decimal_number = int(''.join(map(str, binary_index)), 2)
                return decimal_number - 1

class MixLengthBoxCreator(BoxCreator):
    def __init__(self, setting, add_distribution = False, add_interval = 1e5):
        super().__init__()
        self.setting = setting
        self.add_distribution = add_distribution
        self.step = 0
        self.add_interval = add_interval
        # self.sample_left_bounds =  [0.05, 0.075, 0.1]
        # self.sample_right_bounds = [0.5,  0.75,  1.0]
        self.sample_left_bounds =  [0.001, 0.001, 0.001]
        self.sample_right_bounds = [0.4, 0.5, 0.6]
        self.change_distribution()

    def generate_box_size(self,  **kwargs):
        self.step += 1
        if self.setting == 2:
            next_box = (round(np.random.uniform(self.sample_left_bound, self.sample_right_bound), 3),
                        round(np.random.uniform(self.sample_left_bound, self.sample_right_bound), 3),
                        round(np.random.uniform(self.sample_left_bound, self.sample_right_bound), 3))
        else:
            next_box = (round(np.random.uniform(self.sample_left_bound, self.sample_right_bound), 3),
                        round(np.random.uniform(self.sample_left_bound, self.sample_right_bound), 3),
                        round(np.random.uniform(self.sample_left_bound, self.sample_right_bound), 3)
                        # np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5])
                        )  # may create bug here
        self.box_list.append(next_box)

    def change_distribution(self):
        length = len(self.sample_left_bounds) if not self.add_distribution else min(int(self.step//self.add_interval + 1), len(self.sample_left_bounds))
        sample_left_bounds = self.sample_left_bounds[0:length]
        sample_right_bounds = self.sample_right_bounds[0:length]
        binary_index = np.random.randint(0, len(sample_left_bounds))
        self.sample_left_bound  = sample_left_bounds[binary_index]
        self.sample_right_bound = sample_right_bounds[binary_index]
        return binary_index


class LoadBoxCreator(BoxCreator):
    def __init__(self, data_name=None):
        super().__init__()
        self.data_name = data_name
        print(self.data_name)
        print("load data set successfully!")
        self.index = 0
        self.box_index = 0
        global box_trajs
        if self.data_name.endswith('.pt'):
            raise RuntimeError('PctContinuous2/LoadBoxCreator no longer supports torch .pt datasets in the pure Jittor environment. Please convert the dataset to .pkl/.npy/.npz first, or use the sampling-based continuous mode.')
        elif self.data_name.endswith('.pkl'):
            with open(self.data_name, 'rb') as f:
                box_trajs = pickle.load(f)
        elif self.data_name.endswith('.npy'):
            box_trajs = np.load(self.data_name, allow_pickle=True)
        elif self.data_name.endswith('.npz'):
            npz_file = np.load(self.data_name, allow_pickle=True)
            first_key = list(npz_file.keys())[0]
            box_trajs = npz_file[first_key]
        else:
            raise RuntimeError('Unsupported dataset format for PctContinuous2 LoadBoxCreator: {}'.format(self.data_name))
        self.traj_nums = len(box_trajs)

    def reset(self, index=None):
        self.box_list.clear()
        self.recorder = []
        if index is None:
            self.index += 1
        else:
            self.index = index
        self.boxes = np.array(box_trajs[self.index])
        self.boxes = self.boxes.tolist()
        self.box_index = 0
        self.box_set = self.boxes
        self.box_set.append([100, 100, 100])

    def generate_box_size(self, **kwargs):
        if self.box_index < len(self.box_set):
            self.box_list.append(self.box_set[self.box_index])
            self.recorder.append(self.box_set[self.box_index])
            self.box_index += 1
        else:
            self.box_list.append((10, 10, 10))
            self.recorder.append((10, 10, 10))
            self.box_index += 1



def truncated_normal(mean, std_dev, lower_bound, upper_bound, size=1):
    a = (lower_bound - mean) / std_dev
    b = (upper_bound - mean) / std_dev
    samples = truncnorm.rvs(a, b, loc=mean, scale=std_dev, size=size).astype(float)
    return samples


class ICRABoxCreator(BoxCreator):
    def __init__(self):
        super().__init__()

    def generate_box_size(self,  **kwargs):
        next_box = (round(truncated_normal(0.45, 0.09, 0.1, 0.8)[0],3),
                    round(truncated_normal(0.3,  0.05, 0.1, 0.50)[0],3),
                    round(truncated_normal(0.17, 0.03, 0.001, 0.30)[0],3))
        self.box_list.append(next_box)