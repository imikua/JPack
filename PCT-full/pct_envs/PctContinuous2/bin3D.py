import copy
import time

from .space import Space, tolerance, item_decimals
import numpy as np
import gym
from .binCreator import RandomBoxCreator, LoadBoxCreator, ContinuousBoxCreator, MixContinuousBoxCreator, MixLengthBoxCreator, ICRABoxCreator
import random
from copy import deepcopy

class PackingContinuous(gym.Env):
    def __init__(self,
                 args,
                 **kwags):

        self.setting = args.setting
        self.internal_node_holder = args.internal_node_holder
        self.leaf_node_holder = args.leaf_node_holder
        self.next_holder = 1
        self.item_set = args.item_size_set

        self.shuffle = args.shuffle
        self.bin_size = args.container_size

        if args.sample_from_distribution:
            self.size_minimum = args.sample_left_bound
            self.sample_left_bound = args.sample_left_bound
            self.sample_right_bound = args.sample_right_bound
            if args.distribution == 'uniform' or args.distribution == 'normal':
                self.box_creator = ContinuousBoxCreator(self.setting, self.sample_left_bound, self.sample_right_bound,
                                                    args.distribution == 'normal', args.normal_mean, args.normal_std)
            elif args.distribution == 'mixture':
                self.box_creator = MixContinuousBoxCreator(self.setting, self.sample_left_bound, self.sample_right_bound)
            elif args.distribution == 'icra':
                self.box_creator = ICRABoxCreator()
            else:
                assert args.distribution == 'mixlength'
                self.box_creator = MixLengthBoxCreator(self.setting, args.add_distribution)

        else:
            self.size_minimum = np.min(np.array(self.item_set))
            self.box_creator = RandomBoxCreator(self.item_set)

        if self.setting == 2: self.orientation = 6
        else: self.orientation = 2
        self.preview = args.preview
        self.select = args.select

        # The class that maintains the contents of the bin.
        box_bound = args.box_bound
        self.space = Space(*self.bin_size, self.size_minimum, self.internal_node_holder, box_bound=box_bound, args = args)

        if args.load_dataset:
            self.box_creator = LoadBoxCreator(args.dataset_path)

        self.test = args.load_dataset
        self.observation_space = gym.spaces.Box(low=0.0, high=self.space.height,
                                                shape=((self.internal_node_holder + self.leaf_node_holder + self.next_holder) * 9,))

        # Gym/VecEnv 需要 action_space 属性。
        # 本项目的 PCT 架构动作是“结构化/组合动作”，真实含义由上层策略网络解释，
        # 因此这里提供一个占位空间用于通过 Gym 的接口检查与 VecEnv 初始化。
        # 若后续需要严格定义动作语义，可根据 env.step(action) 的实际解析改为 MultiDiscrete/Box。
        self.action_space = gym.spaces.Discrete(1)

        self.next_box_vec = np.zeros((self.next_holder, 9))
        self.no_internal_node_input = args.no_internal_node_input
        # self.LNES = args.lnes
        self.LNES = 'EMS'  # Leaf Node Expansion Schemes: EMS
        self.large_scale   = args.large_scale
        self.height_reward = args.height_reward
        # self.leaf_without_maxz = args.leaf_without_maxz
        self.leaf_without_maxz = True
        self.drl_method = args.drl_method

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            self.SEED = seed
        return [seed]


    # Calculate space utilization inside a bin.
    def get_box_ratio(self):
        coming_box = self.next_box
        return (coming_box[0] * coming_box[1] * coming_box[2]) / (self.space.plain_size[0] * self.space.plain_size[1] * self.space.plain_size[2])

    def set_next_box(self, next_box):
        next_box = [round(x, 3) for x in next_box]
        return self.cur_observation(next_box = next_box)

    def shot_this_time_step(self):
        self.shot = copy.deepcopy(self)
        return True
    def recover_to_shot(self):
        self.__dict__.update(self.shot.__dict__)
        return True
    def print_next_box(self):
        print(self.next_box)
        return True

    def set_box_creator(self, params):
        normal_mean, normal_std = params
        self.box_creator = ContinuousBoxCreator(self.setting, self.sample_left_bound, self.sample_right_bound,
                                                True, normal_mean, normal_std)
        return True

    def reset(self):
        self.distribution_label = self.box_creator.reset() # todo this part should return from environment
        self.packed = []
        self.space.reset()
        self.box_creator.generate_box_size()
        cur_observation = self.cur_observation()
        self.maxXYZ = np.zeros(3)
        self.minXYZ = copy.deepcopy(self.bin_size)
        return cur_observation

    # Count and return all PCT nodes.
    def cur_observation(self, selected_box_idx=0, next_box=None):
        if next_box is not None:
            self.next_box = next_box
            self.next_den = 1
        else:
            self.next_box = np.array(self.gen_next_box(selected_box_idx))
            if self.test:
                if self.setting == 3: self.next_den = self.next_box[3]
                else: self.next_den = 1
                self.next_box = [round(x, 3) for x in self.next_box]
            else:
                if self.setting < 3: self.next_den = 1
                else:
                    self.next_den = np.random.random()
                    while self.next_den == 0:
                        self.next_den = np.random.random()

        self.next_box = self.next_box[0:3]
        box_vec = np.array(self.space.box_vec)
        leaf_nodes = self.get_possible_position()
        next_box = np.sort(self.next_box)
        self.next_box_vec[:, 3:6] = next_box
        self.next_box_vec[:, 0] = self.next_den
        self.next_box_vec[:, -1] = 1

        next_box_vec = deepcopy(self.next_box_vec) # change real size to 1

        if self.large_scale:
            container_size = self.space.plain_size
            box_vec[:, 0:6] = (box_vec[:, 0:6].reshape((-1, 3)) / container_size).reshape((-1, 6))
            leaf_nodes[:, 0:6] = (leaf_nodes[:, 0:6].reshape((-1, 3)) / container_size).reshape((-1, 6))
            next_box_vec = deepcopy(self.next_box_vec)
            next_box_vec[:, 3:6] = next_box_vec[:, 3:6] / container_size[0:3]
        if self.no_internal_node_input:
            box_vec[:] = 0

        if len(box_vec) < self.internal_node_holder:
            box_vec = np.concatenate((box_vec, np.zeros((self.internal_node_holder - len(box_vec), 9))))
        else:
            box_vec = box_vec[-self.internal_node_holder:]
        self.leaf_nodes = leaf_nodes
        return np.reshape(np.concatenate((box_vec, leaf_nodes, next_box_vec)), (-1))

    # Generate the next item to be placed.
    def gen_next_box(self, selected_box_idx):
        if not selected_box_idx:
            length=1
        else:
            length = self.select
        return self.box_creator.preview(length)[selected_box_idx]

    # Detect potential leaf nodes and check their feasibility.
    def get_possible_position(self):
        if   self.LNES == 'EMS':
            allPostion = self.space.EMSPoint(self.next_box, self.setting)
        elif self.LNES == 'EV':
            allPostion = self.space.EventPoint(self.next_box, self.setting)
        else:
            assert False, 'Wrong LNES'

        leaf_node_idx = 0
        leaf_node_vec = np.zeros((self.leaf_node_holder, 9))
        tmp_list = []

        if self.LNES == 'EMS':
            leaf_emss_idxes = np.arange(len(self.space.leaf_emss))

            # todo change this to global shuffle
            # if self.shuffle:
            #     np.random.shuffle(leaf_emss_idxes)

            for idx in leaf_emss_idxes:
                ems = self.space.leaf_emss[idx]
                leaf_nodes = ems.leaf_nodes
                tmp_leaf_nodes = []

                for position in leaf_nodes:
                    xs, ys, zs, xe, ye, ze, _, _, _ = position
                    x = xe - xs
                    y = ye - ys
                    z = ze - zs

                    if self.space.drop_box_virtual([x, y, z], (xs, ys), False, self.next_den, self.setting):
                        if self.leaf_without_maxz:
                            new_node = [xs, ys, zs, xe, ye, ze, 0, 0, 1]
                        else:
                            new_node = [xs, ys, zs, xe, ye, self.bin_size[2], 0, 0, 1]
                        tmp_leaf_nodes.append([xs, ys, zs, xe, ye, ze, 0, 0, 1])
                        if new_node not in tmp_list:
                            tmp_list.append(new_node)

                ems.leaf_nodes = tmp_leaf_nodes

            if len(tmp_list) != 0:
                if self.shuffle:
                    if len(tmp_list) > self.leaf_node_holder:
                        if self.test:
                            selected_idx = np.linspace(0, len(tmp_list) - 1, self.leaf_node_holder, dtype=int)
                            tmp_list = np.array(tmp_list)[selected_idx]
                        else:
                            np.random.shuffle(tmp_list)

                tmp_list = tmp_list[0:min(len(tmp_list), self.leaf_node_holder)]
                leaf_node_vec[0:len(tmp_list)] = np.array(tmp_list)

        elif self.LNES == 'EV':
            for position in allPostion:
                xs, ys, zs, xe, ye, ze = position
                x = xe - xs
                y = ye - ys
                z = ze - zs

                if self.space.drop_box_virtual([x, y, z], (xs, ys), False, self.next_den, self.setting):
                    # tmp_list.append([xs, ys, zs, xe, ye, ze, 0, 0, 1])
                    tmp_list.append([xs, ys, zs, xe, ye, self.bin_size[2], 0, 0, 1])
                    leaf_node_idx += 1

                if leaf_node_idx >= self.leaf_node_holder: break

            if len(tmp_list) != 0:
                leaf_node_vec[0:len(tmp_list)] = np.array(tmp_list)

        return leaf_node_vec

    # Convert the selected leaf node to the placement of the current item.
    def LeafNode2Action(self, leaf_node):
        if np.sum(leaf_node[0:6]) == 0: return (0, 0, 0), self.next_box
        x,y = [round(data, 6) for data in leaf_node[3:5] - leaf_node[0:2]]
        record = [0,1,2]
        for r in record:
            if np.isclose(x, self.next_box[r], atol=tolerance):
                record.remove(r)
                break
        for r in record:
            if np.isclose(y, self.next_box[r], atol=tolerance):
                record.remove(r)
                break
        assert len(record) == 1, 'Box size mismatch: {}, {}'.format(self.next_box, leaf_node)
        z = self.next_box[record[0]]
        action = (0, leaf_node[0], leaf_node[1])
        next_box = (x, y, z)
        return action, next_box

    def update_max_min(self, packed_box):
        self.maxXYZ[0] = max(self.maxXYZ[0], packed_box.x + packed_box.lx)
        self.maxXYZ[1] = max(self.maxXYZ[1], packed_box.y + packed_box.ly)
        self.maxXYZ[2] = max(self.maxXYZ[2], packed_box.z + packed_box.lz)
        self.minXYZ[0] = min(self.minXYZ[0], packed_box.x)
        self.minXYZ[1] = min(self.minXYZ[1], packed_box.y)
        self.minXYZ[2] = min(self.minXYZ[2], packed_box.z)

    def calulate_surface_area(self):
        x = self.maxXYZ[0] - self.minXYZ[0]
        y = self.maxXYZ[1] - self.minXYZ[1]
        z = self.maxXYZ[2] - self.minXYZ[2]
        return 2 * (x * y + x * z + y * z)

    def step(self, action, selected_box_idx=0):
        if len(action) != 3:
            action, next_box = self.LeafNode2Action(action)
        else:
            next_box = self.next_box

        idx = [round(float(x), 6) for x in action[1:3]]
        bin_index = 0
        rotation_flag = action[0]
        succeeded = self.space.drop_box(next_box, idx, rotation_flag, self.next_den, self.setting)

        if not succeeded:
            reward = 0.0
            done = True
            info = {'counter': len(self.space.boxes), 'ratio': self.space.get_ratio(),
                    'reward': self.space.get_ratio() * 10, 'packed': self.packed,
                    'final_length': len(self.packed),
                    'distribution_label': self.distribution_label}
            return self.cur_observation(selected_box_idx), reward, done, info

        ################################################
        ############# cal leaf nodes here ##############
        ################################################
        packed_box = self.space.boxes[-1]

        if  self.LNES == 'EMS':
            self.space.GENEMS([packed_box.lx, packed_box.ly, packed_box.lz, *packed_box.vertex_high])

        self.packed.append(
            [packed_box.x, packed_box.y, packed_box.z, packed_box.lx, packed_box.ly, packed_box.lz, bin_index])

        self.update_max_min(packed_box)

        box_ratio = self.get_box_ratio()
        self.box_creator.drop_box(selected_box_idx)  # remove current box from the list
        self.box_creator.generate_box_size()  # add a new box to the list
        reward = box_ratio * 10

        if self.height_reward:
            reward += (self.bin_size[2] - packed_box.lz) / len(self.packed) * 0.1

        done = False
        info = dict()
        info['counter'] = len(self.space.boxes)
        info['distribution_label'] = self.distribution_label
        return self.cur_observation(), reward, done, info

