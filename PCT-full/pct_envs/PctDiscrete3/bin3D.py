import copy
import time

from .space import Space
import numpy as np
import gym
from .binCreator import RandomBoxCreator, LoadBoxCreator, BoxCreator
import random
from copy import deepcopy
from .draw_tools import render_all
# render_all(items, args.container_size, [], True, output_image_path, step_counter)


def get_true_y(frontier, length, height, lx, x, y, z):
    # x, y, z = next_box
    # frontier = self.frontier
    ly_list = []
    for lz in range(0, height - z):
        max_y = np.max(frontier[lx:lx + x, lz:lz + z])
        if max_y + y > length:
            max_y = 1e10
        ly_list.append(max_y)
    return np.argmin(ly_list)

def min_pooling_2d_torch(matrix, kernel_size):
    kx, ky = kernel_size
    matrix_np = np.asarray(matrix)
    out_x = matrix_np.shape[0] - kx + 1
    out_y = matrix_np.shape[1] - ky + 1
    result = np.empty((out_x, out_y), dtype=matrix_np.dtype)
    for i in range(out_x):
        for j in range(out_y):
            result[i, j] = np.min(matrix_np[i:i + kx, j:j + ky])
    return result

class PackingDiscrete(gym.Env):
    def __init__(self,
                args,
                **kwags):

        self.internal_node_holder = args.internal_node_holder
        self.leaf_node_holder = args.leaf_node_holder
        self.next_holder = 1

        self.shuffle = args.shuffle
        self.bin_size = [int(x) for x in args.container_size]
        self.size_minimum = np.min(np.array(args.item_size_set))
        self.setting = args.setting
        self.item_set = args.item_size_set
        self.orientation = 2
        self.preview = args.preview
        self.select = args.select
        self.packe_candidate_list = None
        self.practical_constrain = args.practical_constrain.split('_') if args.practical_constrain is not None else []
        # The class that maintains the contents of the bin.
        self.space = Space(*self.bin_size, self.size_minimum, self.internal_node_holder,
                           physics = 'physics' in self.practical_constrain or 'stability' in self.practical_constrain,
                           robot_in_roop='robot' in self.practical_constrain,
                           check_area = 'area' in self.practical_constrain,
                           distribution = args.distribution, args = args)


        # Generator for train/test data
        if not args.load_dataset:
            assert args.item_size_set is not None
            item_prob = None
            item_size_set = args.item_size_set
            # if args.distribution == 'deli_prob':
            if args.distribution == 'deli_deploy':     # maybe here is 'deli_deploy' to follow givenData.get_deli_prob_dataset()
                item_prob = [x[-1] for x in args.item_size_set]
                item_size_set = np.array([x[:-1] for x in args.item_size_set]).astype('int')
            self.box_creator = RandomBoxCreator(item_size_set, item_prob = item_prob)
            assert isinstance(self.box_creator, BoxCreator)
        if args.load_dataset:
            self.box_creator = LoadBoxCreator(args.dataset_path)

        # self.test = args.load_dataset
        self.test = False

        self.next_box_vec = np.zeros((self.next_holder, 9))

        self.LNES = args.lnes  # Leaf Node Expansion Schemes: EMS (recommend), EV, EP, CP, FC
        self.large_scale = args.large_scale
        self.model_architecture = args.model_architecture
        self.distribution = args.distribution

        if self.model_architecture == 'PCT':
            self.observation_space = gym.spaces.Box(low=0.0, high=self.space.height,
                                                shape=((self.internal_node_holder + self.leaf_node_holder + self.next_holder) * 9,))
            self.action_space = None

        elif self.model_architecture == 'CDRL':
            area = int(self.bin_size[0] * self.bin_size[1])
            obs_len = area * 6
            act_len = 2 * area
            self.observation_space = gym.spaces.Box(low=0.0, high=self.bin_size[2], shape=(obs_len,))
            self.action_space = gym.spaces.Discrete(act_len)

        elif self.model_architecture == 'PackE':
            area = int(self.bin_size[0] * self.bin_size[1])
            obs_len = area * 3 + 6
            act_len = 2 * area
            self.observation_space = gym.spaces.Box(low=0.0, high=self.bin_size[2], shape=(obs_len,))
            self.action_space = gym.spaces.Discrete(act_len)

        elif self.model_architecture == 'Attend2Pack':
            area = int(self.bin_size[0] * self.bin_size[2])
            mask_len = 2 * self.bin_size[0]
            obs_len = area * 2 + 3 + mask_len
            self.observation_space = gym.spaces.Box(low=0.0, high=self.bin_size[1], shape=(obs_len,))
            self.action_space = gym.spaces.Discrete(2 * self.bin_size[0])

        elif self.model_architecture == 'RCQL':
            obs_len = (self.internal_node_holder + self.next_holder) * 9
            self.observation_space = gym.spaces.Box(low=0.0, high=np.max(self.bin_size), shape=(obs_len,))
            self.action_space = None

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

    def reset(self):
        self.box_creator.reset()
        self.space.low_bound = np.min(np.array(self.box_creator.box_set))       # fix low bound
        self.packed = []
        self.constraint_rewards = []
        self.space.reset()
        self.box_creator.generate_box_size()
        cur_observation = self.cur_observation()
        if self.model_architecture == 'RCQL':
            self.reward_g = 0

        return cur_observation

    def get_box_plain(self):
        x_plain = np.ones((self.bin_size[0], self.bin_size[1]), dtype=np.int32) * self.next_box[0]
        y_plain = np.ones((self.bin_size[0], self.bin_size[1]), dtype=np.int32) * self.next_box[1]
        z_plain = np.ones((self.bin_size[0], self.bin_size[1]), dtype=np.int32) * self.next_box[2]
        # c_plain = np.ones((self.bin_size[0], self.bin_size[1]), dtype=np.int32) * self.next_load_bearing * 0.1
        observation = [x_plain, y_plain, z_plain]
        if self.setting == 3:
            d_plain = np.ones((self.bin_size[0], self.bin_size[1]), dtype=np.int32) * self.next_den
            observation.append(d_plain)
        return observation

    # Count and return all PCT nodes.
    def cur_observation(self, selected_box_idx=0):
        self.next_box = self.gen_next_box(0)

        if self.test:
            if self.setting == 3:
                if len(self.next_box) >= 4:
                    self.next_den = self.next_box[3]
                else:
                    self.next_den = np.random.random()
                    while self.next_den == 0:
                        self.next_den = np.random.random()
            else:
                self.next_den = 1
            self.next_box = [int(self.next_box[0]), int(self.next_box[1]), int(self.next_box[2])]
        else:
            if self.setting < 3:
                self.next_den = 1
            else:
                self.next_den = np.random.random()
                while self.next_den == 0:
                    self.next_den = np.random.random()

        if self.model_architecture == 'PCT':
            boxes = []
            leaf_nodes = []

            self.next_cat = np.random.randint(0, 4)
            box_vec = deepcopy(self.space.box_vec)
            boxes.append(box_vec)
            leaf_nodes.append(self.get_possible_position())

            next_box = sorted(list(self.next_box))
            self.next_box_vec[:, 3:6] = next_box
            self.next_box_vec[:, 0] = self.next_den
            self.next_box_vec[:, 1] = self.next_cat
            self.next_box_vec[:, -1] = 1

            next_box_vec = deepcopy(self.next_box_vec)
            if self.large_scale:
                max_len = np.max(self.space.plain_size)
                boxes[0][:, 0:6] = boxes[0][:, 0:6] / max_len
                leaf_nodes[0][:, :-1] = leaf_nodes[0][:, :-1] / max_len
                next_box_vec = deepcopy(self.next_box_vec)
                next_box_vec[:, 3:6] = next_box / max_len
            return np.reshape(np.concatenate((*boxes, *leaf_nodes, next_box_vec)), (-1))

        elif self.model_architecture == 'CDRL':
            self.next_cat = np.random.randint(0, 4)
            hmap = copy.deepcopy(self.space.plain)
            masks = self.get_possible_position_CDRL()
            size = self.get_box_plain()
            if np.sum(masks) == 0:
                masks = np.ones_like(masks)
            hmap = hmap[0:self.bin_size[0], 0:self.bin_size[1]] / self.bin_size[2]
            masks = masks[:, 0:self.bin_size[0], 0:self.bin_size[1]]
            return np.reshape(np.stack((hmap, *masks, *size)), newshape=(-1,))

        elif self.model_architecture == 'PackE':
            hmap = copy.deepcopy(self.space.plain)
            masks, self.packe_candidate_list = self.get_possible_position_PackE()
            if np.sum(masks) == 0:
                masks = np.ones_like(masks)
            next_box = [self.next_box[0],
                        self.next_box[1],
                        self.next_box[2],
                        self.bin_size[0] - self.next_box[0],
                        (self.bin_size[1] - self.next_box[1]) * self.bin_size[0],
                        (self.bin_size[1] - self.next_box[1]) * self.bin_size[0] + self.bin_size[0] - self.next_box[0]
                        ]
            next_box = np.array(next_box, dtype=np.float64)
            next_box /= self.bin_size[0] * self.bin_size[1] * 1.0
            hmap = hmap[0:self.bin_size[0], 0:self.bin_size[1]] / self.bin_size[2]
            masks = masks[:, 0:self.bin_size[0], 0:self.bin_size[1]]
            observation = np.concatenate((hmap.reshape(-1), masks.reshape(-1), next_box))
            return observation

        elif self.model_architecture == 'Attend2Pack':
            mask = self.get_possible_position_Attend2Pack()
            observation = np.reshape(np.concatenate((self.space.last_frontier.reshape(-1),
                                             self.space.frontier.reshape(-1),
                                             np.array(self.next_box).reshape(-1),
                                             mask)), -1)
            return observation

        elif self.model_architecture == 'RCQL':
            boxes = []
            box_vec = deepcopy(self.space.box_vec)
            boxes.append(box_vec)
            next_box = list(self.next_box)
            self.next_box_vec[:, 3:6] = next_box
            self.next_box_vec[:, -1] = 1
            return np.reshape(np.concatenate((*boxes, self.next_box_vec)), (-1))

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
            allPostion = self.space.EMSPoint(self.next_box,  self.setting)
        elif self.LNES == 'EV':
            allPostion = self.space.EventPoint(self.next_box,  self.setting)
        elif self.LNES == 'EP':
            allPostion = self.space.ExtremePoint2D(self.next_box, self.setting)
        elif self.LNES == 'CP':
            allPostion = self.space.CornerPoint(self.next_box, self.setting)
        elif self.LNES == 'FC':
            allPostion = self.space.FullCoord(self.next_box, self.setting)
        else:
            assert False, 'Wrong LNES'

        if self.shuffle:
            np.random.shuffle(allPostion)

        leaf_node_idx = 0
        leaf_node_vec = np.zeros((self.leaf_node_holder, 9))
        tmp_list = []

        for position in allPostion:
            xs, ys, zs, xe, ye, ze = position
            x = xe - xs
            y = ye - ys
            z = ze - zs

            if self.space.drop_box_virtual([x, y, z], (xs, ys), False, self.next_den, self.setting):
                tmp_list.append([xs, ys, zs, xe, ye, self.bin_size[2], 0, 0, 1])
                # tmp_list.append([xs, ys, zs, xe, ye, ze, 0, 0, 1])
                leaf_node_idx += 1

            if leaf_node_idx >= self.leaf_node_holder: break

        if len(tmp_list) != 0:
            leaf_node_vec[0:len(tmp_list)] = np.array(tmp_list)
        return leaf_node_vec


    def get_possible_position_CDRL(self):
        width  = self.bin_size[0]
        length = self.bin_size[1]
        max_axis = int(max(width, length))
        masks = []
        rot_num = 2
        for rot in range(rot_num):  # 0 x y z, 1 x z y, 2 y x z, 3 y z x, 4 z x y, 5 z y x
            if rot == 0:
                sizex, sizey, sizez = self.next_box[0], self.next_box[1], self.next_box[2]
            elif rot == 1:
                sizex, sizey, sizez = self.next_box[1], self.next_box[0], self.next_box[2]
            sizex, sizey, sizez = int(sizex), int(sizey), int(sizez)

            action_mask = np.zeros((max_axis, max_axis))
            if self.setting != 2:
                for lx in range(self.bin_size[0] - sizex):
                    for ly in range(self.bin_size[1] - sizey):
                        if self.space.drop_box_virtual([sizex, sizey, sizez], (lx, ly), False, self.next_den, self.setting):
                            action_mask[lx, ly] = 1
            else:
                if sizex <= self.bin_size[0] and sizey <= self.bin_size[1]:
                    valid_mask, M, N = self.space.check_box_rule_batch(sizex, sizey, sizez, without_stable_check=True)
                    action_mask[0:M, 0:N] = valid_mask
            masks.append(action_mask)
        masks = np.array(masks)
        return masks

    def get_possible_position_Attend2Pack(self):
        width  = self.bin_size[0]
        masks = []
        rot_num =  2
        for rot in range(rot_num):
            if rot == 0:
                sizex, sizey, sizez = self.next_box[0], self.next_box[1], self.next_box[2]
            elif rot == 1:
                sizex, sizey, sizez = self.next_box[1], self.next_box[0], self.next_box[2]
            sizex, sizey, sizez = int(sizex), int(sizey), int(sizez)
            action_mask = np.zeros(width)
            ly_martrix = min_pooling_2d_torch(self.space.frontier, (sizex, sizez))
            for lx in range(ly_martrix.shape[0]):
                # ly = np.argmin(ly_martrix[lx])
                # 此处可以尝试修改为同时得到配套的lx、ly，省去两者不一的情况
                # 另外一点可能存在的问题是此处确定lx时检查的ly的mask与后面依据lx确定的ly并不一致，两者似有冲突
                ly = get_true_y(self.space.frontier, self.space.plain_size[1], self.space.height, lx, sizex, sizey, sizez)
                if ly + sizey <= self.bin_size[1] and lx + sizex <= self.bin_size[0] and \
                        self.space.drop_box_virtual([sizex, sizey, sizez], (lx, ly), False, self.next_den, self.setting):
                    action_mask[lx] = 1
            masks.append(action_mask)
        return np.concatenate(masks)

    def get_possible_position_PackE(self):  # check here mask is right???
        width  = self.bin_size[0]
        length = self.bin_size[1]
        height = self.bin_size[2]
        max_axis = int(max(width, length))
        masks = []
        packe_candidate_list = self.space.ExtremePointClean()
        check_candidate_list = []
        rot_num = 2
        for rot in range(rot_num):  # 0 x y z, 1 x z y, 2 y x z, 3 y z x, 4 z x y, 5 z y x
            if rot == 0:
                sizex, sizey, sizez = self.next_box[0], self.next_box[1], self.next_box[2]
            elif rot == 1:
                sizex, sizey, sizez = self.next_box[1], self.next_box[0], self.next_box[2]
            sizex, sizey, sizez = int(sizex), int(sizey), int(sizez)
            action_mask = np.zeros((max_axis, max_axis))

            if sizex < self.bin_size[0] and sizey < self.bin_size[1]:
                max_h_martix, M, N = self.space.check_box_rule_batch(sizex, sizey, sizez)
                stable_idxs = np.where(max_h_martix >= 0)
                if len(stable_idxs[0]) != 0:
                    candidate = np.concatenate([stable_idxs[0].reshape(-1, 1), stable_idxs[1].reshape(-1, 1), max_h_martix[stable_idxs].reshape(-1, 1)], axis=1)
                    if len(check_candidate_list) == 0:
                        check_candidate_list = candidate
                    else:
                        check_candidate_list = np.concatenate([check_candidate_list, candidate], axis=0)
                action_mask[0:M, 0:N] = np.where(max_h_martix >= 0, 1, 0)
            masks.append(action_mask)
        masks = np.array(masks)     # here mask is right???
        if len(check_candidate_list) == 0:
            check_candidate_list = np.array([[0, 0, 0]])
            masks = np.ones_like(masks)
        first_fit_idx = np.argmin(check_candidate_list[:, 0] + check_candidate_list[:, 1])
        floor_building_idx = np.argmin(check_candidate_list[:, 2])
        column_building_idx = np.argmax(check_candidate_list[:, 2])
        packe_candidate_list.append(check_candidate_list[first_fit_idx][0:2])
        packe_candidate_list.append(check_candidate_list[floor_building_idx][0:2])
        packe_candidate_list.append(check_candidate_list[column_building_idx][0:2])
        packe_candidate_list = np.unique(np.round(np.array(packe_candidate_list), 6), axis=0)
        self.masks = masks
        return masks, packe_candidate_list


    # Convert the selected leaf node to the placement of the current item.
    def LeafNode2Action(self, leaf_node):
        if np.sum(leaf_node[0:6]) == 0: return (0, 0, 0), self.next_box
        x = int(leaf_node[3] - leaf_node[0])
        y = int(leaf_node[4] - leaf_node[1])
        z = list(self.next_box)
        z.remove(x)
        z.remove(y)
        z = z[0]
        action = (0, int(leaf_node[0]), int(leaf_node[1]))
        next_box = (x, y, int(z))
        return action, next_box

    def rotate_next_box(self, rot):
        if rot == 0:
            sizex, sizey, sizez = self.next_box[0], self.next_box[1], self.next_box[2]
        else:
            assert rot == 1
            sizex, sizey, sizez = self.next_box[1], self.next_box[0], self.next_box[2]

        return [sizex, sizey, sizez]

    def step(self, action):
        selected_box_idx = 0
        if self.model_architecture == 'CDRL' or self.model_architecture == 'RCQL':
            rotation_flag = action[0]
            next_box = self.rotate_next_box(rotation_flag)
            rotation_flag = 0

        elif self.model_architecture == 'PackE':
            rotation_flag, lx, ly = np.unravel_index(action, (2, self.bin_size[0], self.bin_size[1]))
            next_box = self.next_box

            action = [rotation_flag, int(lx), int(ly)]
        elif self.model_architecture == 'Attend2Pack':
            rotation_flag, lx = np.unravel_index(action, (2, self.bin_size[0]))
            next_box = self.next_box if not rotation_flag else [self.next_box[1], self.next_box[0], self.next_box[2]]
            ly = self.space.get_frontier_y(lx, next_box)
            rotation_flag = 0
            action = [rotation_flag, lx, ly]
        else:
            assert self.model_architecture == 'PCT'
            if len(action) != 3: action, next_box = self.LeafNode2Action(action)
            else:
                next_box = self.next_box
            rotation_flag = action[0]

        idx = [action[1], action[2]]
        bin_index = 0
        succeeded = self.space.drop_box(next_box, idx, rotation_flag, self.next_den, self.setting, model_architecture = self.model_architecture)
        constraint_reward = 0.0

        if 'variance' in self.practical_constrain:
            constraint_reward = -self.space.get_variance()
        elif 'robot' in self.practical_constrain:
            constraint_reward = (self.bin_size[0] * self.bin_size[1] * self.bin_size[2] - np.sum(self.space.left_space)) / (self.bin_size[0] * self.bin_size[1] * self.bin_size[2] - np.sum(self.space.plain))
        self.constraint_rewards.append(constraint_reward)
        if 'stability' in self.practical_constrain:
            all_distances = self.space.interface.distance_to_registion()
            if np.max(all_distances) > 0.1:
                succeeded = False

        if not succeeded:
            # print(label)
            # self.space.drop_box(next_box, idx, rotation_flag, self.next_den, self.setting,
            #                                 model_architecture=self.model_architecture)
            # masks, packe_candidate_list = self.get_possible_position_PackE()
            # label = masks[rotation_flag, lx, ly]
            # if len(self.space.boxes) == 0:
            #     print('Failed to place the first box')
            if 'physics' in self.practical_constrain:
                score = self.space.get_loading_force()
                if self.setting != 3:
                    constraint_reward =  - score * 0.002
                else:
                    constraint_reward =  - score * 0.01
            elif 'variance' in self.practical_constrain:
                score = np.mean(self.constraint_rewards)
                if self.distribution == 'deli':
                    constraint_reward = score * 5
                else:
                    constraint_reward = score * 10
            elif 'robot' in self.practical_constrain:
                score = np.mean(self.constraint_rewards)
                constraint_reward = score * 20
            else:
                score = 0
                constraint_reward = 0
            reward = 0.0 + constraint_reward

            done = True
            info = {'counter': len(self.space.boxes), 'ratio': self.space.get_ratio(),
                    'reward': self.space.get_ratio() * 10,
                    'packed': self.packed,
                    'constraint_reward': score,
                    'final_length': len(self.packed)}

            # if self.model_architecture == 'PackE':
            #     packed_box = self.space.boxes[-1]
            #     self.packed.append(
            #         [packed_box.x, packed_box.y, packed_box.z, packed_box.lx, packed_box.ly, packed_box.lz, bin_index])
            # render_all(self.packed, self.bin_size, [], True, './images', 0)

            return self.cur_observation(selected_box_idx), reward, done, info

        ################################################
        ############# cal leaf nodes here ##############
        ################################################
        packed_box = self.space.boxes[-1]

        if  self.LNES == 'EMS':
            self.space.GENEMS([packed_box.lx, packed_box.ly, packed_box.lz,
                                           packed_box.lx + packed_box.x, packed_box.ly + packed_box.y,
                                           packed_box.lz + packed_box.z])

        self.packed.append(
            [packed_box.x, packed_box.y, packed_box.z, packed_box.lx, packed_box.ly, packed_box.lz, bin_index])

        box_ratio = self.get_box_ratio()
        self.box_creator.drop_box(selected_box_idx)  # remove current box from the list
        self.box_creator.generate_box_size()  # add a new box to the list
        
        reward = box_ratio * 10
        if self.model_architecture == 'PackE':
            reward = self.space.get_ratio()

        done = False
        info = dict()
        info['counter'] = len(self.space.boxes)
        observation =  self.cur_observation()

        if self.model_architecture == 'PackE':
            qurey = np.array(idx)
            packe_candidate_list = self.packe_candidate_list
            distance = np.linalg.norm(packe_candidate_list - qurey, axis=1)
            packe_heuristic_reward = np.exp(np.min(distance) / np.max(self.bin_size))  - 1
            packe_heuristic_reward *= 0.5
            reward += packe_heuristic_reward
        if self.model_architecture == 'RCQL':
            max_h = np.max(self.space.plain)
            new_reward_g = max_h * self.bin_size[0] * self.bin_size[1] - self.space.get_ratio() * self.bin_size[0] * self.bin_size[1] * self.bin_size[2]
            new_reward_g /= self.bin_size[0] * self.bin_size[1] * self.bin_size[2]
            reward = new_reward_g - self.reward_g
            self.reward_g = new_reward_g
        # print('max_h', np.max(self.space.plain))
        return observation, reward, done, info

