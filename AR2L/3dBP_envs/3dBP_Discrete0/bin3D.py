import copy
import sys

import numpy as np
import gym
import random

try:
    from .space import Space
    from .binCreator import BoxCreator, RandomSeqCreator, LoadBoxCreator
except:
    from space import Space
    from binCreator import BoxCreator, RandomSeqCreator, LoadBoxCreator
m

class PackingDiscrete(gym.Env):
    def __init__(self,
                 setting=1,
                 container_size=(10, 10, 10),
                 item_set=None,
                 data_name='data/flat_long/opai.txt',
                 threshold=0.01,
                 num_box = 40,
                 num_next_box = 1,
                 candidate_pos_nums = 120,
                 node_dim=9,
                 sparse_reward=True,
                 **kwargs
                 ):

        """
        bin_state:   x, y, z, w, l, h, density, 0, isEmbed
        box_state:   0, 0, 0, w, l, h, density, 0, isEmbed
        constraint:  x, y, z, w, l, h, density, isFeasi, isEmbed
        """
        super(PackingDiscrete, self).__init__()

        self.num_box = num_box
        self.num_next_box = num_next_box
        self.node_dim = node_dim
        self.candidate_pos_nums = candidate_pos_nums

        self.bin_size = container_size
        self.minimum_size = np.min(np.array(item_set))
        self.maximum_size = np.max(np.array(item_set))
        self.setting = setting
        self.item_set = item_set

        self.data_name = data_name
        self.threshold = threshold

        self.orientation = 2 if setting != 2 else 6
        self.sparse_reward = sparse_reward
        self.space = Space(*self.bin_size, self.minimum_size, self.maximum_size, self.num_box)
        # fix by wzf : add LoadBoxCreator
        # self.box_creator = RandomSeqCreator(item_set, setting)
        self.box_creator = LoadBoxCreator(data_name)

        self.observation_size = (self.num_box + self.num_next_box + self.candidate_pos_nums) * self.node_dim
        self.observation_space = gym.spaces.Box(low=0., high=self.space.height, shape=((self.observation_size,)))
        self.action_space = gym.spaces.Discrete(n=self.candidate_pos_nums)

        self.next_box = None
        self.next_box_density = None
        self.next_box_vec = np.zeros((self.node_dim, ))

        self.packed_box = [(
            self.space.boxes[0].w, self.space.boxes[0].l, self.space.boxes[0].h,
            self.space.boxes[0].x, self.space.boxes[0].y, self.space.boxes[0].z,
            self.space.boxes[0].density, self.space.boxes[0].box_order,
        )]
        self.packed_box_num = len(self.packed_box)

        self.is_inner = False


    def reset(self):
        self.box_creator.reset()
        self.space.reset()

        self.next_box = None
        self.next_box_density = None
        self.next_box_vec = np.zeros((self.node_dim, ))

        self.packed_box = [(
            self.space.boxes[0].w, self.space.boxes[0].l, self.space.boxes[0].h,
            self.space.boxes[0].x, self.space.boxes[0].y, self.space.boxes[0].z,
            self.space.boxes[0].density, self.space.boxes[0].box_order,
        )]
        self.packed_box_num = len(self.packed_box)

        self.is_inner = False
        return self.cur_observation()

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            self.SEED = seed
        return [seed]

    def get_box_ratio(self):
        assert self.next_box is not None
        return np.prod(np.array(self.next_box[0:3])) / np.prod(self.space.plain_size)

    def generate_next_box(self):
        return self.box_creator.preview(1)[0]

    def generate_next_n_box(self):
        return self.box_creator.preview(self.num_next_box)

    def cur_observation(self):
        self.packed_box_num = len(self.packed_box)
        self.next_n_box = np.array(self.generate_next_n_box())

        self.next_box = None
        self.next_box_density = None
        self.next_box_vec = np.zeros((self.node_dim, ))

        self.next_n_box_density = self.next_n_box[:, 3][:, None]
        self.next_n_box = self.next_n_box[:, :3]

        self.next_n_box_vec = np.zeros((self.num_next_box, self.node_dim))
        self.next_n_box_vec[:self.next_n_box.shape[0], 3:7] = \
            np.concatenate((self.next_n_box, self.next_n_box_density), axis=-1)
        self.next_n_box_vec[:self.next_n_box.shape[0], -1] = 1.

        self.bin_nodes = copy.deepcopy(self.space.box_vec)
        self.graph_nodes = np.zeros((self.num_box, self.node_dim))
        bin_node_num = min(len(self.bin_nodes), self.num_box)
        self.graph_nodes[:bin_node_num, :] = np.array(self.bin_nodes)[-bin_node_num:]

        base_obs = np.reshape(np.concatenate((self.graph_nodes, self.next_n_box_vec), axis=0), (-1,))

        # IMPORTANT: reset() must return fixed-size observation matching observation_space.
        # The original torch code only returned base_obs, but VecEnv buffers are allocated
        # using observation_space, so we must append candidate position features.
        # At reset (not inner), we don't have a selected next_box; we can still compute
        # candidates for the FIRST next box in the preview list to keep shape consistent.
        try:
            # Use the first next box as candidate target
            next_box = self.next_n_box[0]
            next_den = float(self.next_n_box_density[0][0])
            feas_pos_vec, unfeas_pos_vec = self.compute_feasible_ems(
                self.space, next_box, next_den, self.packed_box_num
            )
            cand = feas_pos_vec
            if cand.shape[0] < self.candidate_pos_nums:
                cand = np.concatenate((cand, unfeas_pos_vec), axis=0)
            if cand.shape[0] < self.candidate_pos_nums:
                padding = np.zeros((self.candidate_pos_nums - cand.shape[0], self.node_dim), dtype=float)
                cand = np.concatenate((cand, padding), axis=0)
            cand = cand[:self.candidate_pos_nums]
            cand_flat = np.reshape(cand, (-1,))
        except Exception:
            # Fallback to zeros if anything unexpected happens
            cand_flat = np.zeros((self.candidate_pos_nums * self.node_dim,), dtype=float)

        observation = np.concatenate((base_obs, cand_flat), axis=0)
        return observation


    def inner_observation(self, box_idx):

        assert self.is_inner
        self.packed_box_num = len(self.packed_box)

        bin_nodes = copy.deepcopy(self.space.box_vec)
        graph_nodes = np.zeros((self.num_box, self.node_dim))
        bin_node_num = min(len(bin_nodes), self.num_box)
        graph_nodes[:bin_node_num, :] = np.array(bin_nodes)[-bin_node_num:]

        next_box, next_box_density = self.next_box, self.next_box_density
        self.next_box_vec[3:7] = np.array([next_box[0], next_box[1], next_box[2], next_box_density])
        self.next_box_vec[-1] = 1.

        perm_box = np.concatenate((self.next_n_box[0:box_idx], self.next_n_box[box_idx+1:]), axis=0)
        perm_den = np.concatenate((self.next_n_box_density[0:box_idx], self.next_n_box_density[box_idx+1:]), axis=0)
        perm_seq = np.concatenate((perm_box, perm_den), axis=1)

        perm_seqVec = np.zeros((self.num_next_box - 1, self.node_dim))
        perm_seqVec[:, 3:7] = perm_seq
        perm_seqVec[:, -1] = 1.
        observation = np.reshape(np.concatenate((graph_nodes, self.next_box_vec[None], perm_seqVec), axis=0), (-1,))

        feas_pos_vec, unfeas_pos_vec = self.compute_feasible_ems(
            self.space, next_box, next_box_density, self.packed_box_num
        )

        if feas_pos_vec.shape[0] >= self.candidate_pos_nums:
            feas_pos_vec = np.reshape(feas_pos_vec[:self.candidate_pos_nums, ...], (-1,))
            return np.concatenate((observation, feas_pos_vec), axis=0)
        else:
            feas_pos_vec = np.concatenate((feas_pos_vec, unfeas_pos_vec), axis=0)
            if feas_pos_vec.shape[0] < self.candidate_pos_nums:
                padding_pos_vec = np.zeros((self.candidate_pos_nums - feas_pos_vec.shape[0], *feas_pos_vec.shape[1:]))
                feas_pos_vec = np.concatenate((feas_pos_vec, padding_pos_vec), axis=0)
            feas_pos_vec = np.reshape(feas_pos_vec[:self.candidate_pos_nums, ...], (-1,))
            return np.concatenate((observation, feas_pos_vec), axis=0)


    def compute_feasible_ems(self, space, next_box, next_box_density, packed_box_num):
        possible_positions = space.EMSPoint(next_box, self.setting)

        feasible_set = np.zeros((0, self.node_dim), dtype=float)
        unfeasible_set = np.zeros((0, self.node_dim), dtype=float)
        for position in possible_positions:
            x, y, z, x_w, y_l, z_h = position
            w, l, h = x_w - x, y_l - y, z_h - z

            isFeasible, feasible_z = space.drop_box_virtual(
                [w, l, h], (x, y), next_box_density, packed_box_num, self.setting
            )
            if isFeasible:
                feasible_set = np.concatenate(
                    (feasible_set, [[x, y, feasible_z, w, l, h, next_box_density, 1., 1.]]), axis=0
                )
            else:
                unfeasible_set = np.concatenate(
                    (unfeasible_set, [[x, y, feasible_z, w, l, h, next_box_density, 0., 1.]]), axis=0
                )
        self.clear_up_virtual_box(space)
        return feasible_set, unfeasible_set

    def clear_up_virtual_box(self, space):
        for box in space.boxes:
            box.up_virtual_edges = dict()
        return

    def generate_ems(self, space, packed_box):
        space.GENEMS(
            [
                packed_box.x,
                packed_box.y,
                packed_box.z,
                packed_box.x + packed_box.w,
                packed_box.y + packed_box.l,
                packed_box.z + packed_box.h,
            ]
        )
        return


    def decode_action(self, action):
        x, y, z, w, l, h, density, box_idx, self.is_inner = action
        box_idx = int(box_idx)

        if not self.is_inner:
            if not abs(w * l * h - np.prod(self.next_box)) < 1e-3:
                return 0, 0, 0, self.next_box[0], self.next_box[1], self.next_box[2], self.next_box_density, box_idx

        else:
            self.next_box = self.next_n_box[box_idx]
            self.next_box_density = self.next_n_box_density[box_idx][0]
            assert abs(w * l * h - np.prod(self.next_box)) < 1e-3 and abs(self.next_box_density - density) < 1e-3

        return x, y, z, w, l, h, density, box_idx

    def step(self, action):
        x, y, z, w, l, h, density, box_idx = self.decode_action(action)

        if not self.is_inner:
            succeeded = self.space.drop_box([w, l, h], [x,y], density, self.setting)

            if not succeeded:
                ratio = self.space.get_ratio()
                reward = ratio * 10 * self.sparse_reward
                done = True
                info = {'counter': len(self.space.boxes)-1,
                        'ratio': ratio,
                        'reward': reward}
                return self.cur_observation(), reward, done, info

            self.packed_box.append(
                (
                    self.space.boxes[-1].w, self.space.boxes[-1].l, self.space.boxes[-1].h,
                    self.space.boxes[-1].x, self.space.boxes[-1].y, self.space.boxes[-1].z,
                    self.space.boxes[-1].density, self.space.boxes[-1].box_order,
                )
            )

            box_ratio = self.get_box_ratio()
            reward = box_ratio * 10 * (1 - self.sparse_reward)
            done = False
            info = {'counter': len(self.space.boxes)-1}

            self.generate_ems(self.space, self.space.boxes[-1])

            self.box_creator.drop_box(index=box_idx)
            return self.cur_observation(), reward, done, info

        else:
            return self.inner_observation(box_idx), 0., False, {}
