import os, sys
from shutil import copyfile, copytree
import argparse
import numpy as _np
import numpy as np
from gym.envs.registration import register
import gym
from time import time
import re
import random
import pickle

import jittor as jt
import jittor.nn as nn

import matplotlib
if sys.platform != 'linux': matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


# ========== Jittor helper functions ==========

def orthogonal_init(weight, gain=1.0):
    """Orthogonal initialization (replaces nn.init.orthogonal_)"""
    shape = weight.shape
    if len(shape) < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")
    rows = shape[0]
    cols = _np.prod(shape[1:])
    flat_shape = (rows, cols) if rows >= cols else (cols, rows)
    a = _np.random.normal(0.0, 1.0, flat_shape)
    q, r = _np.linalg.qr(a)
    d = _np.diag(r)
    ph = _np.sign(d)
    q *= ph
    if rows < cols:
        q = q.T
    q = q.reshape(shape)
    weight.assign(jt.array(q.astype(_np.float32)) * gain)
    return weight

def constant_init(bias, val):
    """Constant initialization (replaces nn.init.constant_)"""
    bias.assign(jt.full_like(bias, val))
    return bias

def safe_save(state_dict, path):
    """Save model state dict using pickle (avoids jt.save importing torch)"""
    save_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, jt.Var):
            save_dict[k] = v.data
        else:
            save_dict[k] = v
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(save_dict, f)

def safe_load(path):
    """Load model state dict from pickle or jt.load"""
    try:
        with open(path, 'rb') as f:
            state_dict = pickle.load(f)
        return state_dict
    except Exception:
        return jt.load(path)


def get_args():
    parser = argparse.ArgumentParser(description='3D Bin Packing Arguments')
    parser.add_argument('--setting', type=int, default=1,
                        help='Experiemnt setting: 1 | 2 | 3')
    # parser.add_argument('--container-size', type=float, default=10,
    #                     help='The width, length and height of the container')
    parser.add_argument('--max-item-size', type=int, default=5,
                        help='the maximum size of box')
    parser.add_argument('--min-item-size', type=int, default=1,
                        help='the minimum size of box')
    parser.add_argument('--continuous', action='store_true', default=False,
                        help='Use continuous environment or discrete environment')
    parser.add_argument('--num-box', type=int, default=80,
                        help='The maximum number of nodes to represent the bin state')
    parser.add_argument('--num-next-box', type=int, default=5,
                        help='The maximum number of next box, default is 5')
    parser.add_argument('--num-candidate-action', type=int, default=120,
                        help='The maximum number of particles to represent the feasible actions')
    parser.add_argument('--node-dim', type=int, default=9,
                        help='The vector size to represent one node')
    parser.add_argument('--sparse-reward', type=int, default=1,
                        help='The reward from env can be dense (0) or sparse (1)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Donot use cuda')
    parser.add_argument('--device', type=int, default=0,
                        help='Choose GPUs to train model')
    parser.add_argument('--seed', type=int, default=0,
                        help='Set the random seed')

    parser.add_argument('--training-algorithm', type=str, default='ppo',
                        help='Choose one training algorithm: ppo')
    parser.add_argument('--num-processes', type=int, default=16,
                        help='The number of parallel processes used for training')
    parser.add_argument('--num-steps', type=int, default=30,
                        help='The rollout length for n-step training')
    # parser.add_argument('--num-processes', type=int, default=1,
    #                     help='The number of parallel processes used for training')
    # parser.add_argument('--num-steps', type=int, default=4,
    #                     help='The rollout length for n-step training')
    parser.add_argument('--actor-loss-coef', type=float, default=1.0,
                        help='The coefficient of actor loss')
    parser.add_argument('--critic-loss-coef', type=float, default=1.0,
                        help='The coefficient of critic loss')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='The maximum norm of gradients')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Discount factor of Return')

    parser.add_argument('--embedding-size', type=int, default=64,
                        help='Dimension of the input embedding')
    parser.add_argument('--hidden-size', type=int, default=128,
                        help='Dimension of hidden layers')
    parser.add_argument('--gat-layer-num', type=int, default=1,
                        help='The number of GAT layers')

    parser.add_argument('--print-log-interval', type=int, default=10,
                        help='The frequency of printing the training logs')
    parser.add_argument('--sample-from-distribution', action='store_true', default=False,
                        help='Sample continuous item size from a Uniform distribution')
    parser.add_argument('--sample-left-bound', type=float, default=1.,
                        help='The left bound of the uniform distribution')
    parser.add_argument('--sample-right-bound', type=float, default=5.,
                        help='the right bound of the uniform distribution')
    parser.add_argument('--unit-interval', type=float, default=1.,
                        help='the unit interval for height samples')

    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Set learning rate for a2c (default: 1e-6) or ppo (default: 3e-4)')
    parser.add_argument('--num-env-steps', type=int, default=10e6,
                        help='number of environment steps to train (default: 10e6)')

    parser.add_argument('--use-gae', action='store_true', default=True,
                        help='choose whether to use GAE for advantage approximation or not')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument('--clip-param', type=float, default=0.1,
                        help='ppo clip parameter (default: 0.1)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--ppo-epoch', type=int, default=1,
                        help='number of ppo epochs (default: 10)')
    parser.add_argument('--use-proper-time-limits', action='store_true', default=False,
                        help='compute returns taking into account time limits')
    parser.add_argument('--value-loss-coef', type=float, default=1.,
                        help='The coefficient of value loss of PPO')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='The coefficient of entropy of PPO')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.5)

    # add by wzf
    parser.add_argument('--model-save-interval', type=int, default=100, help='How often to save the model')
    parser.add_argument('--model-update-interval', type=int, default=10e3, help='How often to create a new model')
    # parser.add_argument('--model-save-interval', type=int, default=20, help='How often to save the model')
    # parser.add_argument('--model-update-interval', type=int, default=60, help='How often to create a new model')
    parser.add_argument('--model-save-path', type=str, default='./logs/experiment',
                        help='The path to save the trained model')
    parser.add_argument('--custom', type=str, help='Test name', default='time_series')
    parser.add_argument('--container-size', type=int, help='Container size', nargs="*", default=[120, 120, 100])
    parser.add_argument('--dataset-path', type=str, help='The path to load dataset')

    args = parser.parse_args()
    args.no_cuda = args.no_cuda | (not jt.has_cuda)

    if args.no_cuda:
        args.device = 'cpu'

    # args.container_size = int(args.container_size) if not args.continuous else args.container_size
    # args.container_size = 3 * (args.container_size,)
    args.env_id = '3dBP-Discrete-v0' if not args.continuous else '3dBP-Continuous-v0'
    # args.item_size_set = DiscreteBoxData(lower=args.min_item_size, higher=args.max_item_size)
    args.item_size_set = [[43, 20,  7],[45, 30,  6]]

    # Custom presets only apply to discrete mode (they override container_size for specific datasets)
    if not args.continuous:
        if args.custom == 'time_series':
            args.container_size = [134, 125, 100]
            args.threshold = 0.02
        elif args.custom == 'occupancy':
            args.container_size = [120, 100, 170]
            args.threshold = 0.02
        elif args.custom == 'flat_long':
            args.container_size = [250, 120, 100]
            args.threshold = 0.04
        else:
            args.threshold = 0.02  # default for discrete mode without a preset
    else:
        # Continuous mode: threshold is used as the ratio threshold for adv policy
        # Use a reasonable default; it will be dynamically updated during training anyway
        args.threshold = 0.5

    args.normFactor = 1. / max(args.container_size)

    return args


def DecodeObs4Place(observation, bin_node_len, box_node_len, candidata_node_len, node_dim):
    # Accept flattened observation (B, obs_dim)
    if len(observation.shape) == 2:
        bsz = observation.shape[0]
        obs_dim = observation.shape[1]
        graph_size = bin_node_len + box_node_len + candidata_node_len
        expect = graph_size * node_dim
        if int(obs_dim) == int(expect):
            observation = observation.view(bsz, graph_size, node_dim)

    bin_node = observation[:, 0:bin_node_len, 0:node_dim-2]
    box_node = observation[:, bin_node_len:bin_node_len+box_node_len, 3:node_dim-2]
    candidata_node = observation[:, bin_node_len+box_node_len:, 0:node_dim-1]

    assert candidata_node.shape[1] == candidata_node_len
    assert observation[:, bin_node_len:bin_node_len+box_node_len, 0:3].sum() == 0

    full_mask = observation[:, :, -1]
    valid_mask = observation[:, bin_node_len+box_node_len:, node_dim-2]

    return bin_node, box_node, candidata_node, valid_mask, full_mask

def DecodeObs4Adv(observation, bin_node_len, next_item_num, node_dim):
    # Accept flattened observation (B, obs_dim)
    # NOTE: env observations may include candidate_pos nodes as well:
    #   obs_dim = (bin_node_len + next_item_num + candidate_pos_num) * node_dim
    if len(observation.shape) == 2:
        bsz = observation.shape[0]
        obs_dim = int(observation.shape[1])
        base_graph = bin_node_len + next_item_num
        if obs_dim % node_dim == 0:
            graph_size = obs_dim // node_dim
            # Only reshape when the flattened length is at least the base graph
            if graph_size >= base_graph:
                observation = observation.view(bsz, graph_size, node_dim)

    bin_node = observation[:, 0:bin_node_len, 0:node_dim-2]
    box_node = observation[:, bin_node_len:bin_node_len+next_item_num, 3:node_dim-2]

    assert observation[:, bin_node_len:bin_node_len+next_item_num, 0:3].sum() == 0

    # full_mask is per-node mask flag stored in last feature dim
    full_mask = observation[:, :, -1]
    # valid_mask is per-next-item flag stored in last feature dim for the next_item part
    valid_mask = observation[:, bin_node_len:bin_node_len+next_item_num, -1]

    return bin_node, box_node, valid_mask, full_mask

def DecodeObs4Critic(observation, bin_node_len, next_item_num, node_dim):
    # Accept flattened observation (B, obs_dim)
    if len(observation.shape) == 2:
        bsz = observation.shape[0]
        obs_dim = observation.shape[1]
        graph_size = bin_node_len + next_item_num
        expect = graph_size * node_dim
        if int(obs_dim) == int(expect):
            observation = observation.view(bsz, graph_size, node_dim)

    bin_node = observation[:, 0:bin_node_len, 0:node_dim-2]
    box_node = observation[:, bin_node_len:bin_node_len+next_item_num, 3:node_dim-2]

    assert observation[:, bin_node_len:bin_node_len+next_item_num, 0:3].sum() == 0

    full_mask = observation[:, 0:bin_node_len+next_item_num, -1]
    return bin_node, box_node, full_mask



def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr, minimum_lr=1e-4):
    """Decreases the learning rate linearly"""
    if epoch - 12000 >= 0:
        decay_ratio = epoch / float(total_num_epochs) if epoch < total_num_epochs else \
            float(total_num_epochs-1)/float(total_num_epochs)
        lr = initial_lr - (initial_lr * decay_ratio)
        lr = minimum_lr if lr < minimum_lr else lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def DiscreteBoxData(lower=1, higher=5, resolution=1):
    item_size_set = []
    for i in range(lower, higher + 1):
        for j in range(lower, higher + 1):
            for k in range(lower, higher + 1):
                item_size_set.append((i * resolution, j * resolution, k * resolution))
    return item_size_set

def registration_envs():
    """Register custom Gym envs.

    This may be called multiple times (e.g., once in the main process and again
    inside spawned subprocess workers). Gym raises if an id is registered twice,
    so we guard registration to make this function idempotent.
    """
    try:
        from gym.envs import registry as _registry  # gym<=0.21
        _spec = getattr(_registry, "env_specs", {})
        has_env = lambda _id: _id in _spec
    except Exception:
        # gym>=0.22
        try:
            from gym.envs.registration import registry as _registry  # type: ignore
            has_env = lambda _id: _id in _registry
        except Exception:
            has_env = lambda _id: False

    def _safe_register(_id: str, _entry_point: str):
        if has_env(_id):
            return
        try:
            register(id=_id, entry_point=_entry_point)
        except Exception as e:
            # If another call registered it just before us, ignore only that case.
            msg = str(e)
            if "Cannot re-register id" in msg or "re-register" in msg:
                return
            raise

    _safe_register('3dBP-Discrete-v0', '3dBP_envs.3dBP_Discrete0:PackingDiscrete')
    _safe_register('3dBP-Continuous-v0', '3dBP_envs.3dBP_Continuous0:PackingContinuous')
