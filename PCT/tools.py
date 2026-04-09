import os
import jittor as jt
from shutil import copyfile, copytree
import jittor.nn as nn
import argparse
import givenData
import numpy as np
import pickle
from gym.envs.registration import register


def safe_save(state_dict, path):
    """Save state_dict using pickle (avoids jt.save's internal torch import)."""
    data = {}
    for k, v in state_dict.items():
        if isinstance(v, jt.Var):
            data[k] = v.data
        elif isinstance(v, np.ndarray):
            data[k] = v
        else:
            data[k] = v
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def safe_load(path):
    """Load state_dict from pickle or jt.load format."""
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception:
        return jt.load(path)

def orthogonal_init(weight, gain=1.0):
    """Implement orthogonal initialization for Jittor (not built-in)."""
    shape = weight.shape
    if len(shape) < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")
    rows = shape[0]
    cols = int(np.prod(shape[1:]))
    flat_shape = (rows, cols) if rows > cols else (cols, rows)
    a = np.random.normal(0.0, 1.0, flat_shape)
    q, r = np.linalg.qr(a)
    d = np.diag(r)
    ph = np.sign(d)
    q *= ph
    if rows < cols:
        q = q.T
    q = q.reshape(shape)
    q *= gain
    weight.assign(jt.array(q.astype(np.float32)))
    return weight

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight, gain=gain)
    bias_init(module.bias)
    return module

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        # Keep bias as an explicitly registered module field. In Jittor, Vars
        # stored on non-underscore attributes are treated as parameters by the
        # module parameter traversal, while `_bias` can be skipped.
        bias_np = np.array(bias.data if isinstance(bias, jt.Var) else bias, dtype=np.float32)
        # Match the torch layout used by the original KFAC code: [out_dim, 1].
        # KFAC expects AddBias gradients to be a 2D matrix so the Fisher
        # preconditioning matmul keeps the same semantics as PyTorch.
        self.bias = jt.array(bias_np.reshape((-1, 1)))

    def execute(self, x):
        if x.ndim == 2:
            bias = self.bias.transpose(0, 1).reshape(1, -1)
        elif x.ndim == 1:
            bias = self.bias.transpose(0, 1).reshape(1, -1)
        elif x.ndim == 3:
            bias = self.bias.transpose(0, 1).reshape(1, 1, -1)
        else:
            assert False

        return x + bias

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def backup(timeStr, args, upper_policy = None):
    if args.evaluate:
        targetDir = os.path.join('./logs/evaluation', timeStr)
    else:
        targetDir = os.path.join('./logs/experiment', timeStr)

    if not os.path.exists(targetDir):
        os.makedirs(targetDir)
    copyfile('attention_model.py', os.path.join(targetDir, 'attention_model.py'))
    copyfile('distributions.py',    os.path.join(targetDir, 'distributions.py'))
    copyfile('envs.py',    os.path.join(targetDir, 'envs.py'))
    copyfile('evaluation.py', os.path.join(targetDir, 'evaluation.py'))
    copyfile('evaluation_tools.py', os.path.join(targetDir, 'evaluation_tools.py'))
    copyfile('givenData.py',    os.path.join(targetDir, 'givenData.py'))
    copyfile('graph_encoder.py', os.path.join(targetDir, 'graph_encoder.py'))
    copyfile('kfac.py',    os.path.join(targetDir, 'kfac.py'))
    copyfile('main.py',    os.path.join(targetDir, 'main.py'))
    copyfile('model.py',   os.path.join(targetDir, 'model.py'))
    copyfile('storage.py',   os.path.join(targetDir, 'storage.py'))
    copyfile('tools.py',   os.path.join(targetDir, 'tools.py'))
    copyfile('train_tools.py', os.path.join(targetDir, 'train_tools.py'))

    gymPath = './pct_envs'
    envName = args.id.split('-v')
    envName = envName[0] + envName[1]
    envPath = os.path.join(gymPath, envName)
    copytree(envPath, os.path.join(targetDir, envName))

    if upper_policy is not None:
        safe_save(upper_policy.state_dict(), os.path.join(args.model_save_path, timeStr, 'upper-first-' + timeStr + ".pt"))

# Parsing PCT node from state returned in environment
def get_leaf_nodes(observation, internal_node_holder, leaf_node_holder):
    unify_obs = observation.reshape((observation.shape[0], -1, 9))
    leaf_nodes = unify_obs[:, internal_node_holder:internal_node_holder + leaf_node_holder, :]
    return unify_obs, leaf_nodes

def get_leaf_nodes_with_factor(observation,  batch_size, internal_node_holder, leaf_node_holder):
    unify_obs = observation.reshape((batch_size, -1, 9))
    # unify_obs[:, :, 0:6] *= factor
    leaf_nodes = unify_obs[:, internal_node_holder:internal_node_holder + leaf_node_holder, :]
    return unify_obs, leaf_nodes

'''
Parsing the raw state returned in environment:

internal_nodes    : A packed item vector, [x1, y1, z1, x2, y2, z2, density(optional) ]
                    x1, y1, z1 are coordinates of a packed item
                    x2 = x1 + x, y2 = y1 + y, z2 = z1 + z
                    x, y, z are sizes of a packed item (a little different from the original paper,
                    these two description have similar performance.).
leaf_nodes        : A placement vector, [x1, y1, z1, x2, y2, z2]
                    x1, y1, z1 are coordinates of a placement.
                    x2 = x1 + x, y2 = y1 + y, z2 = z1 + z
                    x, y, z are  sizes of the current item after an axis-aligned orientation (a little different from the original paper,
                    these two description have similar performance.).
next_item         : The next item to be packed [density(optional), 0, 0,x, y, z]
                    x, y, z are  sizes of the current item.
invalid_leaf_nodes: The mask which indicates whether this placement is feasible.
full_mask         : The mask which indicates whether this node should be encode by GAT.
'''
def observation_decode_leaf_node(observation, internal_node_holder, internal_node_length, leaf_node_holder):
    internal_nodes = observation[:, 0:internal_node_holder, 0:internal_node_length]
    leaf_nodes = observation[:, internal_node_holder:internal_node_holder + leaf_node_holder, 0:8]
    current_box = observation[:,internal_node_holder + leaf_node_holder:, 0:6]
    valid_flag = observation[:,internal_node_holder: internal_node_holder + leaf_node_holder, 8]
    full_mask = observation[:, :, -1]
    return internal_nodes, leaf_nodes, current_box, valid_flag, full_mask

def load_policy(load_path, upper_policy):
    print(load_path)
    assert os.path.exists(load_path), 'File does not exist'
    pretrained_state_dict = safe_load(load_path)
    if isinstance(pretrained_state_dict, (list, tuple)) and len(pretrained_state_dict) == 2:
        pretrained_state_dict, ob_rms = pretrained_state_dict

    load_dict = {}
    for k, v in pretrained_state_dict.items():
        if isinstance(v, np.ndarray):
            v = jt.array(v)
        elif not isinstance(v, jt.Var):
            v = jt.array(np.array(v))
        if 'actor.embedder.layers' in k:
            load_dict[k.replace('module.weight', 'weight')] = v
        else:
            load_dict[k.replace('module.', '')] = v

    load_dict = {k.replace('add_bias.', ''): v for k, v in load_dict.items()}
    load_dict = {k.replace('_bias', 'bias'): v for k, v in load_dict.items()}
    remapped = {}
    for k, v in load_dict.items():
        if k == 'critic.bias':
            remapped['critic.value_bias'] = v
        else:
            remapped[k] = v
    load_dict = remapped
    for k, v in load_dict.items():
        if k.endswith('value_bias'):
            if v.ndim == 0:
                load_dict[k] = v.reshape((1, 1))
            elif v.ndim == 1:
                load_dict[k] = v.reshape((-1, 1))
            else:
                load_dict[k] = v
        elif v.ndim <= 3:
            load_dict[k] = v.squeeze(dim=-1)
    upper_policy.load_state_dict(load_dict)
    print('Loading pre-train upper model', load_path)
    return upper_policy

def get_args():
    parser = argparse.ArgumentParser(description='PCT arguments')
    parser.add_argument('--exp-name', type=str, default=None, help='Experiment name (if not provided, the program will prompt via stdin)')
    parser.add_argument('--setting', type=int, default=2, help='Experiment setting, please see our paper for details')
    parser.add_argument('--lnes', type=str, default='EMS', help='Leaf Node Expansion Schemes: EMS (recommend), EV, EP, CP, FC')
    parser.add_argument('--internal-node-holder', type=int, default=80, help='Maximum number of internal nodes')
    parser.add_argument('--leaf-node-holder', type=int, default=50, help='Maximum number of leaf nodes')
    parser.add_argument('--shuffle',type=bool, default=True, help='Randomly shuffle the leaf nodes')
    parser.add_argument('--no-shuffle', action='store_false', dest='shuffle', help='Disable shuffling the leaf nodes (recommended way to turn shuffle off)')
    parser.add_argument('--continuous', action='store_true', help='Use continuous enviroment, otherwise the enviroment is discrete')

    parser.add_argument('--no-cuda',action='store_true', help='Forbidden cuda')
    parser.add_argument('--device', type=int, default=0, help='Which GPU will be called')
    parser.add_argument('--seed',   type=int, default=4, help='Random seed')

    parser.add_argument('--use-acktr', type=bool, default=True, help='Use acktr, otherwise A2C (note: argparse type=bool is tricky; use --use-a2c to reliably disable)')
    parser.add_argument('--use-a2c', action='store_false', dest='use_acktr', help='Use A2C (Adam) optimizer instead of ACKTR')
    parser.add_argument('--num-processes', type=int, default=64, help='The number of parallel processes used for training')
    parser.add_argument('--num-steps', type=int, default=5, help='The rollout length for n-step training')
    parser.add_argument('--learning-rate', type=float, default=1e-6, metavar='η', help='Learning rate, only works for A2C')
    parser.add_argument('--actor-loss-coef',        type=float, default=1.0, help='The coefficient of actor loss')
    parser.add_argument('--critic-loss-coef',       type=float, default=1.0, help='The coefficient of critic loss')
    parser.add_argument('--max-grad-norm',          type=float, default=0.5, help='Max norm of gradients')
    parser.add_argument('--embedding-size',     type=int, default=64,  help='Dimension of input embedding')
    parser.add_argument('--hidden-size',        type=int, default=128, help='Dimension of hidden layers')
    parser.add_argument('--gat-layer-num',      type=int, default=1, help='The number GAT layers')
    parser.add_argument('--gamma', type=float, default=1.0, metavar='γ', help='Discount factor')

    parser.add_argument('--model-save-interval',    type=int,   default=200   , help='How often to save the model')
    parser.add_argument('--model-update-interval',  type=int,   default=20e3 , help='How often to create a new model')
    parser.add_argument('--model-save-path',type=str, default='./logs/experiment', help='The path to save the trained model')
    parser.add_argument('--print-log-interval',     type=int,   default=10, help='How often to print training logs')

    parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    parser.add_argument('--evaluation-episodes', type=int, default=100, metavar='N', help='Number of episodes evaluated')
    parser.add_argument('--load-model', action='store_true', help='Load the trained model')
    parser.add_argument('--model-path', type=str, help='The path to load model')
    parser.add_argument('--load-dataset', action='store_true', help='Load an existing dataset, otherwise the data is generated on the fly')
    parser.add_argument('--dataset-path', type=str, help='The path to load dataset')

    parser.add_argument('--sample-from-distribution', action='store_true', help='Sample continuous item size from a uniform distribution U(a,b), otherwise sample items from \'item_size_set\' in \'givenData.py\'')
    parser.add_argument('--sample-left-bound', type=float, metavar='a', help='The parametre a of distribution U(a,b)')
    parser.add_argument('--sample-right-bound', type=float, metavar='b', help='The parametre b of distribution U(a,b)')

    args = parser.parse_args()

    if args.no_cuda: args.device = 'cpu'

    args.container_size = givenData.container_size
    args.item_size_set  = givenData.item_size_set

    if args.sample_from_distribution and args.sample_left_bound is None:
        args.sample_left_bound = 0.1 * min(args.container_size)
    if args.sample_from_distribution and args.sample_right_bound is None:
        args.sample_right_bound = 0.5 * min(args.container_size)

    if args.continuous:
        args.id = 'PctContinuous-v0'
    else:
        args.id = 'PctDiscrete-v0'

    if args.setting == 1:
        args.internal_node_length = 6
    elif args.setting == 2:
        args.internal_node_length = 6
    elif args.setting == 3:
        args.internal_node_length = 7
    if args.evaluate:
        args.num_processes = 1
    args.normFactor = 1.0 / np.max(args.container_size)

    return args

def get_args_heuristic():
    parser = argparse.ArgumentParser(description='Heuristic baseline arguments')

    parser.add_argument('--continuous', action='store_true', help='Use continuous enviroment, otherwise the enviroment is discrete')
    parser.add_argument('--setting', type=int, default=2, help='Experiment setting, please see our paper for details')
    # parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of episodes evaluated')
    parser.add_argument('--load-dataset', action='store_true', help='Load an existing dataset, otherwise the data is generated on the fly')
    parser.add_argument('--dataset-path', type=str, help='The path to load dataset')
    parser.add_argument('--heuristic', type=str, default='LSAH', help='Options: LSAH DBL MACS OnlineBPH HM BR RANDOM')


    args = parser.parse_args()
    args.container_size = givenData.container_size
    args.item_size_set  = givenData.item_size_set
    args.evaluate = True

    if args.continuous:
        assert args.heuristic == 'LSAH' or args.heuristic == 'OnlineBPH' or args.heuristic == 'BR', 'only LSAH, OnlineBPH, and BR allowed for continuous environment'

    if args.setting == 1:
        args.internal_node_length = 6
    elif args.setting == 2:
        args.internal_node_length = 6
    elif args.setting == 3:
        args.internal_node_length = 7
    if args.evaluate:
        args.num_processes = 1
    args.normFactor = 1.0 / np.max(args.container_size)

    return args

def registration_envs():
    register(
        id='PctDiscrete-v0',                                  # Format should be xxx-v0, xxx-v1
        entry_point='pct_envs.PctDiscrete0:PackingDiscrete',  # Expalined in envs/__init__.py
    )
    register(
        id='PctContinuous-v0',
        entry_point='pct_envs.PctContinuous0:PackingContinuous',
    )
