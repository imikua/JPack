import os
import pickle
from shutil import copyfile, copytree

import jittor as jt
from jittor import nn
import numpy as np
from gym.envs.registration import register

# from   vis_boxes import vis_internal_nodes
# import vispy.scene
# import vispy.io


def orthogonal_init(tensor, gain=1.0):
    shape = tuple(tensor.shape)
    if len(shape) < 2:
        flat = np.ones(shape, dtype=np.float32)
    else:
        rows = shape[0]
        cols = int(np.prod(shape[1:]))
        flat_shape = (rows, cols)
        a = np.random.normal(0.0, 1.0, flat_shape)
        q, r = np.linalg.qr(a if rows >= cols else a.T)
        q = q if rows >= cols else q.T
        q = q.reshape(shape).astype(np.float32)
        flat = q * gain
    tensor.assign(jt.array(flat, dtype=tensor.dtype))


def constant_init(tensor, value=0.0, **kwargs):
    tensor.assign(jt.full(tensor.shape, value, dtype=tensor.dtype))


def init(module, weight_init, bias_init, gain=1):
    if hasattr(module, 'weight') and module.weight is not None:
        weight_init(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        bias_init(module.bias)
    return module


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        # NOTE: In Jittor, member vars starting with '_' are NOT treated as
        # parameters.  We must use a name without leading underscore so that
        # this bias is included in model.parameters() and gets updated.
        # Create a fresh leaf Var (not derived from any computation graph)
        # so that jt.grad / optimizer.backward can properly compute its gradient.
        import numpy as _np
        bias_data = bias.numpy().flatten().astype(_np.float32)
        self.bias_param = jt.array(bias_data.reshape(-1, 1))

    def execute(self, x):
        if len(x.shape) == 2:
            bias = self.bias_param.transpose((1, 0)).reshape((1, -1))
        elif len(x.shape) == 1:
            bias = self.bias_param.transpose((1, 0)).reshape((1, -1))
        elif len(x.shape) == 3:
            bias = self.bias_param.transpose((1, 0)).reshape((1, 1, -1))
        elif len(x.shape) == 4:
            bias = self.bias_param.transpose((1, 0)).reshape((1, -1, 1, 1))
        else:
            raise AssertionError('Unsupported input rank for AddBias.')

        return x + bias

    forward = execute


def backup(timeStr, args, upper_policy=None):
    if args.evaluate:
        targetDir = os.path.join('./logs/evaluation', timeStr)
    else:
        targetDir = os.path.join('./logs/experiment', timeStr)

    if not os.path.exists('./logs/runinfo'):
        os.makedirs('./logs/runinfo')

    if not os.path.exists(targetDir):
        os.makedirs(targetDir)

    for file in os.listdir('./'):
        if file.endswith('.py'):
            copyfile(file, os.path.join(targetDir, file))

    gymPath = './pct_envs'
    envName = args.id.split('-v')
    envName = envName[0] + envName[1]
    envPath = os.path.join(gymPath, envName)
    try:
        copytree(envPath, os.path.join(targetDir, envName))
    except:
        pass

    with open(os.path.join(targetDir, 'args.txt'), 'w') as f_args:
        for k, v in args.__dict__.items():
            f_args.write('%s:%s\n' % (k, v))

    if upper_policy is not None:
        safe_save(upper_policy.state_dict(), os.path.join(args.model_save_path, timeStr, 'upper-first-' + timeStr + '.pkl'))

    with open(os.path.join(targetDir, 'args.txt'), 'w') as f_args:
        for k, v in args.__dict__.items():
            f_args.write('%s:%s\n' % (k, v))


# Parsing PCT node from state returned in environment
def get_leaf_nodes(observation, internal_node_holder, leaf_node_holder):
    unify_obs = observation.reshape((observation.shape[0], -1, 9))
    leaf_nodes = unify_obs[:, internal_node_holder:internal_node_holder + leaf_node_holder, :]
    return unify_obs, leaf_nodes


def get_leaf_nodes_with_factor(observation, batch_size, internal_node_holder, leaf_node_holder):
    unify_obs = observation.reshape((batch_size, -1, 9))
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
    current_box = observation[:, internal_node_holder + leaf_node_holder:, 0:6]
    valid_flag = observation[:, internal_node_holder: internal_node_holder + leaf_node_holder, 8]
    full_mask = observation[:, :, -1]
    return internal_nodes, leaf_nodes, current_box, valid_flag, full_mask


def recursive_plot_ems(cur_node, ax, x, y, dx, dy, rec_cnt):
    if rec_cnt > 6:
        return
    if cur_node.sub_container:
        is_subc = 'True'
    else:
        is_subc = 'False'
    info = 'nodes: ' + str(cur_node.tree_node_num) + '\n' + \
            'children: ' + str(cur_node.valid_children) + '\n' + \
            'objs: ' + str(len(cur_node.placed_items)) + '\n' + \
            'is_subc: ' + is_subc + '\n' + \
            'lnodes' + str(len(cur_node.leaf_nodes)) + '\n'
    cnt = 0
    for item in cur_node.placed_items:
        tmp = list(item)
        for i in range(len(item)):
            tmp[i] = round(item[i], 3)
        if cnt % 2:
            info = info + str(tmp) + '\n'
        else:
            info = info + str(tmp) + ' '
        cnt += 1

    ax.text(x, y, info, bbox=dict(facecolor='white', alpha=1), ha='center', va='center')

    if cur_node.children:
        next_dx = dx / len(cur_node.children)
        next_x = x - dx / 2 + next_dx / 2

        next_y = y - dy
        for child in cur_node.children:
            ax.plot([x, next_x], [y, next_y], 'k-')
            recursive_plot_ems(child, ax, next_x, next_y, next_dx, dy, rec_cnt + 1)
            next_x += next_dx


def load_policy(load_path, upper_policy, args):
    print(load_path)
    assert os.path.exists(load_path), 'File does not exist'
    pretrained_state_dict = safe_load(load_path)
    if len(pretrained_state_dict) == 2:
        pretrained_state_dict, ob_rms = pretrained_state_dict

    load_dict = {}
    for k, v in pretrained_state_dict.items():
        if 'actor.embedder' in k:
            load_dict[k.replace('module.weight', 'weight')] = v
        else:
            load_dict[k.replace('module.', '')] = v
    load_dict = {k.replace('add_bias.', ''): v for k, v in load_dict.items()}
    load_dict = {k.replace('_bias', 'bias'): v for k, v in load_dict.items()}
    if args.model_architecture == 'PCT':
        load_dict = {k.replace('critic', 'critic_moduel'): v for k, v in load_dict.items()}
        load_dict = {k.replace('critic_moduel_moduel', 'critic_moduel'): v for k, v in load_dict.items()}
        load_dict = {k.replace('critic_moduel_distill', 'critic_distill'): v for k, v in load_dict.items()}

    remapped = {}
    for k, v in load_dict.items():
        if k == 'critic.bias' or k == 'critic_moduel.bias':
            remapped['critic_moduel.value_bias'] = v
        elif k == 'critic_distill.bias':
            remapped['critic_distill.value_bias'] = v
        else:
            remapped[k] = v
    load_dict = remapped

    for k, v in load_dict.items():
        if hasattr(v, 'shape') and k.endswith('value_bias'):
            if len(v.shape) == 0:
                load_dict[k] = v.reshape((1, 1))
            elif len(v.shape) == 1:
                load_dict[k] = v.reshape((-1, 1))
            else:
                load_dict[k] = v
        elif hasattr(v, 'shape') and len(v.shape) <= 3 and v.shape[-1] == 1:
            load_dict[k] = v.squeeze(-1)

    upper_policy.load_parameters(load_dict)

    print('Loading pre-train upper model', load_path)

    return upper_policy


def draw(internal_nodes, img_idx, img_path, root_path, container_size=[1, 1, 1], container_position=[0, 0, 0], bin_size=[1, 1, 1]):
    if not os.path.exists(os.path.join(root_path, str(img_idx))):
        os.makedirs(os.path.join(root_path, str(img_idx)))
    img_path = os.path.join(root_path, str(img_idx), img_path)
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=False, size=(1920, 1080))
    view = canvas.central_widget.add_view()
    camera = vispy.scene.cameras.TurntableCamera(fov=60,
                                                 center=container_position,
                                                 distance=2 * np.linalg.norm(container_size))
    view.camera = camera
    vis_internal_nodes(internal_nodes, view, container_size=container_size, container_position=container_position, bin_size=bin_size)
    img = canvas.render()
    vispy.io.write_png(img_path, img)
    canvas.close()


_REGISTERED_ENVS = False


def registration_envs():
    global _REGISTERED_ENVS
    if _REGISTERED_ENVS:
        return
    register(
        id='PctDiscrete-v2',
        entry_point='pct_envs.PctDiscrete2:PackingDiscrete',
    )
    register(
        id='PctDiscrete-v0',
        entry_point='pct_envs.PctDiscrete0:PackingDiscrete',
    )
    register(
        id='PctDiscrete-v3',
        entry_point='pct_envs.PctDiscrete3:PackingDiscrete',
    )
    register(
        id='PctContinuous-v0',
        entry_point='pct_envs.PctContinuous2:PackingContinuous',
    )
    register(
        id='PctContinuous-v2',
        entry_point='pct_envs.PctContinuous2:PackingContinuous'
    )
    _REGISTERED_ENVS = True


def safe_save(state_dict, path):
    save_obj = {}
    for k, v in state_dict.items():
        if hasattr(v, 'numpy'):
            save_obj[k] = v.numpy()
        else:
            save_obj[k] = v
    with open(path, 'wb') as f:
        pickle.dump(save_obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def safe_load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    load_obj = {}
    for k, v in obj.items():
        if isinstance(v, np.ndarray):
            load_obj[k] = jt.array(v)
        else:
            load_obj[k] = v
    return load_obj
