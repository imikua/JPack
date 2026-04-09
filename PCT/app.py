import numpy as np
import copy
import matplotlib.pyplot as plt
from . import ems_tools as ET
from .container  import Container
from packer.pct_model.model import DRL_GAT
from packer.pct_model.tools import *
import functools

def get_policy(method):
    # LSAH   MACS   RANDOM  OnlineBPH   DBL  BR
    if method == 'left_bottom':
        PCT_policy = 'heuristic'
            
    elif method == 'heightmap_min':
        PCT_policy = 'heuristic'
        
    elif method == 'pct':
        PCT_policy =  DRL_GAT()
        # Load the trained model
        model_path = '/media/wzf/goodluck/Workspace/rl/PCT-5-5-10-2024.06.05-16-35-33_2024.06.06-15-20-11.pt'
        PCT_policy = load_policy(model_path, PCT_policy)
        print('Pre-train model loaded!', model_path)
        PCT_policy.eval()
        
    elif method == 'LSAH':
        PCT_policy = 'heuristic'
        
    elif method == 'MACS':
        PCT_policy = 'heuristic'
    
    elif method == 'RANDOM':
        PCT_policy = 'heuristic'
    
    elif method == 'OnlineBPH':
        PCT_policy = 'heuristic'
        
    elif method == 'DBL':
        PCT_policy = 'heuristic'
        
    elif method == 'BR':
        PCT_policy = 'heuristic'    
    
    return PCT_policy


def deep_left_bottom(x,y):
    ''' use for sort function
        x: [ id, [x,y,z] ]
        y: [ id, [x,y,z] ]
        比较两个EMS的位置, 并根据它们的x、y、z坐标生成一个排序值, 确保盒子尽可能地放置在容器的底部且尽可能地向左和向下摆放
        
        example:
            sort_pos = sorted( enumerate(pos_list), key=functools.cmp_to_key(deep_left_bottom))
    '''
    a = x[1]
    b = y[1]
    diff = np.sign(a - b)
    return diff[2] * 100 + diff[0] * 10 + diff[1] * 1


def left_bottom(container, box):

    origin_ems, ems, ems_mask = container.get_ems()     # 获取容器的空余空间信息，包括原始的EMS、计算后的EMS和EMS掩码
    ems_size_mask = ET.compute_box_ems_mask(box[None,:], origin_ems, True)  # 将box的尺寸与容器的原始EMS进行比较，计算出一个EMS掩码。这个掩码用于确定容器中哪些区域足够大，可以放置新盒子

    valid_ems_ids = np.where(ems_size_mask == 1)[0]     # 得到有效的EMS ID
    valid_ems = origin_ems[valid_ems_ids]       # 有效EMS的位置

    list_ems = list(valid_ems)
    sort_ems = sorted( enumerate(list_ems), key=functools.cmp_to_key(deep_left_bottom))

    if len(sort_ems) == 0:
        return None

    ems_id = valid_ems_ids[sort_ems[0][0]]        
    pos = origin_ems[ems_id][:3]
    return pos


def heightmap_min(env):
    bin_size = env.bin_size
    bestScore = 1e10
    bestAction = []

    next_box = env.next_box
    next_den = env.next_den

    for lx in range(bin_size[0] - next_box[0] + 1):
        for ly in range(bin_size[1] - next_box[1] + 1):
            # Find the most suitable placement within the allowed orientation.
            for rot in range(env.orientation):
                if rot == 0:
                    x, y, z = next_box
                elif rot == 1:
                    y, x, z = next_box
                elif rot == 2:
                    z, x, y = next_box
                elif rot == 3:
                    z, y, x = next_box
                elif rot == 4:
                    x, z, y = next_box
                elif rot == 5:
                    y, z, x = next_box

                # Check the feasibility of this placement
                feasible, heightMap = env.space.drop_box_virtual([x, y, z], (lx, ly), False,
                                                                    next_den, env.setting, False, True)
                if not feasible:
                    continue

                # Score the given placement.
                score = lx + ly + 100 * np.sum(heightMap)
                if score < bestScore:
                    bestScore = score
                    env.next_box = [x, y, z]
                    bestAction = [rot, lx, ly]


    if len(bestAction) != 0:
        rec = env.space.plain[bestAction[1]:bestAction[1] + env.next_box[0], bestAction[2]:bestAction[2] + env.next_box[1]]
        lz = np.max(rec)
        # Place this item in the environment with the best action.
        env.step(bestAction)
        done = False
    else:
        # No feasible placement, this episode is done.
        lz = None
        done = True

    return  done, bestAction, lz


def LASH(env, maxXY, minXY):
    '''
    Solving a new 3D bin packing problem with deep reinforcement learning method.
    https://arxiv.org/abs/1708.05930 
    '''
    bin_size = env.bin_size

    # maxXY = [0,0]
    # minXY = [bin_size[0], bin_size[1]]


    bestScore = bin_size[0] * bin_size[1] + bin_size[1] * bin_size[2] + bin_size[2] * bin_size[0]
    EMS = env.space.EMS

    bestAction = None
    next_box = env.next_box
    next_den = env.next_den

    for ems in EMS:
        # Find the most suitable placement within the allowed orientation.
        if np.sum(np.abs(ems)) == 0:
            continue
        for rot in range(env.orientation):
            if rot == 0:
                x, y, z = next_box
            elif rot == 1:
                y, x, z = next_box
            elif rot == 2:
                z, x, y = next_box
            elif rot == 3:
                z, y, x = next_box
            elif rot == 4:
                x, z, y = next_box
            elif rot == 5:
                y, z, x = next_box

            if ems[3] - ems[0] >= x and ems[4] - ems[1] >= y and ems[5] - ems[2] >= z:
                lx, ly = ems[0], ems[1]
                # Check the feasibility of this placement
                feasible, height = env.space.drop_box_virtual([x, y, z], (lx, ly), False,
                                                            next_den, env.setting, returnH=True)

                if feasible:
                    score = (max(lx + x, maxXY[0]) - min(lx, minXY[0])) * (
                                max(ly + y, maxXY[1]) - min(ly, minXY[1])) \
                            + (height + z) * (max(ly + y, maxXY[1]) - min(ly, minXY[1])) \
                            + (height + z) * (max(lx + x, maxXY[0]) - min(lx, minXY[0]))

                    # The placement which keeps pack items with less surface area is better.
                    if score < bestScore:
                        bestScore = score
                        env.next_box = [x, y, z]
                        bestAction = [rot, lx, ly, height, ems[3] - ems[0], ems[4] - ems[1], ems[5] - ems[2]]

                    elif score == bestScore and bestAction is not None:
                        if min(ems[3] - ems[0] - x, ems[4] - ems[1] - y, ems[5] - ems[2] - z) < \
                                min(bestAction[4] - x, bestAction[5] - y, bestAction[6] - z):
                            env.next_box = [x, y, z]
                            bestAction = [rot, lx, ly, height, ems[3] - ems[0], ems[4] - ems[1], ems[5] - ems[2]]
    
    if bestAction is not None:
        x, y, _ = env.next_box
        _, lx, ly, _, _, _, _ = bestAction

        if lx + x > maxXY[0]: maxXY[0] = lx + x
        if ly + y > maxXY[1]: maxXY[1] = ly + y
        if lx < minXY[0]: minXY[0] = lx
        if ly < minXY[1]: minXY[1] = ly
        
        rec = env.space.plain[bestAction[1]:bestAction[1] + env.next_box[0], bestAction[2]:bestAction[2] + env.next_box[1]]
        lz = np.max(rec)
        # Place this item in the environment with the best action.
        _, _, done, _ = env.step(bestAction[0:3])
    else:
        # No feasible placement, this episode is done.
        lz = None
        done = True

    return done, bestAction, lz, maxXY, minXY


def MACS(env):
    '''
    Tap-net: transportand-pack using reinforcement learning.
    https://dl.acm.org/doi/abs/10.1145/3414685.3417796
    '''
    def calc_maximal_usable_spaces(ctn, H):
        '''
        Score the given placement.
        This score function comes from https://github.com/Juzhan/TAP-Net/blob/master/tools.py
        '''
        score = 0
        for h in range(H):
            level_max_empty = 0
            # build the histogram map
            hotmap = (ctn[:, :, h] == 0).astype(int)
            histmap = np.zeros_like(hotmap).astype(int)
            for i in reversed(range(container_size[0])):
                for j in range(container_size[1]):
                    if i==container_size[0]-1: histmap[i, j] = hotmap[i, j]
                    elif hotmap[i, j] == 0: histmap[i, j] = 0
                    else: histmap[i, j] = histmap[i+1, j] + hotmap[i, j]

            # scan the histogram map
            for i in range(container_size[0]):
                for j in range(container_size[1]):
                    if histmap[i, j] == 0: continue
                    if j>0 and histmap[i, j] == histmap[i, j-1]: continue
                    # look right
                    for j2 in range(j, container_size[1]):
                        if j2 == container_size[1] - 1: break
                        if histmap[i, j2+1] < histmap[i, j]: break
                    # look left
                    for j1 in reversed(range(0, j+1)):
                        if j1 == 0: break
                        if histmap[i, j1-1] < histmap[i, j]: break
                    area = histmap[i, j] * (j2 - j1 + 1)
                    if area > level_max_empty: level_max_empty = area
            score += level_max_empty
        return score

    def update_container(ctn, pos, boxSize):
        _x, _y, _z = pos
        block_x, block_y, block_z = boxSize
        ctn[_x:_x+block_x, _y:_y+block_y, _z:_z+block_z] = block_index + 1
        under_space = ctn[_x:_x+block_x, _y:_y+block_y, 0:_z]
        ctn[_x:_x+block_x, _y:_y+block_y, 0:_z][ under_space==0 ] = -1


    container_size = env.bin_size
    container = np.zeros(env.bin_size)

    bestScore = -1e10
    EMS = env.space.EMS

    bestAction = None
    next_box = env.next_box
    next_den = env.next_den

    for ems in EMS:
        # Find the most suitable placement within the allowed orientation.
        for rot in range(env.orientation):
            if rot == 0:
                x, y, z = next_box
            elif rot == 1:
                y, x, z = next_box
            elif rot == 2:
                z, x, y = next_box
            elif rot == 3:
                z, y, x = next_box
            elif rot == 4:
                x, z, y = next_box
            elif rot == 5:
                y, z, x = next_box

            if ems[3] - ems[0] >= x and ems[4] - ems[1] >= y and ems[5] - ems[2] >= z:
                for corner in range(4):
                    if corner == 0:
                        lx, ly = ems[0], ems[1]
                    elif corner == 1:
                        lx, ly = ems[3] - x, ems[1]
                    elif corner == 2:
                        lx, ly = ems[0], ems[4] - y
                    elif corner == 3:
                        lx, ly = ems[3] - x, ems[4] - y

                    # Check the feasibility of this placement
                    feasible, height = env.space.drop_box_virtual([x, y, z], (lx, ly), False,
                                                                        next_den, env.setting, returnH=True)
                    if feasible:
                        updated_containers = container.copy()
                        update_container(updated_containers, np.array([lx, ly, height]), np.array([x, y, z]))
                        score = calc_maximal_usable_spaces(updated_containers, height)

                        if score > bestScore:
                            bestScore = score
                            env.next_box = [x, y, z]
                            bestAction = [0, lx, ly, height]

    if bestAction is not None:
        # Place this item in the environment with the best action.
        update_container(container, bestAction[1:4], env.next_box)
        _, _, done, _ = env.step(bestAction[0:3])
    else:
        # No feasible placement, this episode is done.
        done = True

    return  np.mean(episode_utilization), np.var(episode_utilization), np.mean(episode_length)


def random(env, times = 2000):
    '''
    Randomly pick placements from full coordinates.
    '''
    bin_size = env.bin_size

    next_box = env.next_box
    next_den = env.next_den

    # Check the feasibility of all placements.
    candidates = []
    for lx in range(bin_size[0] - next_box[0] + 1):
        for ly in range(bin_size[1] - next_box[1] + 1):
            for rot in range(env.orientation):
                if rot == 0:
                    x, y, z = next_box
                elif rot == 1:
                    y, x, z = next_box
                elif rot == 2:
                    z, x, y = next_box
                elif rot == 3:
                    z, y, x = next_box
                elif rot == 4:
                    x, z, y = next_box
                elif rot == 5:
                    y, z, x = next_box

                feasible, heightMap = env.space.drop_box_virtual([x, y, z], (lx, ly), False,
                                                                    next_den, env.setting, False, True)
                if not feasible:
                    continue

                candidates.append([[x, y, z], [0, lx, ly]])

    if len(candidates) != 0:
        # Pick one placement randomly from all possible placements
        idx = np.random.randint(0, len(candidates))
        env.next_box = candidates[idx][0]
        env.step(candidates[idx][1])
        done = False
    else:
        # No feasible placement, this episode is done.
        done = True

    return  np.mean(episode_utilization), np.var(episode_utilization), np.mean(episode_length)


def OnlineBPH(env, times = 2000):
    '''
    An Online Packing Heuristic for the Three-Dimensional Container Loading
    Problem in Dynamic Environments and the Physical Internet
    https://doi.org/10.1007/978-3-319-55792-2\_10
    '''

    # Sort the ems placement with deep-bottom-left order.
    EMS = env.space.EMS
    EMS = sorted(EMS, key=lambda ems: (ems[2], ems[1], ems[0]), reverse=False)

    bestAction = None
    next_box = env.next_box
    next_den = env.next_den
    stop = False


    for ems in EMS:
        # Find the first suitable placement within the allowed orientation.
        if np.sum(np.abs(ems)) == 0:
            continue
        for rot in range(env.orientation):
            if rot == 0:
                x, y, z = next_box
            elif rot == 1:
                y, x, z = next_box
            elif rot == 2:
                z, x, y = next_box
            elif rot == 3:
                z, y, x = next_box
            elif rot == 4:
                x, z, y = next_box
            elif rot == 5:
                y, z, x = next_box

            # Check the feasibility of this placement
            if env.space.drop_box_virtual([x, y, z], (ems[0], ems[1]), False, next_den, env.setting):
                env.next_box = [x, y, z]
                bestAction = [0, ems[0], ems[1]]
                stop = True
                break
        if stop: break

    if bestAction is not None:
        # Place this item in the environment with the best action.
        _, _, done, _ = env.step(bestAction)
    else:
        # No feasible placement, this episode is done.
        done = True

    return np.mean(episode_utilization), np.var(episode_utilization), np.mean(episode_length)


def DBL(env, times = 2000):
    '''
    A Hybrid Genetic Algorithm for Packing in 3D with Deepest Bottom Left with Fill Method
    https://doi.org/10.1007/978-3-540-30198-1\_45
    '''
    bin_size = env.bin_size

    bestScore = 1e10
    bestAction = []

    next_box = env.next_box
    next_den = env.next_den

    for lx in range(bin_size[0] - next_box[0] + 1):
        for ly in range(bin_size[1] - next_box[1] + 1):
            # Find the most suitable placement within the allowed orientation.
            for rot in range(env.orientation):
                if rot == 0:
                    x, y, z = next_box
                elif rot == 1:
                    y, x, z = next_box
                elif rot == 2:
                    z, x, y = next_box
                elif rot == 3:
                    z, y, x = next_box
                elif rot == 4:
                    x, z, y = next_box
                elif rot == 5:
                    y, z, x = next_box

                # Check the feasibility of this placement
                feasible, height = env.space.drop_box_virtual([x, y, z], (lx, ly), False,
                                                                    next_den, env.setting, True, False)
                if not feasible:
                    continue

                # Score the given placement.
                score = lx + ly + 100 * height
                if score < bestScore:
                    bestScore = score
                    env.next_box = [x, y, z]
                    bestAction = [0, lx, ly]

    if len(bestAction) != 0:
        # Place this item in the environment with the best action.
        env.step(bestAction)
        done = False
    else:
        # No feasible placement, this episode is done.
        done = True

    return np.mean(episode_utilization), np.var(episode_utilization), np.mean(episode_length)


def BR(env, times = 2000):
    '''
    Online 3D Bin Packing with Constrained Deep Reinforcement Learning
    https://ojs.aaai.org/index.php/AAAI/article/view/16155
    '''
    def eval_ems(ems):
        # Score the given placement.
        s = 0
        valid = []
        for bs in env.item_set:
            bx, by, bz = bs
            if ems[3] - ems[0] >= bx and ems[4] - ems[1] >= by and ems[5] - ems[2] >= bz:
                valid.append(1)
        s += (ems[3] - ems[0]) * (ems[4] - ems[1]) * (ems[5] - ems[2])
        s += len(valid)
        if len(valid) == len(env.item_set):
            s += 10
        return s
            

    bestScore = -1e10
    EMS = env.space.EMS

    bestAction = None
    next_box = env.next_box
    next_den = env.next_den

    for ems in EMS:
        # Find the most suitable placement within the allowed orientation.
        for rot in range(env.orientation):
            if rot == 0:
                x, y, z = next_box
            elif rot == 1:
                y, x, z = next_box
            elif rot == 2:
                z, x, y = next_box
            elif rot == 3:
                z, y, x = next_box
            elif rot == 4:
                x, z, y = next_box
            elif rot == 5:
                y, z, x = next_box

            if ems[3] - ems[0] >= x and ems[4] - ems[1] >= y and ems[5] - ems[2] >= z:
                lx, ly = ems[0], ems[1]
                # Check the feasibility of this placement
                feasible, height = env.space.drop_box_virtual([x, y, z], (lx, ly), False,
                                                                    next_den, env.setting, returnH=True)
                if feasible:
                    score = eval_ems(ems)
                    if score > bestScore:
                        bestScore = score
                        env.next_box = [x, y, z]
                        bestAction = [0, lx, ly, height]


    if bestAction is not None:
        # Place this item in the environment with the best action.
        _, _, done, _ = env.step(bestAction[0:3])
    else:
        # No feasible placement, this episode is done.
        done = True

    return  np.mean(episode_utilization), np.var(episode_utilization), np.mean(episode_length)


def pct(PCT_policy, env, obs, args, eval_freq = 100, factor = 1):
    
    # obs = env.cur_observation()
    # obs = torch.FloatTensor(obs).unsqueeze(dim=0)

    all_nodes, leaf_nodes = get_leaf_nodes_with_factor(obs, 1, args.internal_node_holder, args.leaf_node_holder)
    batchX = torch.arange(1)

    with torch.no_grad():
        selectedlogProb, selectedIdx, policy_dist_entropy, value = PCT_policy(all_nodes, True, normFactor = factor)
    selected_leaf_node = leaf_nodes[batchX, selectedIdx.squeeze()]      # tensor([[ 8.,  0.,  0., 10.,  5., 10.,  0.,  0.,  1.]], device='cuda:0')
    action = selected_leaf_node.cpu().numpy()[0][0:6]
    now_action, box_size = env.LeafNode2Action(action)
    
    # check rot
    init_box_size = env.next_box
    if box_size[0] == init_box_size[0] and box_size[1] == init_box_size[1]:
        rot = 0
    else:
        rot = 1
    
    rec = env.space.plain[now_action[1]:now_action[1] + env.next_box[0], now_action[2]:now_action[2] + env.next_box[1]]
    lz = np.max(rec)
    
    obs, reward, done, infos = env.step(action)

    new_action = (rot,) + now_action[1:]
    # now_action[0] = rot
    
    if done:
        # obs = env.reset()
        pass
    
    return done, new_action, lz, obs



class Packer():
    def __init__(self, container_size) -> None:
        self.container = Container(container_size)
    
    def pack_box(self, real_box, box_unit):
        
        box = np.ceil(real_box / box_unit)
        pos = left_bottom( self.container, box )
        
        placeable = True
        rotation = [0, 0, 0, 1]
        
        return placeable, pos, rotation
    
    def pack_box_v2(self, env, method):
        
        if method == 'left_bottom':
            # box = np.ceil(env.bin_size / 0.03)
            # print(np.ceil(size / 0.03))
            print(env.next_box)
            pos = left_bottom( self.container, np.ceil(env.next_box) )
            # pos = left_bottom( self.container, np.ceil(size / 0.03) )
            placeable = True
            rotation = [0, 0, 0, 1]
            
        elif method == 'heightmap_min':
            action, lz = heightmap_min(env)
            if len(action) == 0:
                placeable = False
                return placeable, [], []
            lx, ly = action[1], action[2]
            pos = np.array([lx, ly, lz], dtype=np.float32)
            
            rotation_flag = action[0]
            if rotation_flag == 0:
                rotation = [0, 0, 0, 1]
            elif rotation_flag == 1:
                rotation = [0, 1, 0, 0]
       
            placeable = True

        return placeable, pos, rotation
    
    def pack_box_v3(self, env, obs, method, policy, args):
        infos = dict()
        infos['next_obs'] = ''
        
        if method == 'left_bottom':
            # box = np.ceil(env.bin_size / 0.03)
            # print(np.ceil(size / 0.03))
            print(env.next_box)
            pos = left_bottom( self.container, np.ceil(env.next_box) )
            # pos = left_bottom( self.container, np.ceil(size / 0.03) )
            placeable = True
            rotation_flag = 0
            
        elif method == 'heightmap_min':
            done, action, lz = heightmap_min(env)
            placeable = not done
            if placeable == False:
                return placeable, [], [], infos
            rotation_flag, lx, ly = action[0], action[1], action[2]
            pos = np.array([lx, ly, lz], dtype=np.float32)
       
            placeable = True
            
        elif method == 'pct':
            done, action, lz, next_obs = pct(policy, env, obs, args, eval_freq=100, factor=1)
            placeable = not done
            if placeable == False:
                return placeable, [], [], infos
            rotation_flag, lx, ly = action[0], action[1], action[2]
            pos = np.array([lx, ly, lz], dtype=np.float32)
            
            infos['next_obs'] = next_obs

        return placeable, pos, rotation_flag, infos
    
    
    def pack_box_v4(self, env, infos, obs, method, policy, args):
        # LSAH   MACS   RANDOM  OnlineBPH   DBL  BR
        
        if method == 'left_bottom':
            # box = np.ceil(env.bin_size / 0.03)
            # print(np.ceil(size / 0.03))
            print(env.next_box)
            pos = left_bottom( self.container, np.ceil(env.next_box) )
            # pos = left_bottom( self.container, np.ceil(size / 0.03) )
            placeable = True
            rotation_flag = 0
            infos['next_obs'] = ''
            
        elif method == 'heightmap_min':
            done, action, lz = heightmap_min(env)
            placeable = not done
            if placeable == False:
                return placeable, [], [], infos
            rotation_flag, lx, ly = action[0], action[1], action[2]
            pos = np.array([lx, ly, lz], dtype=np.float32)
            infos['next_obs'] = ''
            placeable = True
        
        elif method == 'LSAH':
            if len(env.packed) == 0:
                # 初始化
                maxXY = [0,0]
                minXY = [env.bin_size[0], env.bin_size[1]]
            else:
                maxXY = infos['maxXY']
                minXY = infos['minXY']
            
            done, action, lz, new_maxXY, new_minXY = LASH(env, maxXY, minXY)
            
            infos['maxXY'] = new_maxXY
            infos['minXY'] = new_minXY
            infos['next_obs'] = None
            
            placeable = not done
            if placeable == False:
                return placeable, [], [], infos
            rotation_flag, lx, ly = action[0], action[1], action[2]
            pos = np.array([lx, ly, lz], dtype=np.float32)

            placeable = True
        
        elif method == 'MACS':
            if len(env.packed) == 0:
                # 初始化
                maxXY = [0,0]
                minXY = [env.bin_size[0], env.bin_size[1]]
            else:
                maxXY = infos['maxXY']
                minXY = infos['minXY']
            
            done, action, lz, new_maxXY, new_minXY = LASH(env, maxXY, minXY)
            
            infos['maxXY'] = new_maxXY
            infos['minXY'] = new_minXY
            infos['next_obs'] = None
            
            placeable = not done
            if placeable == False:
                return placeable, [], [], infos
            rotation_flag, lx, ly = action[0], action[1], action[2]
            pos = np.array([lx, ly, lz], dtype=np.float32)

            placeable = True
            
        elif method == 'RANDOM':
            if len(env.packed) == 0:
                # 初始化
                maxXY = [0,0]
                minXY = [env.bin_size[0], env.bin_size[1]]
            else:
                maxXY = infos['maxXY']
                minXY = infos['minXY']
            
            done, action, lz, new_maxXY, new_minXY = LASH(env, maxXY, minXY)
            
            infos['maxXY'] = new_maxXY
            infos['minXY'] = new_minXY
            infos['next_obs'] = None
            
            placeable = not done
            if placeable == False:
                return placeable, [], [], infos
            rotation_flag, lx, ly = action[0], action[1], action[2]
            pos = np.array([lx, ly, lz], dtype=np.float32)

            placeable = True
            
        elif method == 'OnlineBPH':
            if len(env.packed) == 0:
                # 初始化
                maxXY = [0,0]
                minXY = [env.bin_size[0], env.bin_size[1]]
            else:
                maxXY = infos['maxXY']
                minXY = infos['minXY']
            
            done, action, lz, new_maxXY, new_minXY = LASH(env, maxXY, minXY)
            
            infos['maxXY'] = new_maxXY
            infos['minXY'] = new_minXY
            infos['next_obs'] = None
            
            placeable = not done
            if placeable == False:
                return placeable, [], [], infos
            rotation_flag, lx, ly = action[0], action[1], action[2]
            pos = np.array([lx, ly, lz], dtype=np.float32)

            placeable = True
            
        elif method == 'DBL':
            if len(env.packed) == 0:
                # 初始化
                maxXY = [0,0]
                minXY = [env.bin_size[0], env.bin_size[1]]
            else:
                maxXY = infos['maxXY']
                minXY = infos['minXY']
            
            done, action, lz, new_maxXY, new_minXY = LASH(env, maxXY, minXY)
            
            infos['maxXY'] = new_maxXY
            infos['minXY'] = new_minXY
            infos['next_obs'] = None
            
            placeable = not done
            if placeable == False:
                return placeable, [], [], infos
            rotation_flag, lx, ly = action[0], action[1], action[2]
            pos = np.array([lx, ly, lz], dtype=np.float32)

            placeable = True
            
        elif method == 'BR':
            if len(env.packed) == 0:
                # 初始化
                maxXY = [0,0]
                minXY = [env.bin_size[0], env.bin_size[1]]
            else:
                maxXY = infos['maxXY']
                minXY = infos['minXY']
            
            done, action, lz, new_maxXY, new_minXY = LASH(env, maxXY, minXY)
            
            infos['maxXY'] = new_maxXY
            infos['minXY'] = new_minXY
            infos['next_obs'] = None
            
            placeable = not done
            if placeable == False:
                return placeable, [], [], infos
            rotation_flag, lx, ly = action[0], action[1], action[2]
            pos = np.array([lx, ly, lz], dtype=np.float32)

            placeable = True
           
        elif method == 'pct':
            done, action, lz, next_obs = pct(policy, env, obs, args, eval_freq=100, factor=1)
            placeable = not done
            if placeable == False:
                return placeable, [], [], infos
            rotation_flag, lx, ly = action[0], action[1], action[2]
            pos = np.array([lx, ly, lz], dtype=np.float32)
            
            infos['next_obs'] = next_obs

        return placeable, pos, rotation_flag, infos
    
    
    def add_box(self, box, pos):
        self.container.add_new_box(box, pos)
    
