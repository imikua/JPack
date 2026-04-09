import numpy as np
import copy
import matplotlib.pyplot as plt
import os
import functools
import itertools
from . import ems_tools as ET
from .space import Space

# =================================== 

def show_corner(corners, heightmap):
    plt.scatter(corners[:,1], corners[:,0])
    plt.imshow(heightmap)
    plt.show()

def show_bridge(bridges, heightmap):
    plt.imshow(heightmap)
    for b in bridges:
        plt.scatter(b[0][1], b[0][0], c='b')
        plt.scatter(b[1][1], b[1][0], c='r')
    plt.show()

# ===============================

def calc_one_position_lb_greedy_3d(block, block_index, container_size,
                                container, positions, stable, heightmap, valid_size, empty_size):
    """
    calculate the latest block's position in the container by lb-greedy in 2D cases
    ---
    params:
    ---
        static params:
            block: int * 3 array, size of the block to pack
            block_index: int, index of the block to pack, previous were already packed
            container_size: 1 x 3 array, size of the container
            reward_type: string
                'C' / 'S' / 'P'
                'C+P'
                'C+P+S'
                just combination of C P S
            packing_strategy: string
        dynamic params:
            container: width * length * height array, the container state
            positions: int * 3 array, coordinates of the blocks, [0, 0] for blocks after block_index
            stable: n * 1 bool list, the blocks' stability state
            heightmap: width * length array, heightmap of the container
            valid_size: int, sum of the packed blocks' size
            empty_size: int, size of the empty space under packed blocks
    return:
    ---
        container: width * length * height array, updated container
        positions: int * 3 array, updated positions
        stable: n * 1 bool list, updated stable
        heightmap: width * length array, updated heightmap
        valid_size: int, updated valid_size
        empty_size: int, updated empty_size
    """
    block_dim = len(block)
    block_x, block_y, block_z = block
    valid_size += block_x * block_y * block_z

    # get empty-maximal-spaces list from heightmap
    # each ems represented as a left-bottom corner
    ems_list = []
    # hm_diff: height differences of neightbor columns, padding 0 in the front
    # x coordinate
    hm_diff_x = np.insert(heightmap, 0, heightmap[0, :], axis=0)
    hm_diff_x = np.delete(hm_diff_x, len(hm_diff_x)-1, axis=0)
    hm_diff_x = heightmap - hm_diff_x
    # y coordinate
    hm_diff_y = np.insert(heightmap, 0, heightmap[:, 0], axis=1)
    hm_diff_y = np.delete(hm_diff_y, len(hm_diff_y.T)-1, axis=1)
    hm_diff_y = heightmap - hm_diff_y

    # get the xy coordinates of all left-deep-bottom corners
    ems_x_list = np.array(np.nonzero(hm_diff_x)).T.tolist()
    ems_y_list = np.array(np.nonzero(hm_diff_y)).T.tolist()
    ems_xy_list = []
    ems_xy_list.append([0,0])
    for xy in ems_x_list:
        x, y = xy
        if y!=0 and [x, y-1] in ems_x_list:
            if heightmap[x, y] == heightmap[x, y-1] and \
                hm_diff_x[x, y] == hm_diff_x[x, y-1]:
                continue
        ems_xy_list.append(xy)
    for xy in ems_y_list:
        x, y = xy
        if x!=0 and [x-1, y] in ems_y_list:
            if heightmap[x, y] == heightmap[x-1, y] and \
                hm_diff_x[x, y] == hm_diff_x[x-1, y]:
                continue
        if xy not in ems_xy_list:
            ems_xy_list.append(xy)

    # sort by y coordinate, then x
    def y_first(pos): return pos[1]
    ems_xy_list.sort(key=y_first, reverse=False)

    # get ems_list
    for xy in ems_xy_list:
        x, y = xy
        if x+block_x > container_size[0] or \
            y+block_y > container_size[1]: continue
        z = np.max( heightmap[x:x+block_x, y:y+block_y] )
        ems_list.append( [ x, y, z ] )
    
    # firt consider the most bottom, sort by z coordinate, then y last x
    def z_first(pos): return pos[2]
    ems_list.sort(key=z_first, reverse=False)

    # if no ems found
    if len(ems_list) == 0:
        valid_size -= block_x * block_y * block_z
        stable[block_index] = False
        return container, positions, stable, heightmap, valid_size, empty_size

    # varients to store results of searching ems corners
    ems_num = len(ems_list)
    pos_ems = np.zeros((ems_num, block_dim)).astype(int)
    is_settle_ems  = [False] * ems_num
    is_stable_ems  = [False] * ems_num
    compactness_ems  = [0.0] * ems_num
    pyramidality_ems = [0.0] * ems_num
    stability_ems    = [0.0] * ems_num
    empty_ems = [empty_size] * ems_num
    under_space_mask  = [[]] * ems_num
    heightmap_ems = [np.zeros(container_size[:-1]).astype(int)] * ems_num
    visited = []

    # check if a position suitable
    def check_position(index, _x, _y, _z):
        # check if the pos visited
        if [_x, _y, _z] in visited: return
        if _z>0 and (container[_x:_x+block_x, _y:_y+block_y, _z-1]==0).all(): return
        visited.append([_x, _y, _z])
        if (container[_x:_x+block_x, _y:_y+block_y, _z] == 0).all():
            is_stable_ems[index] = True
            pos_ems[index] = np.array([_x, _y, _z])
            heightmap_ems[index][_x:_x+block_x, _y:_y+block_y] = _z + block_z
            is_settle_ems[index] = True

    # calculate socres
    def calc_C_P_S(index):
        _x, _y, _z = pos_ems[index]
        # compactness
        height = np.max(heightmap_ems[index])
        bbox_size = height * container_size[0] *container_size[1]
        compactness_ems[index] = valid_size / bbox_size
        # pyramidality
        under_space = container[_x:_x+block_x, _y:_y+block_y, 0:_z]
        under_space_mask[index] = under_space==0
        empty_ems[index] += np.sum(under_space_mask[index])
        # if 'P' in reward_type:
        #     pyramidality_ems[index] = valid_size / (empty_ems[index] + valid_size)
        # # stability
        # if 'S' in reward_type:
        #     stable_num = np.sum(stable[:block_index]) + np.sum(is_stable_ems[index])
        #     stability_ems[index] = stable_num / (block_index + 1)

    # search positions in each ems
    X = int(container_size[0] - block_x + 1)
    Y = int(container_size[1] - block_y + 1)
    for ems_index, ems in enumerate(ems_list):
        # using buttom-left strategy in each ems
        heightmap_ems[ems_index] = heightmap.copy()
        X0, Y0, _z = ems
        for _x, _y  in itertools.product( range(X0, X), range(Y0, Y) ):
            if is_settle_ems[ems_index]: break
            check_position(ems_index, _x, _y, _z)
        if is_settle_ems[ems_index]: calc_C_P_S(ems_index)

    # if the block has not been settled
    if np.sum(is_settle_ems) == 0:
        valid_size -= block_x * block_y * block_z
        stable[block_index] = False
        return container, positions, stable, heightmap, valid_size, empty_size

    # get the best ems
    ratio_ems = [c+p+s for c, p, s in zip(compactness_ems, pyramidality_ems, stability_ems)]
    best_ems_index = np.argmax(ratio_ems)
    while not is_settle_ems[best_ems_index]:
        ratio_ems.remove(ratio_ems[best_ems_index])
        best_ems_index = np.argmax(ratio_ems)

    # update the dynamic parameters
    _x, _y, _z = pos_ems[best_ems_index]
    container[_x:_x+block_x, _y:_y+block_y, _z:_z+block_z] = block_index + 1
    container[_x:_x+block_x, _y:_y+block_y, 0:_z][ under_space_mask[best_ems_index] ] = -1
    positions[block_index] = pos_ems[best_ems_index]
    stable[block_index] = is_stable_ems[best_ems_index]
    heightmap = heightmap_ems[best_ems_index]
    empty_size = empty_ems[best_ems_index]

    return container, positions, stable, heightmap, valid_size, empty_size

def calc_positions_lb_greedy(blocks, container_size):
    '''
    calculate the positions to pack a group of blocks into a container by lb-greedy
    ---
    params:
    ---
        blocks: n x 2/3 array, blocks with an order
        container_size: 1 x 2/3 array, size of the container
        reward_type: string
            'C' / 'S' / 'P'
            'C+P'
            'C+P+S'
            just combination of C P S
        packing_strategy: string
    return:
    ---
        positions: int x 2/3 array, packing positions of the blocks
        container: width (* depth) * height array, the final state of the container
        stable: n x 1 bool list, each element indicates whether a block is placed(hard)/stable(soft) or not
        ratio: float, C / C*S / C+P / (C+P)*S / C+P+S, calculated by the following scores
        scores: 5 integer numbers: valid_size, box_size, empty_size, stable_num and packing_height
    '''
    # Initialize
    blocks = blocks.astype('int')
    blocks_num = len(blocks)
    block_dim = len(blocks[0])
    positions = np.zeros((blocks_num, block_dim)).astype(int)
    container = np.zeros(list(container_size)).astype(int)
    stable = [False] * blocks_num
    heightmap = np.zeros(container_size[:-1]).astype(int)
    valid_size = 0
    empty_size = 0

    for block_index in range(blocks_num):
        container, positions, stable, heightmap, valid_size, empty_size = \
            calc_one_position_lb_greedy_3d(blocks[block_index], block_index, container_size,
                                        container, positions, stable, heightmap, valid_size, empty_size)

    return positions[-1]


# ===============================


def deep_left_bottom(x,y):
    ''' use for sort function
        x: [ id, [x,y,z] ]
        y: [ id, [x,y,z] ]

        exp:

        `
        sort_pos = sorted( enumerate(pos_list), key=functools.cmp_to_key(deep_left_bottom))
        `
    '''
    a = x[1]
    b = y[1]
    diff = np.sign(a - b)
    return diff[2] * 100 + diff[0] * 10 + diff[1] * 1

def ppsg_cut(box_num, size_range, container_size):
    '''
        box_num: int
        size_rage: list[ low, height ]
        container_sze: list[width, length, height]
    '''
    # modify from: AAAI19_Ranked Reward Enabling Self-Play Reinforcement Learning for Bin packing

    def cut(box, pos, cut_axis, cut_loc):

        box_left = []
        box_right = []

        pos_left = []
        pos_right = []
        
        for ax in range(3):
            if ax != cut_axis:
                box_left.append(box[ax])
                box_right.append(box[ax])
                
                pos_left.append(pos[ax])
                pos_right.append(pos[ax])

            else:
                box_left.append( cut_loc )
                box_right.append( box[ax] - cut_loc )
                
                pos_left.append( pos[ax] )
                pos_right.append( pos[ax] + cut_loc )
        
        box_left = np.array(box_left)
        box_right = np.array(box_right)
        pos_left = np.array(pos_left)
        pos_right = np.array(pos_right)
    
        return [ box_left, pos_left ], [box_right, pos_right]

    def update_cutting( box_list, pos_list, size_range, soft_size_cut=False ):
        boxes = np.array(box_list)
        poses = np.array(pos_list)

        if soft_size_cut:
            # allow to cut the small box
            box_id = np.where( boxes >= size_range[0] * 2 )[0][0]
        else:
            # only to cut the large box
            box_id = np.where( boxes > size_range[1] )[0][0]
        cut_axis = np.argmax(boxes[box_id])

        box = boxes[box_id]
        pos = poses[box_id]
        while True:
            min_len = min(size_range[1], box[cut_axis])
            cut_loc = np.random.randint(size_range[0], min_len)

            if box[cut_axis] - cut_loc >= size_range[0]:
                break
        
        left, right = cut(box, pos, cut_axis, cut_loc)    
        left_box = left[0]
        right_box = right[0]

        left_pos = left[1]
        right_pos = right[1]

        new_boxes = []
        new_poses = []

        for i in range(len(boxes)):
            if i != box_id:
                new_boxes.append(boxes[i])
                new_poses.append(poses[i])
            else:
                new_boxes.append(left_box)
                new_boxes.append(right_box)

                new_poses.append(left_pos)
                new_poses.append(right_pos)

        return new_boxes, new_poses

    def cut_sub_box(box, pos, box_num, size_range):

        box_list = []
        pos_list = []

        cut_axis = np.argmax(box)

        if box_num == 1 or box[cut_axis] < size_range[0]*2:
            box_list.append(np.array(box))
            pos_list.append(np.array(pos))
        else:

            while True:
                min_len = min(size_range[1], box[cut_axis])
                cut_loc = np.random.randint(size_range[0], min_len)

                if box[cut_axis] - cut_loc >= size_range[0]:
                    break

            if box_num == 2:
                left_num = 1
                right_num = 1
            else:
                left_num = int(np.ceil(box_num * ( cut_loc / box[cut_axis] )))
                right_num = box_num - left_num
            
            # print(left_num, right_num)

            left, right = cut(box, pos, cut_axis, cut_loc)    
            left_box = left[0]
            right_box = right[0]

            left_pos = left[1]
            right_pos = right[1]

            left_list =  cut_sub_box( left_box, left_pos, left_num, size_range )
            right_list =  cut_sub_box( right_box, right_pos, right_num, size_range )

            box_list += left_list[0]
            box_list += right_list[0]
            
            pos_list += left_list[1]
            pos_list += right_list[1]
            
        return box_list, pos_list

    box_list = []
    pos_list = []

    box = np.array(container_size)
    pos = np.array([0, 0, 0])

    while len(box_list) != box_num:
        box_list, pos_list = cut_sub_box(box, pos, int(box_num * 0.5), size_range)

        while np.max(box_list) > size_range[1]:
            box_list, pos_list = update_cutting(box_list, pos_list, size_range )

        while len(box_list) < box_num:
            box_list, pos_list = update_cutting(box_list, pos_list, size_range, True )

    # sort by the pos to get the candidate order, so we can pack by deep_left strategy
    sort_result = sorted( enumerate(pos_list), key=functools.cmp_to_key(deep_left_bottom))
    sort_boxes = []
    sort_poses = []
    for i, pos in sort_result[::-1]:
        sort_boxes.append(box_list[i])
        sort_poses.append(pos)
    box_list = sort_boxes
    pos_list = sort_poses

    return box_list, pos_list

def left_bottom(container, box, find_ems=False, ret_ems=False):
    # TODO for multi-container
    check_box_stable = False
    if 'hard' in container.stable_rule:
        check_box_stable = True

    if ret_ems:
        while True:
            origin_ems, ems, ems_mask = container.get_ems()
            ems_size_mask, ems_to_box_mask, _ = ET.compute_box_ems_mask(box[None,:], origin_ems, 1, container.each_space, check_box_stable=check_box_stable, check_z=False)


            # valid_ems_ids = np.where(ems_to_box_mask == 1)[0]

            valid_ems_ids = np.where(ems_size_mask == 1)[0]
            valid_ems = origin_ems[valid_ems_ids]

            list_ems = list(valid_ems)
            sort_ems = sorted( enumerate(list_ems), key=functools.cmp_to_key(deep_left_bottom))

            # if len(sort_ems) == 0:
            #     valid_ems_ids = np.where(ems_size_mask == 1)[0]
            #     valid_ems = origin_ems[valid_ems_ids]

            #     list_ems = list(valid_ems)
            #     sort_ems = sorted( enumerate(list_ems), key=functools.cmp_to_key(deep_left_bottom))

            if len(sort_ems) == 0:
                
                container.each_valid_mask = list( np.array(container.each_valid_mask) * 0 )

                if container.container_type == 'multi' and container.init_ctn_num is None:
                    container.add_new_container()
                    continue
                else:
                    return None

            ems_id = valid_ems_ids[sort_ems[0][0]]
            break
    else:
        test_list = container.boxes + [box]
        test_list = np.array(test_list)
        pos = calc_positions_lb_greedy(test_list, container.container_size)
        return pos

    if find_ems:
        return ems_id
    else:
        pos = container.add_new_box(box, ems_id, 0, False)
        container.update_ems()
        return pos

# ===============================

class Container(object):
    
    def __init__(self, container_size, reward_type, unit_scale=1, stable_rule='hard_before_pack', 
                 stable_scale_factor=0.1, use_bridge=False, same_height_threshold=0, min_ems_width=0,
                 ems_type='ems', world_type='real', container_type='multi', pack_type='all', init_ctn_num=None, stable_predict=False):
        '''
        Args:
            container_size: List(int) [3]
            reward_type: string {'C' 'P', 'S' ...}
            stable_rule: string {'hard_before_pack', 'hard_after_pack', 'soft_before_pack', 'soft_after_pack', 'none'}
            stable_scale_factor: flaot, larger means harder
            min_ems_width: int
            ems_type: string {'ems', 'id', 'stair'}
            world_type: string {'real', 'ideal'}
            container_type: string {'multi', 'single'}
            pack_type: string {'all', 'last'}

        '''

        # ideal / real
        #     single / multi
        #         last / all

        assert pack_type != 'last' or init_ctn_num is None, 'pack_type = last can not set with init_ctn_num is None'

        self.ems_type = ems_type
        self.world_type = world_type
        self.container_type = container_type
        self.pack_type = pack_type
        
        self.stable_rule = stable_rule
        if world_type == 'real':
            self.space_holder = 200
            self.stable_scale_factor = stable_scale_factor
        else:
            self.space_holder = None

        self.stable_predict = stable_predict

        #     self.space = Space(*container_size, holder=holder, stable_scale_factor=stable_scale_factor)
        # else:
        #     self.space = None

        self.reward_type = reward_type
        self.use_bridge = use_bridge
        self.same_height_threshold = same_height_threshold
        self.min_ems_width = min_ems_width
        
        self.init_ctn_num = init_ctn_num

        self.packing_mask = None

        self.duo = None

        self.positions = []
        self.boxes = []
        

        self.each_stop_box = []
        self.each_stop_pos = []
        
        
        self.each_container_height = []
        self.each_container_boxes = []
        self.each_container_positions = []
        self.each_box_ids = []
        
        self.each_container_heightmap = []
        self.each_container_idmap = []
        self.each_container_stable = []
        self.each_rotate_state = []
        self.each_space = []

        self.each_container_vboxes = []
        self.each_container_vheightmap = []

        self.each_valid_mask = []

        self.empty_max_spaces = []
        self.height_diffs = []

        self.CPSs = []
        self.Cs = []
        self.Ps = []
        self.Ss = []

        self.new_by_unstable = False

        self.current_box_num = 0

        self.container_size = container_size
        self.vextend_len = 5
        self.vcontainer_size = [ container_size[0] + self.vextend_len, container_size[1] + self.vextend_len, container_size[2]  ]

        self.start_to_stop = False

        self.last_pack_container = 0
        if init_ctn_num is not None:
            for _ in range(init_ctn_num):
                self.add_new_container()
        else:
            self.add_new_container()


    def _move_to_center(self, box, pos, hm):
        bx, by, bz = box
        px, py, pz = pos
        hw, hl = hm.shape

        new_px = px * 1
        new_py = py * 1
        # return new_px, new_py

        before_area = np.sum(hm[ new_px:new_px+bx, new_py:new_py+by ] == pz)
        box_area = bx * by
        
        low_w, low_l = 1, 1

        while pz > 0 and low_w < hw:
            if (hm[ px:px+low_w+1, py:py+low_l ] == pz).all():
                low_w += 1
            else:
                break
        while pz > 0 and low_l < hl:
            if (hm[ px:px+low_w, py:py+low_l+1 ] == pz).all():
                low_l += 1
            else:
                break

        add_sign = 0

        if box_area > before_area:
            add_sign = -1
        elif box_area > before_area * 0.88:
            add_sign = 1
        
        # if box_area > before_area * 0.6:
        if add_sign != 0:
            
            x_move_max = np.ceil(( low_w/2 - bx/2 ) * add_sign)
            y_move_max = np.ceil(( low_l/2 - by/2 ) * add_sign)

            while new_px > 0 and new_px < hw and (new_px - px)*add_sign < x_move_max:
                new_hm = hm[ new_px + add_sign:new_px + add_sign+bx, new_py:new_py+by ]

                if new_hm.max() == pz and np.sum(new_hm == pz) >= before_area:
                    new_px += add_sign
                else:
                    break
            while new_py > 0 and new_py < hl and (new_py-py)*add_sign < y_move_max:
                new_hm = hm[ new_px:new_px+bx, new_py+add_sign:new_py+add_sign+by ]
                if new_hm.max() == pz and np.sum(new_hm == pz) >= before_area:
                    new_py += add_sign
                else:
                    break

        # if new_px != px or new_py != py:
        #     print("----------------------")
        #     print("move to center !!", add_sign, [px, py], [new_px, new_py])
        #     print("----------------------")
        return new_px, new_py
        
        

    def add_new_box(self, box, ems_id=0, corner_id=0, is_rotate=False, real_pos=None, allow_unstable=False, box_id=0):
        '''
        Add a box into current container
        ---
        params:
        ---
            box: int * 3 array, a box
        returns:
        ---
            heightmap: (width) or (width x depth) array, heightmap of the container
        '''
        
        # self.new_container = False
        self.new_by_unstable = False

        box = np.array(box, dtype='int')

        if real_pos is None:
            ems_xy, ems_wl, container_id = ET.compute_packing_pos(box, ems_id, corner_id, self.empty_max_spaces)
        else:
            ems_xy = real_pos[:2]
            ems_wl = None
            container_id = int(real_pos[3])
        
        bx, by, bz = box
        bx = int(bx)
        by = int(by)

        heightmap_max = 50000

        pos = None
        is_stable = None

        heightmap = self.each_container_heightmap[container_id]
        id_map = self.each_container_idmap[container_id]

        while heightmap_max > self.container_size[-1] and self.start_to_stop == False:

            # heightmap = self.heightmap.copy()
            # id_map = self.id_map.copy()
            heightmap = self.each_container_heightmap[container_id].copy()
            id_map = self.each_container_idmap[container_id].copy()

            # empty_size = self.empty_size

            lx = int(ems_xy[0])
            ly = int(ems_xy[1])
            lz = heightmap[lx:lx+bx, ly:ly+by].max()

            # NOTE move to center
            
            lx, ly = self._move_to_center( np.array([bx, by, bz]).astype('int'), np.array([lx, ly, lz]).astype('int'), heightmap )
            
            # bz = int(bz)
            # lz = int(lz)

            # valid_size = self.valid_size + bx * by * bz
            
            # layer_under_box = heightmap[ lx:lx+bx, ly:ly+by ]
            # under_z = layer_under_box - lz
            # empty_mask = (under_z < 0)
            # empty_size = empty_size + np.sum(empty_mask)

            final_z = bz + lz
            if self.same_height_threshold > 0:
                # NOTE adjust the height
                z_diff = np.abs(final_z - heightmap) 
                z_mask = z_diff <= self.same_height_threshold

                valid_area = heightmap[z_mask]
                if len(valid_area) > 0:
                    max_valid_height = np.max(valid_area)
                    if max_valid_height > final_z:
                        final_z = max_valid_height
                    heightmap[z_mask] = final_z

            heightmap[ lx:lx+bx, ly:ly+by ] = final_z
            id_map[ lx:lx+bx, ly:ly+by ] = self.current_box_num + 1

            # if ems_wl is not None:# and lz == 0:
            #     move_threshold = 15
            #     # check can we move the location
            #     ems_w, ems_l = ems_wl
            #     if ems_w - bx < move_threshold:
            #         lx += 2
            #     if ems_l - by < move_threshold:
            #         ly += 2

            pos = np.array([lx, ly, lz])
            
            heightmap_max = heightmap.max()
            

            stop_current_pack = False

            if heightmap_max > self.container_size[-1]:
                stop_current_pack = True
                
            # NOTE HERE
            # if len(self.each_space) > 0 and 'after' in self.stable_rule and 'hard' in self.stable_rule: # must place at a stable location after select
            #     is_stable = self.each_space[container_id].drop_box_virtual([bx,by,bz], [lx,ly], False, 1, 1)
            #     if not is_stable:
            #         stop_current_pack = True
            #         self.new_by_unstable = True


            if stop_current_pack:
                self.each_valid_mask[container_id] = 0
                
                if np.sum(self.each_valid_mask) == 0 or self.stable_predict:
                    # add container
                    if self.container_type == 'multi' and self.init_ctn_num is None:

                        self.add_new_container()
                        container_id += 1

                        heightmap_max = 50000

                        if ems_xy is not None:
                            ems_xy *= 0

                    elif self.container_type == 'single' or self.init_ctn_num is not None:

                        self.each_stop_box[container_id].append(box)
                        self.each_stop_pos[container_id].append(np.array([lx, ly, lz]))

                        self.start_to_stop = True
                        pos = None
                        break
                else:
                    pos = None
                    break
        
        if pos is not None:
            if len(self.each_space) != 0:
                is_stable = self.each_space[container_id].drop_box([bx,by,bz], [lx,ly], False, 1, 1, allow_unstable, 
                                                                   same_height_threshold = self.same_height_threshold,
                                                                   id_map=self.each_container_idmap[container_id])

                if self.duo is not None:
                    new_boxes = self.each_container_boxes[container_id] + [np.array(box)]
                    new_poses = self.each_container_positions[container_id] + [ np.array(pos) ]
                    duo_stable = self.duo.check_stable( new_boxes, new_poses )
                    is_stable =  is_stable and duo_stable
                
                # layer_under_box = heightmap[lx:bx+lx, ly:ly+by]
                # is_stable = ET.check_stable([bx,by,bz], [lx, ly, lz], layer_under_box)

            else:
                is_stable = True

            if not is_stable:
                self.each_stop_box[container_id].append(box)
                self.each_stop_pos[container_id].append(pos)
                
                # print('stop unstable')
                container_num = len(self.each_valid_mask)
                if self.container_type == 'single':
                    self.start_to_stop = True
                    for ci in range(container_num):
                        self.each_valid_mask[ci] = 0

            # update all valid state of container if all
            container_num = len(self.each_valid_mask)
            if self.pack_type == 'all':
                self.each_valid_mask = list(np.ones(container_num))

                # NOTE HERE
                for ci in range(container_num):
                    self.each_valid_mask[ci] = len(self.each_stop_box[ci]) == 0

            self.each_rotate_state[container_id].append(is_rotate)
            self.each_container_stable[container_id].append(is_stable)

            self.each_container_boxes[container_id].append(box)
            self.each_container_positions[container_id].append(pos)
            self.each_box_ids[container_id].append(box_id)

            self.boxes.append(box)
            self.positions.append(pos)

            self.each_container_height[ container_id] = np.max(heightmap)
            
            # UPDATE stable_heightmap
            if len(self.each_space) > 0:
                w = self.container_size[0]
                l = self.container_size[1]
                self.each_space[container_id].plain[:w, :l] = heightmap.copy()

            self.each_container_heightmap[container_id] = heightmap
            self.each_container_idmap[container_id] = id_map

            self.last_pack_container = container_id
            
        else:
            return None

        self.current_box_num = self.current_box_num + 1
        
        return pos

    def update_ems(self, min_ems_width=None):

        if min_ems_width is None:
            min_ems_width = self.min_ems_width
        
        # boxes = self.current_boxes
        # positions = self.current_positions

        # if self.use_bridge == False:
        #     boxes = None
        #     positions = None

        self.empty_max_spaces = ET.compute_ems(self.each_container_idmap, self.each_container_heightmap, self.container_size, min_ems_width, self.ems_type, self.each_valid_mask)

    def get_ems(self, ems_dim=7, for_test=False, max_ems_num=None):

        ems = self.empty_max_spaces
        # ems = np.array(ems).reshape(-1, 12)[:, :ems_dim]
        ems = np.array(ems).reshape(-1, 9)[:, :ems_dim]

        # pos | right_top_pos
        left_botton_corner = ems[:, :3]
        right_top_corner = ems[:, 3:6]
        ems[:, 3:6] = right_top_corner - left_botton_corner

        origin_ems = copy.deepcopy(ems)
        
        # pos | size
        if len(ems) > 0:
            size_min_height = ems[:, 5].min()
            ems[:, 5] = ems[:, 5] - size_min_height

        ems_len = len(ems)
        
        if for_test or max_ems_num is None:
            max_ems_num = ems_len
        else:
            max_ems_num = max_ems_num
        # max_ems_num = self.max_ems_num

        # ems_per_box = 5
        all_ems = np.zeros(( max_ems_num, ems_dim), dtype=np.float32 )
        origin_all_ems = np.zeros(( max_ems_num, ems_dim), dtype=np.float32 )
        # all_ems = np.zeros(( ems_len, 6), dtype=np.float32 )
        all_ems[:ems_len] = ems
        origin_all_ems[:ems_len] = origin_ems

        # ems_masks = np.zeros((ems_len), dtype=bool)
        ems_masks = np.zeros(( max_ems_num ), dtype=bool)
        ems_masks[:ems_len] = 1

        return origin_all_ems, all_ems, ems_masks

    def get_heightmap(self, heightmap_type='full'):

        last_heightmap = self.each_container_heightmap[self.last_pack_container]

        heightmap = copy.deepcopy(last_heightmap)
        if heightmap_type == 'full':
            hm = last_heightmap
        elif heightmap_type == 'zero':
            hm = last_heightmap - np.min(last_heightmap)
        elif heightmap_type == 'diff':
            if len(heightmap.shape) == 2:
                # x coordinate
                hm_diff_x = np.insert(last_heightmap, 0, heightmap[0, :], axis=0)
                hm_diff_x = np.delete(hm_diff_x, len(hm_diff_x)-1, axis=0)
                hm_diff_x = heightmap - hm_diff_x
                # hm_diff_x = np.delete(hm_diff_x, 0, axis=0)
                # y coordinate
                hm_diff_y = np.insert(heightmap, 0, heightmap[:, 0], axis=1)
                hm_diff_y = np.delete(hm_diff_y, len(hm_diff_y.T)-1, axis=1)
                hm_diff_y = heightmap - hm_diff_y
                # hm_diff_y = np.delete(hm_diff_y, 0, axis=1)
                # combine
                width = last_heightmap.shape[0]
                length = last_heightmap.shape[1]
                hm = np.zeros( (2, width, length) ).astype(int)
                hm[0] = hm_diff_x
                hm[1] = hm_diff_y
            else:
                hm = last_heightmap
                hm = np.insert(hm, len(hm)-1, hm[-1])
                hm = np.delete(hm, 0)
                hm = hm - last_heightmap
                hm = np.delete(hm, len(hm)-1)
        return hm

    def update_heightmap(self, container_id, new_heightmap=None, new_idmap=None, same_height_threshold=0):

        if new_heightmap is not None:
            heightmap = new_heightmap
        else:
            heightmap = self.each_container_heightmap[container_id].copy()

        if same_height_threshold > 0:
            # final_z = bz + lz

            all_h = np.unique(heightmap)
            all_h = np.sort(all_h)[::-1]

            for final_z in all_h:
                # NOTE adjust the height
                z_diff = final_z - heightmap
                
                z_mask = (z_diff < same_height_threshold) * (z_diff >= 0) == True

                valid_area = heightmap[z_mask]
                if len(valid_area) > 0:
                    max_valid_height = np.max(valid_area)
                    if max_valid_height > final_z:
                        final_z = max_valid_height
                    heightmap[z_mask] = final_z

        self.each_container_heightmap[container_id] = heightmap
        self.each_container_height[ container_id ] = np.max(heightmap)

        # UPDATE stable_heightmap
        if len(self.each_space) > 0:
            w = self.container_size[0]
            l = self.container_size[1]
            self.each_space[container_id].plain[:w, :l] = heightmap.copy()

        if new_idmap is not None:
            self.each_container_idmap[container_id] = new_idmap
        
        self.update_ems()

        return heightmap

    def change_heightmap_size(self, new_size):
        ol, ow = self.container_size[:2]
        nl, nw = new_size[:2]
        
        space_num = len(self.each_space)
        container_num = len(self.each_container_heightmap)
        for i in range(container_num):
            hm = self.each_container_heightmap[i]
            idm = self.each_container_idmap[i]

            new_hm = hm[ :nl, :nw ]
            new_idm = idm[ :nl, :nw ]

            self.each_container_heightmap[i] = new_hm
            self.each_container_idmap[i] = new_idm
            
            if space_num > 0:
                space = self.each_space[i]
                space.change_space_size( new_size )
        
        self.container_size = new_size

    def set_init_height(self, heightmap):
        self.each_container_heightmap[0] = heightmap

    def get_max_height_diff(self, min_ems_width, gripper_width):
        height_diff = 0

        # [ [x,y,z], [x,y,z], [0,0,0] ]
        all_ems = np.array(self.empty_max_spaces)

        container_num = len(self.each_container_height)
        for ci in range(container_num):
            max_h = -1
            min_h = 100000

            if len(all_ems) == 0: continue
            
            ci_hm = self.each_container_heightmap[ci]
            ci_ems = all_ems[ all_ems[:, 2][:, 0] == ci ]
            for ems in ci_ems:
                if ems[1][2] > 0:
                    ems_wl = (np.array(ems[1]) - np.array(ems[0]))[:2]
                    if np.min(ems_wl) < min_ems_width:
                        continue
                    ez = ems[0][2]

                    x, y, z = ems[0].astype('int')

                    # max_h = np.max(ci_hm[  max( 0, x-gripper_width) : x+gripper_width, max( 0, y-gripper_width) : y+gripper_width ])
                    
                    # if height_diff < max_h - z:
                    #     height_diff = max_h - z

                    if ez < min_h:
                        min_h = ez
                    if ez > max_h:
                        max_h = ez

            if height_diff < max_h - min_h:
                height_diff = max_h - min_h

        return height_diff

    def ems_to_height(self, min_ems_width=0):
        # [ [x,y,z], [x,y,z], [0,0,0] ]
        all_ems = np.array(self.empty_max_spaces)
        all_heightmaps = []

        container_num = len(self.each_container_height)
        for ci in range(container_num):
            ci_heightmap = np.zeros_like(self.each_container_heightmap[0])

            if len(all_ems) == 0: continue
            
            ci_ems = all_ems[ all_ems[:, 2][:, 0] == ci ]
            for ems in ci_ems:
                if ems[1][2] > 0:
                    ems_wl = (np.array(ems[1]) - np.array(ems[0]))[:2]
                    if np.min(ems_wl) < min_ems_width:
                        continue

                    x, y, z = ems[0].astype('int')
                    w, l = ems_wl.astype('int')

                    ci_heightmap[ x:x+w, y:y+l ] = z + l

            all_heightmaps.append(ci_heightmap)
        
        return all_heightmaps


    def set_packing_mask(self, mask):
        self.packing_mask = mask

    def add_new_container(self):

        new_container_id = len(self.each_valid_mask)
        
        self.each_container_height.append(0)
        self.each_container_boxes.append([])
        self.each_container_positions.append([])
        self.each_box_ids.append([])

        self.each_stop_pos.append([])
        self.each_stop_box.append([])

        self.each_container_heightmap.append(
            np.zeros(self.container_size[:-1], dtype=np.float32)
        )
        self.each_container_idmap.append(np.zeros(self.container_size[:-1], dtype=np.int32))
        self.each_container_stable.append([])
        self.each_rotate_state.append([])

        self.each_container_vboxes.append([])
        self.each_container_vheightmap.append(
            np.zeros(self.vcontainer_size[:-1], dtype=np.float32))
        
        self.each_valid_mask.append(1)

        if self.world_type == 'real':
            space = Space(*self.container_size, holder=self.space_holder, stable_scale_factor=self.stable_scale_factor)
            self.each_space.append(space)

            # self.duo = Duo(self.container_size)

        # if self.space is not None:
        #     self.space.reset()

        # self.empty_max_spaces += [[[0,0,0], list(self.container_size), [1]*3, [new_container_id]*3]]
        self.empty_max_spaces += [[[0,0,0], list(self.container_size), [new_container_id]*3]]

        self.start_to_stop = False

    def clear_container(self, with_reward=True):
        
        self.current_box_num = 0
        self.last_pack_container = 0

        self.positions.clear()
        self.boxes.clear()

        self.each_container_height.clear()
        self.each_container_boxes.clear()
        self.each_container_positions.clear()
        self.each_box_ids.clear()

        self.each_stop_pos.clear()
        self.each_stop_box.clear()

        self.each_container_heightmap.clear()
        self.each_container_idmap.clear()
        self.each_container_stable.clear()
        self.each_rotate_state.clear()
        self.each_space.clear()

        self.each_container_vboxes.clear()
        self.each_container_vheightmap.clear()

        self.each_valid_mask.clear()
        self.empty_max_spaces.clear()

        if with_reward:
            self.CPSs.clear()
            self.Cs.clear()
            self.Ps.clear()
            self.Ss.clear()

        if self.init_ctn_num is not None:
            for _ in range(self.init_ctn_num):
                self.add_new_container()
        else:
            self.add_new_container()

    def calc_ratio(self, height_mode='container'):

        # TODO, for S and P?

        all_v = 0
        
        # delta 
        delta_int = 0
        delta_float = 0

        self.Cs.clear()
        container_num = len(self.each_container_boxes)
        if len(self.each_container_boxes[-1]) == 0:
            container_num -= 1
        
        for ci in range( container_num ):
            vol = 0
            boxes = self.each_container_boxes[ci]

            # if self.pack_type == 'last':
            #     if ci < container_num - 1:
            #         height = self.container_size[2]
            #     else:
            #         height = self.each_container_height[ci]
            # elif self.pack_type == 'all':
            #     height = self.each_container_height[ci]

            if height_mode == 'container':
                height = self.container_size[2]
            elif height_mode == 'current':
                height = self.each_container_height[ci]

            for box in boxes:
                bv = box[0] * box[1] * box[2]
                all_v += bv
                vol += bv

            if height == 0:
                c = 0
            else:
                c = vol / ( self.container_size[0] * self.container_size[1] * height )
            
            self.Cs.append( c )
        
        ideal_container_num = all_v / (self.container_size[0] * self.container_size[1] * self.container_size[2])

        delta_float = container_num - ideal_container_num
        delta_int = container_num - np.ceil(ideal_container_num)

        ratio = np.mean(self.Cs)
        return ratio, delta_float, delta_int


    def save_states(self, boxes=None, pos=None, save_dir="./pack/test/100-100-note-test", remove_list=[]):
        # if boxes is None:
        #     boxes = []
        #     pos = []
        #     for i in range(len(self.boxes)):
        #         if i not in remove_list:
        #             boxes.append(self.boxes[i])
        #             pos.append(self.positions[i])
        
        container_num = len(self.each_container_boxes)
        origin_all_ems, all_ems, ems_masks = self.get_ems()

        for i in range(container_num):
            os.makedirs(save_dir + f'-{i}', exist_ok=True)

            container_ems = origin_all_ems[origin_all_ems[:, -1] == i]
            
            np.save( os.path.join(save_dir + f'-{i}', "pack_info_box"), np.array(self.each_container_boxes[i]))
            np.save( os.path.join(save_dir + f'-{i}', "pack_info_pos"), np.array(self.each_container_positions[i]))
            np.save( os.path.join(save_dir + f'-{i}', "pack_info_id"), np.array(self.each_box_ids[i]))
            np.save( os.path.join(save_dir + f'-{i}', "pack_info_stop"), list(self.each_stop_pos[i]) + list(self.each_stop_box[i]) )

            np.save( os.path.join(save_dir + f'-{i}', "pack_info_ems_box"), container_ems[:, 3:6] )
            np.save( os.path.join(save_dir + f'-{i}', "pack_info_ems_pos"), container_ems[:, :3] )
        np.save( save_dir, np.array([container_num]) )