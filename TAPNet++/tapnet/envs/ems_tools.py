import numpy as np
import copy
import itertools
import matplotlib.pyplot as plt

# corners, left_bottom_corner, x_borders, y_borders = self.compute_corners(heightmap)
# self.compute_empty_space(left_bottom_corner, x_borders, y_borders, heightmap, empty_max_spaces, 'right', 'right')
# self.compute_empty_space(corners, x_borders, y_borders, heightmap, empty_max_spaces, 'left-right', 'left-right')

from scipy.spatial import ConvexHull
from matplotlib.path import Path

def check_stable(box, pos, layer_under_box=None, mask_under_box=None):
    '''
    check for 3D packing
    '''
    if layer_under_box is None and mask_under_box is None: return True
    
    if (pos[2]==0):
        if mask_under_box is None:
            return True
        # else:
        #     return False

    if mask_under_box is not None:
        if (mask_under_box == 0).any(): return False


    # obj_center = ( pos[0] + box[0]/2, pos[1] + box[1]/2  )
    obj_center = ( box[0]/2, box[1]/2  )
    # valid points right under this object
    
    if layer_under_box is None:
        layer_under_box = np.ones([box[0], box[1]]) * pos[2]
    if mask_under_box is None:
        mask_under_box = np.ones_like(layer_under_box)
    max_height = layer_under_box.max()
    x, y = np.where( (layer_under_box == max_height) * (mask_under_box == True) )

    points = np.concatenate([[x],[y]]).transpose()

    if(len(points) > box[0]*box[1]/2):
        return True
    if(len(points)==0 or len(points)==1):
        return False
    elif(len(points)==2):
        # whether the center lies on the line of the two points
        a = obj_center[0] - points[0][0]
        b = obj_center[1] - points[0][1]
        c = obj_center[0] - points[1][0]
        d = obj_center[1] - points[1][1]
        # same ratio and opposite signs
        if (b==0 or d==0):
            if (b!=d): return False
            else: return (a<0)!=(c<0) 
        return ( a/b == c/d and (a<0)!=(c<0) and (b<0)!=(d<0) )
    else:
        # calculate the convex hull of the points
        points = np.array(points)
        try:
            convex_hull = ConvexHull(points)
        except:
            # error means co-lines
            min_point = points[np.argmin( points[:,0] )]
            max_point = points[np.argmax( points[:,0] )]
            points = np.array( (min_point, max_point) )
            a = obj_center[0] - points[0][0]
            b = obj_center[1] - points[0][1]
            c = obj_center[0] - points[1][0]
            d = obj_center[1] - points[1][1]
            if (b==0 or d==0):
                if (b!=d): return False
                else: return (a<0)!=(c<0)
            return ( a/b == c/d and (a<0)!=(c<0) and (b<0)!=(d<0) )

        hull_path = Path(points[convex_hull.vertices])
        return hull_path.contains_point((obj_center))

def compute_packing_pos(box, ems_id, corner_id, empty_max_spaces, id_map=None):
    ems_xyz = np.array(empty_max_spaces[ems_id][0]) # get xy
    ems_XYZ = np.array(empty_max_spaces[ems_id][1]) # get xy

    # ems_type = np.array(empty_max_spaces[ems_id][2])[0] # get space_type
    ems_type = 0

    container_id = np.array(empty_max_spaces[ems_id][-1])[0] # get container_id

    ems_xy = ems_xyz[:2] # get xy
    
    if False:
    # if True:
        # NOTE 这里近似 EMS
        net_w = 20
        real_w = 100
        reso = real_w / net_w
        ems_xy = (np.ceil(ems_xy / real_w * net_w) * reso).astype('int')
    
    ems_wl = (ems_XYZ - ems_xyz)[:2] # get size width and length
    ems_wl = np.array(ems_wl, dtype='int')
    box = np.array(box, dtype='int')

    if ems_type == -100:
        # remove bridge, is container_id
        x = int(ems_xy[0])
        y = int(ems_xy[1])
        w = int(ems_wl[0])
        l = int(ems_wl[1])

        corners, left_bottom_corners, x_borders, y_borders, _ = compute_corners(id_map[ x:x+w, y:y+l ])
        # empty_spaces = []
        # compute_empty_space(container_size, corners, x_borders, y_borders, id_map, id, empty_max_spaces, 'right', 'right', 1)

        threshold = 1e-4
        left_corner_id = 0
        # for c_i, c in enumerate(left_bottom_corners):
        #     if abs(ems_xy[0] - (c[0] + x)) < threshold and abs(ems_xy[1] - (c[1] + y)) < threshold:
        #         left_corner_id = c_i
        #         break

        small_pos = left_bottom_corners[left_corner_id][:2]
        large_pos = left_bottom_corners[1-left_corner_id][:2]

        pos_diff_x = large_pos[0] - small_pos[0]
        pos_diff_y = large_pos[1] - small_pos[1]

        if abs(pos_diff_x) < threshold:
            half_len = np.ceil(box[1] / 2.0)
            pos_offset = pos_diff_y if pos_diff_y < half_len else half_len
            # ems_xy[1] = y + large_pos[1] - pos_offset

            tmp = y + large_pos[1] - pos_offset
            if tmp + box[1] > y+l:
                tmp = y+l - box[1]
            ems_xy[1] = tmp
            
        elif abs(pos_diff_y) < threshold:
            half_len = np.ceil(box[0] / 2.0)
            pos_offset = pos_diff_x if pos_diff_x < half_len else half_len
            # ems_xy[0] = x + large_pos[0] - pos_offset

            tmp = x + large_pos[0] - pos_offset
            if tmp + box[0] > x+w:
                tmp = x+w - box[0]
            ems_xy[0] = tmp
            
        else:
            assert "error left corner of bridge ems"

    if corner_id == 0:
        ems_xy = ems_xy
    elif corner_id == 1:
        ems_xy += [ ems_wl[0] - box[0], 0 ]
    elif corner_id == 2:
        ems_xy += [ 0, ems_wl[1] - box[1] ]
    elif corner_id == 3:
        ems_xy += [ ems_wl[0] - box[0], ems_wl[1] - box[1] ]
    elif corner_id == 4:
        ems_xy += [ int(ems_wl[0]/2 - box[0]/2), int(ems_wl[1]/2 - box[1]/2) ]

    return ems_xy, ems_wl, container_id

def find_bridge_corners(same_height_box_pairs, target_box):
    # TODO ?
    bridge_corners = []
    
    for box_pairs in same_height_box_pairs:
        box_small, box_large, pos_small, pos_large, neighbor_axis, cross_axis = box_pairs

        corner = np.zeros(2)
        if pos_small[cross_axis] < pos_large[cross_axis]:
            corner[cross_axis] = pos_large[cross_axis]
        else:
            corner[cross_axis] = pos_small[cross_axis]

        corner[neighbor_axis] = pos_small[neighbor_axis] + box_small[neighbor_axis]
        axis_offset = np.ceil(target_box[neighbor_axis] / 2)
        if axis_offset > box_small[neighbor_axis]:
            axis_offset = box_small[neighbor_axis]

        corner[neighbor_axis] -= axis_offset
        bridge_corners.append(corner)
    
    return bridge_corners

def compute_corners(heightmap, get_xy_corners=False):
    # NOTE find corners by heightmap
    
    hm_shape = heightmap.shape
    extend_hm = np.ones( (hm_shape[0]+2, hm_shape[1]+2) ) * 10000
    extend_hm[1:-1, 1:-1] = heightmap

    x_diff_hm_1 = extend_hm[:-1] - extend_hm[1:]
    x_diff_hm_1 = x_diff_hm_1[:-1, 1:-1]

    x_diff_hm_2 = extend_hm[1:] - extend_hm[:-1]
    x_diff_hm_2 = x_diff_hm_2[1:, 1:-1]

    y_diff_hm_1 = extend_hm[:, :-1] - extend_hm[:, 1:]
    y_diff_hm_1 = y_diff_hm_1[1:-1, :-1]

    y_diff_hm_2 = extend_hm[:, 1:] - extend_hm[:, :-1]
    y_diff_hm_2 = y_diff_hm_2[1:-1, 1:]
    
    x_diff_hms = [x_diff_hm_1 != 0, x_diff_hm_2 != 0]
    y_diff_hms = [y_diff_hm_1 != 0, y_diff_hm_2 != 0]

    corner_hm = np.zeros_like(heightmap)
    for xhm in x_diff_hms:
        for yhm in y_diff_hms:
            corner_hm += xhm * yhm

    left_bottom_hm = (x_diff_hm_1 != 0) * (y_diff_hm_1 != 0)

    left_bottom_corners = np.where(left_bottom_hm > 0)
    left_bottom_corners = np.array(left_bottom_corners).transpose()

    corners = np.where(corner_hm > 0)
    corners = np.array(corners).transpose()

    x_borders = list(np.where(x_diff_hm_1.sum(axis=1))[0])
    y_borders = list(np.where(y_diff_hm_1.sum(axis=0))[0])
    
    cross_corners = []
    if get_xy_corners:
        xy_corners = list(itertools.product(x_borders, y_borders))
        xy_corners = np.array(xy_corners)
        
        if len(left_bottom_corners) >= 3:
            try:
                convex_hull = ConvexHull(left_bottom_corners)
                hull_path = Path(left_bottom_corners[convex_hull.vertices])
                xy_inside_hull = hull_path.contains_points(xy_corners)
                # lb_inside_hull = hull_path.contains_points(left_bottom_corner)

                for c in left_bottom_corners:
                    c = list(c)
                    cross_corners.append(c)
                
                for i, c in enumerate(xy_corners):
                    c = list(c)
                    if c[0] == 0 or c[1] == 0:
                        cross_corners.append(c)

                    if xy_inside_hull[i] and c not in cross_corners:
                        cross_corners.append(c)
            except:
                pass

    x_borders.append(hm_shape[0])
    y_borders.append(hm_shape[1])

    return corners, left_bottom_corners, x_borders, y_borders, cross_corners

def compute_bridge_corners(id_map, heightmap, boxes, positions):

    def compute_small_large(box_small, box_large, pos_small, pos_large, axis):

        if pos_small[axis] < pos_large[axis]:
            value_small = pos_large[axis]
        else:
            value_small = pos_small[axis]

        if pos_small[axis] + box_small[axis] < pos_large[axis] + box_large[axis]:
            value_large = pos_small[axis] + box_small[axis]
        else:
            value_large = pos_large[axis] + box_large[axis]
        return value_small, value_large
    
    _, id_left_bottom_corner, id_x_borders, id_y_borders, xy_corners = compute_corners(id_map)
    
    bridge_corners = []

    height_to_id = {}
    id_to_corner = {}

    # max_height = container_size[2]

    bridge_spaces = []

    # id_to_height = {}
    for id_lbc in id_left_bottom_corner:
        box_id = id_map[id_lbc[0], id_lbc[1]]
        height = heightmap[id_lbc[0], id_lbc[1]]

        # TODO allow threshold of height
        
        # id_to_height[ box_id ] = height
        if height not in height_to_id:
            height_to_id[ height ] = []
        if box_id not in height_to_id[ height ]:
            height_to_id[ height ].append(box_id)

        if box_id not in id_to_corner:
            id_to_corner[box_id] = []
        id_to_corner[box_id].append(id_lbc)

    same_height_box_pairs = []

    for height in height_to_id:
        same_height_ids = height_to_id[height]
        if len(same_height_ids) > 1:
            # check neighbor
            for a in same_height_ids:
                for b in same_height_ids:
                    if a >= b: continue

                    corners_a = id_to_corner[a]
                    corners_b = id_to_corner[b]

                    origin_box_a = boxes[a-1]
                    origin_box_b = boxes[b-1]

                    origin_pos_a = positions[a-1]
                    origin_pos_b = positions[b-1]

                    for corner_a in corners_a:
                        for corner_b in corners_b:
                            
                            box_a = [ 
                                origin_box_a[0] - (corner_a[0] - origin_pos_a[0]),
                                origin_box_a[1] - (corner_a[1] - origin_pos_a[1])
                                ]

                            box_b = [ 
                                origin_box_b[0] - (corner_b[0] - origin_pos_b[0]),
                                origin_box_b[1] - (corner_b[1] - origin_pos_b[1])
                                ]
                            
                            pos_a = corner_a
                            pos_b = corner_b

                            min_x = min(pos_a[0], pos_b[0])
                            min_y = min(pos_a[1], pos_b[1])

                            max_x = max(box_a[0] + pos_a[0], box_b[0] + pos_b[0])
                            max_y = max(box_a[1] + pos_a[1], box_b[1] + pos_b[1])
                            
                            neighbor_axis = -1 # no

                            if (max_x - min_x) -  (box_a[0] + box_b[0]) < 0:
                                neighbor_axis = 1 # neighbor along y
                            elif (max_y - min_y) -  (box_a[1] + box_b[1]) < 0:
                                neighbor_axis = 0 # neighbor along x
                            else:
                                continue

                            cross_axis = 1 - neighbor_axis

                            if pos_a[neighbor_axis] < pos_b[neighbor_axis]:
                                box_small = box_a
                                box_large = box_b

                                pos_small = pos_a
                                pos_large = pos_b
                            else:
                                box_small = box_b
                                box_large = box_a

                                pos_small = pos_b
                                pos_large = pos_a

                            threshold = 1e-4
                            gap = abs(pos_large[neighbor_axis] - pos_small[neighbor_axis]) - box_small[neighbor_axis]
                            
                            if gap < threshold:
                                # same_height_box_pairs.append([box_small, box_large, pos_small, pos_large, neighbor_axis, cross_axis])

                                cross_small, cross_large = compute_small_large(box_small, box_large, pos_small, pos_large, cross_axis)
                                neighbor_small = pos_small[neighbor_axis]
                                # neighbor_large = pos_large[neighbor_axis] + box_large[neighbor_axis]

                                left_bottom = [0,0]
                                left_bottom[neighbor_axis] = neighbor_small
                                left_bottom[cross_axis] = cross_small

                                bridge_spaces.append(left_bottom)

                                # right_top = [0,0,0]
                                # right_top[neighbor_axis] = neighbor_large
                                # right_top[cross_axis] = cross_large
                                # right_top[2] = max_height

                                # bridge_spaces.append([ left_bottom, right_top, [1,1,1]])

                            else:
                                continue

    return bridge_spaces, id_x_borders, id_y_borders

def compute_stair_corners(heightmap, corners):


    corners, _, _, _, _ = compute_corners(heightmap)

    stair_hm = np.zeros_like(heightmap)
    corner_heights = heightmap[corners[:,0], corners[:,1]]
    sort_ids = np.argsort(corner_heights)
    sort_corners = corners[sort_ids]

    for c in sort_corners:
        cx, cy = c
        h = heightmap[cx, cy]
        stair_hm[:cx+1, :cy+1] = h
    
    # bin_x, bin_y = heightmap.shape
    # stair_hm = np.zeros([bin_x, bin_y])
    # for xx in reversed(range(bin_x-1)):
    #     for yy in reversed(range(bin_y-1)):
    #         stair_hm[xx, yy] = max(stair_hm[xx+1, yy], stair_hm[xx, yy+1], heightmap[xx, yy])

    _, slb_corner, _, _, _ = compute_corners(stair_hm)
    return slb_corner



def compute_empty_space(container_size, corners, x_borders, y_borders, id_map, heightmap, empty_space_list, x_side='left-right', y_side='left-right', space_type=0, min_ems_width=0, container_id=0):
    '''
    space_type:
        0 # normal
        1 # bridge
    '''
    
    # NOTE find ems from corners
    def check_valid_height_layer(height_layer, space_type):
        if space_type == 0:
            return (height_layer <= 0).all()
        elif space_type == 1:
            return (height_layer == 0).all()
        else:
            assert space_type, "no valid space_type \in {0, 1}"

    for corner in corners:
        x,y = corner
        # h = int(heightmap[x, y])
        h = heightmap[x, y]
        if h == container_size[2]: continue

        h_layer = heightmap - h

        for axes in itertools.permutations(range(2), 2):
            x_small = x
            x_large = x+1
            
            y_small = y
            y_large = y+1

            for axis in axes:
                if axis == 0:
                    if 'left' in x_side:
                        for xb in x_borders:
                            if x_small > xb:
                                # if (h_layer[xb:x, y_small:y_large] <= 0).all():
                                if check_valid_height_layer(h_layer[xb:x, y_small:y_large], space_type):
                                    x_small = xb
                            else: break

                    if 'right' in x_side:
                        for xb in x_borders[::-1]:
                            if x_large < xb:
                                if check_valid_height_layer(h_layer[x:xb, y_small:y_large], space_type):
                                # if (h_layer[x:xb, y_small:y_large] <= 0).all():
                                    x_large = xb
                            else: break
                
                elif axis == 1:
                    if 'left' in y_side:
                        for yb in y_borders:
                            if y_small > yb:
                                if check_valid_height_layer(h_layer[ x_small:x_large, yb:y], space_type):
                                # if (h_layer[ x_small:x_large, yb:y] <= 0).all():
                                    y_small = yb
                            else: break

                    if 'right' in y_side:
                        for yb in y_borders[::-1]:
                            if y_large < yb:
                                if check_valid_height_layer(h_layer[ x_small:x_large, y:yb], space_type):
                                # if (h_layer[ x_small:x_large, y:yb] <= 0).all():
                                    y_large = yb
                            else: break

            # if (h_layer[ x_small:x_large, y_small:y_large] <= 0).all():
            ems_h_layer = h_layer[ x_small:x_large, y_small:y_large]

            if check_valid_height_layer(ems_h_layer, space_type):
                
                w = x_large - x_small
                l = y_large - y_small
                if w == 0:
                    w = 1
                if l == 0:
                    l = 1

                # h_size = (ems_h_layer == 0).sum() / ( 1.0 * w * l )

                # new_ems = [[x_small, y_small, h], [x_large, y_large, container_size[2]], [h_size]*3, [container_id]*3 ]
                new_ems = [[x_small, y_small, h], [x_large, y_large, container_size[2]],[container_id]*3 ]

                # NOTE remove small ems
                if min_ems_width > 0:
                    if x_large - x_small < min_ems_width or y_large - y_small < min_ems_width:
                        new_ems = None

                if space_type == 1:
                    id_layer = id_map[ x_small:x_large, y_small:y_large]
                    if len(np.unique(id_layer)) == 1:
                        new_ems = None

                if new_ems is not None and new_ems not in empty_space_list:
                    empty_space_list.append( new_ems )

def compute_ems(id_maps, heightmaps, container_size, min_ems_width=0, ems_type='ems-id-stair', valid_container_mask=None):
    empty_max_spaces = []
    # corners, left_bottom_corners, x_borders, y_borders, xy_corners = compute_corners(heightmap)

    # if boxes is not None:
    #     # NOTE bridge corners, maybe no to use, they need boxes and positions
    #     bridge_corners, id_x_borders, id_y_borders = compute_bridge_corners(id_map, heightmap, boxes, positions)
    #     compute_empty_space(container_size, bridge_corners, id_x_borders, id_y_borders, id_map, heightmap, empty_max_spaces, 'right', 'right', 1, min_ems_width=min_ems_width)

    # NOTE xy_corners
    # corners, id_left_bottom_corners, x_borders, y_borders, xy_corners = compute_corners(heightmap, True)
    # compute_empty_space(container_size, xy_corners, x_borders, y_borders, id_map, heightmap, empty_max_spaces, 'right', 'right', min_ems_width=min_ems_width)

    container_num = len(heightmaps)

    if valid_container_mask is None:
        valid_container_mask = np.ones(container_num)
    else:
        valid_container_mask = np.array(valid_container_mask)

    for container_id in range(len(heightmaps)):
        if valid_container_mask[container_id] == 0: continue
        
        heightmap = heightmaps[container_id].copy()
        id_map = id_maps[container_id]

        # id_map = heightmap.copy()
        
        corners = None
        # NOTE normal_corners
        if 'id' in ems_type:
            corners, id_left_bottom_corners, x_borders, y_borders, xy_corners = compute_corners(id_map)
            compute_empty_space(container_size, id_left_bottom_corners, x_borders, y_borders, id_map, heightmap, empty_max_spaces, 'right', 'right', min_ems_width=min_ems_width, container_id=container_id)

        # NOTE stair corners
        if 'stair' in ems_type:
            if corners is None:
                corners, id_left_bottom_corners, x_borders, y_borders, xy_corners = compute_corners(id_map)
            stair_corners = compute_stair_corners(heightmap, corners)
            compute_empty_space(container_size, stair_corners, x_borders, y_borders, id_map, heightmap, empty_max_spaces, 'right', 'right', min_ems_width=min_ems_width, container_id=container_id)

        # NOTE ems
        # compute_empty_space(container_size, id_left_bottom_corners, x_borders, y_borders, id_map, heightmap, empty_max_spaces, 'left', 'left', min_ems_width=min_ems_width, container_id=container_id)
        if 'ems' in ems_type:
            if corners is None:
                corners, hei_left_bottom_corners, x_borders, y_borders, xy_corners = compute_corners(id_map)
            compute_empty_space(container_size, corners, x_borders, y_borders, id_map, heightmap, empty_max_spaces, 'left-right', 'left-right', min_ems_width=min_ems_width, container_id=container_id)

    return empty_max_spaces

def compute_box_ems_mask(box_states, ems, box_num=None, all_container_space=None, heightmaps=None, packing_mask=None, remove_list=None, \
                         check_box_stable=True, gripper_size=None, check_z=True, \
                            same_height_threshold=0, container=None, corner_num=1):
    # can place or not

    if False:
    # if True:
        box_states = box_states.copy()
        ems = ems.copy()

        # NOTE 这里近似 EMS
        net_w = 20
        real_w = 100
        reso = real_w / net_w

        max_s = (net_w-1) * reso

        ems_pos = ems[:,:3]
        ems_size = ems[:,3:6]
        ems_top = ems_pos + ems_size

        ems_pos = np.ceil((ems_pos / real_w * net_w)) * reso
        ems_top = np.ceil(ems_top / real_w * net_w)* reso
        ems_size = ems_top - ems_pos
        ems[:,:3] = ems_pos
        ems[:,3:6] = ems_size

        box_states = np.ceil(box_states / real_w * net_w) * reso
        ems = np.ceil(ems / real_w * net_w) * reso
    

    # box_states: state_num * 3
    # ems: ems_num * 6
    
    # pos | size, ems_len x 6
    ems_x = ems[:, 3:4]
    ems_y = ems[:, 4:5]
    ems_z = ems[:, 5:6]
    
    box_x = box_states[:, 0][None, :]
    box_y = box_states[:, 1][None, :]
    box_z = box_states[:, 2][None, :]

    ems_to_box_x = (ems_x - box_x) >= 0
    ems_to_box_y = (ems_y - box_y) >= 0
    ems_to_box_z = (ems_z - box_z) >= 0

    if check_z:
        ems_to_box_mask = ems_to_box_x * ems_to_box_y * ems_to_box_z
    else:
        ems_to_box_mask = ems_to_box_x * ems_to_box_y

    ems_size_mask = ems_to_box_mask.copy()

    ems_box_grasp_mask = ems_to_box_mask.copy() * 1
    
    ems_to_box_mask = ems_to_box_mask[None, :, :].repeat(corner_num, 0)

    rotate_states = None
    if box_num is None:
        # TODO remove fix number 3
        rotate_states = [ p for p in itertools.permutations(range(3), 3) ]
        box_num = int(len(box_states) / len(rotate_states))

    if gripper_size is not None:
        valid_pair = np.where(ems_to_box_mask > 0)
        valid_pair = np.column_stack(valid_pair)

        for pair in valid_pair:
            corner_idx = pair[0]
            ems_idx = pair[1]
            box_idx = pair[2]

            if remove_list is not None:
                real_box_idx = box_idx % box_num
                if real_box_idx in remove_list: continue

            # if ems.shape[1] == 7:
            if ems.shape[1] == 6:
                container_id = 0
            else:
                container_id = int(ems[ems_idx][-1])
            lx, ly, lz = ems[ems_idx][:3].astype('int')
            ex, ey, ez = ems[ems_idx][3:6].astype('int')
            bx, by, bz = box_states[box_idx].astype('int')

            if corner_idx == 0:
                lx, ly = lx, ly
            elif corner_idx == 1:
                lx += ex - bx
                ly = ly
            elif corner_idx == 2:
                lx = lx
                ly += ey - by
            elif corner_idx == 3:
                lx += ex - bx
                ly += ey - by
            elif corner_idx == 4:
                lx += int(ex/2 - bx/2)
                ly += int(ey/2 - by/2)
            
            # check gripper region
            gx, gy, gz = gripper_size

            hm = heightmaps[container_id]

            x_left = max( 0, int(lx + bx/2 - gx / 2) )
            x_right = int(lx + bx/2 + gx / 2)
            
            y_left = max( 0, int(ly + by/2 - gy / 2) )
            y_right = int(ly + by/2 + gy / 2)
            
            max_h = np.max(hm[ x_left:x_right, y_left:y_right ])

            # plt.plot([y_left, y_left], [x_left, x_right] )
            # plt.plot([y_right, y_right], [x_left, x_right] )
            # plt.plot([y_left, y_right], [x_left, x_left] )
            # plt.plot([y_left, y_right], [x_right, x_right] )

            if max_h - (lz + bz) > gz:
                ems_to_box_mask[ corner_idx, ems_idx, box_idx] = False
                


    if packing_mask is not None or check_box_stable:
        # or gripper_size is not None:
        
        valid_pair = np.where(ems_to_box_mask > 0)
        valid_pair = np.column_stack(valid_pair)

        for pair in valid_pair:
            corner_idx = pair[0]
            ems_idx = pair[1]
            box_idx = pair[2]

            if remove_list is not None:
                real_box_idx = box_idx % box_num
                if real_box_idx in remove_list: continue

            # only consider z rotation
            if rotate_states is not None:
                if rotate_states[box_idx // box_num][2] != 2: continue
    
            # if ems.shape[1] == 7:
            if ems.shape[1] == 6:
                container_id = 0
            else:
                container_id = int(ems[ems_idx][-1])

            lx, ly, lz = ems[ems_idx][:3].astype('int')
            ex, ey, ez = ems[ems_idx][3:6].astype('int')
            bx, by, bz = box_states[box_idx].astype('int')

            if corner_idx == 0:
                lx, ly = lx, ly
            elif corner_idx == 1:
                lx += ex - bx
                ly = ly
            elif corner_idx == 2:
                lx = lx
                ly += ey - by
            elif corner_idx == 3:
                lx += ex - bx
                ly += ey - by
            elif corner_idx == 4:
                lx += int(ex/2 - bx/2)
                ly += int(ey/2 - by/2)
            
            lx = int(lx)
            ly = int(ly)

            bx = int(bx)
            by = int(by)

            
            # check in packing space
            in_packing_mask = True
            if packing_mask is not None:
                mask_under_box = packing_mask[lx:lx+bx, ly:by+ly]
                if (mask_under_box == 0).any():
                    in_packing_mask = False

            # check stable
            if check_box_stable and in_packing_mask:
                if len(all_container_space) > 0:
                    space = all_container_space[container_id]
                    is_stable = space.drop_box_virtual([bx,by,bz], [lx, ly], False, 1, 1, same_height_threshold = same_height_threshold, 
                                                       id_map=container.each_container_idmap[container_id])
                    
                    if container is not None:
                        if container.duo is not None:
                            hm = heightmaps[container_id]

                            real_lz = hm[ lx:lx+bx, ly:ly+by ].max()
                            new_boxes = container.each_container_boxes[container_id] + [np.array([bx, by, bz])]
                            new_poses = container.each_container_positions[container_id] + [ np.array([ lx, ly, real_lz ]) ]
                            duo_stable = container.duo.check_stable( new_boxes, new_poses )
                            is_stable =  is_stable and duo_stable


                    # layer_under_box = heightmap[lx:bx+lx, ly:ly+by]
                    # is_stable = check_stable([bx,by,bz], [lx, ly, lz], layer_under_box)
                    
                else:
                    # layer_under_box = heightmap[lx:x+lx, ly:ly+y]
                    # is_stable = check_stable([x,y,z], [lx, ly, lz], layer_under_box)
                    # is_stable = False
                    is_stable = True
            else:
                # layer_under_box = None
                is_stable = True
                # is_stable = False

            if not in_packing_mask or not is_stable:
                ems_to_box_mask[corner_idx, ems_idx, box_idx] = False

                if not in_packing_mask:
                    ems_size_mask[pair[0], pair[1]] = False

    return ems_size_mask, ems_to_box_mask, ems_box_grasp_mask

def normal_size(box, ems, container_size, unit_scale=1, scale_to_large=False):
    unit_scale = unit_scale
    heightmap_width = container_size[0]
    heightmap_length = container_size[1]

    if heightmap_length < heightmap_width:
        max_height = heightmap_width
    else:
        max_height = heightmap_length

    max_height = container_size[2]

    box[:,0] /= (heightmap_width * 1.0)
    ems[:,0] /= (heightmap_width * 1.0)
    ems[:,3] /= (heightmap_width * 1.0)

    box[:,1] /= (heightmap_length * 1.0)
    ems[:,1] /= (heightmap_length * 1.0)
    ems[:,4] /= (heightmap_length * 1.0)

    box[:,2] /= (max_height * 1.0)
    ems[:,2] /= (max_height * 1.0)
    ems[:,5] /= (max_height * 1.0)

    # # scale into 100

    # box *= 100
    # ems *= 100
    
    if scale_to_large:
    # if True:
        # box[:,0] *= 100.0
        # ems[:,0] *= 100.0
        # ems[:,3] *= 100.0
        
        # box[:,1] *= 100.0
        # ems[:,1] *= 100.0
        # ems[:,4] *= 100.0

        # box[:,2] *= 100.0
        # ems[:,2] *= 100.0
        # ems[:,5] *= 100.0

        box[:,0] *= heightmap_width / unit_scale
        ems[:,0] *= heightmap_width / unit_scale
        ems[:,3] *= heightmap_width / unit_scale
        
        box[:,1] *= heightmap_length / unit_scale
        ems[:,1] *= heightmap_length / unit_scale
        ems[:,4] *= heightmap_length / unit_scale

        box[:,2] *= max_height / unit_scale
        ems[:,2] *= max_height / unit_scale
        ems[:,5] *= max_height / unit_scale



    # box[:,0] /= 100.0
    # ems[:,0] /= 100.0
    # ems[:,3] /= 100.0

    # box[:,1] /= 100.0
    # ems[:,1] /= 100.0
    # ems[:,4] /= 100.0

    # # ems_max_height = np.max(ems[:,2])
    # # if ems_max_height > 0:
    # #     max_height = ems_max_height
    # box[:,2] /= 100.0
    # ems[:,2] /= 100.0
    # ems[:,5] /= 100.0

    # box[:,0] *= 100
    # ems[:,0] *= 100
    # ems[:,3] *= 100
    
    # box[:,1] *= 100
    # ems[:,1] *= 100
    # ems[:,4] *= 100

    # box[:,2] *= 100
    # ems[:,2] *= 100
    # ems[:,5] *= 100
    
    return box, ems

