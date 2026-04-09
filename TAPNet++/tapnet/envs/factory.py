import numpy as np
import os
from tqdm import tqdm

from scipy.spatial.transform import Rotation
import copy
from .container import Container, left_bottom, ppsg_cut


x_ = np.linspace(-1., 1., 20)
y_ = np.linspace(-1., 1., 20)
z_ = np.linspace(-1., 1., 20)
x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
POINTS_INSIDE_BOX = np.concatenate([[x,y,z]]).reshape(3, -1).transpose(1,0)

data_prec_list = []
data_box_list = []

def load_fact(data_folder):
    global data_prec_list
    global data_box_list

    if len(data_box_list) > 0:
        return
    
    print("Loading pre data ...")
    for i in tqdm(range(1000)):
        box_path = os.path.join(data_folder, f"{i}_box.npy")
        prec_path = os.path.join(data_folder, f"{i}_pre.npy")

        box_list = list(np.load(box_path))
        data_box_list.append(box_list)
        
        if os.path.exists(prec_path):
            prec_graph = np.load(prec_path, allow_pickle=True).item()
            data_prec_list.append(prec_graph)



class BlockNode(object):
    def __init__(self, node_index, size, pos, rot=None) -> None:
        self.index = node_index
        self.blocks_to = {
            'x-left': [],
            'x-right': [],
            'y-left': [],
            'y-right': [],
            'z-left': [],
            'z-right': [],
            'move': []
        }
        self.blocked_by = {
            'x-left': [],
            'x-right': [],
            'y-left': [],
            'y-right': [],
            'z-left': [],
            'z-right': [],
            'move': []
        }

        self.side_spaces = {
            'x-left': None,
            'x-right': None,
            'y-left': None,
            'y-right': None,
            'z-left': None,
            'z-right': None,
            'move': None
        }

        self.side_codes = {
            'x': None,
            'y': None,
            'z': None,
        }

        self.size = np.array(size)
        self.pos = np.array(pos)
        self.rot = np.array(rot)

        self.bottom_pos = np.array(pos)
        if rot is not None:
            z = rot[:,2]
            self.bottom_pos += z * size[2] / 2

    def can_access(self, access_type):
        return len(self.blocked_by['move']) == 0 and (  len(self.blocked_by[access_type + '-left']) == 0 or len(self.blocked_by[access_type + '-right']) == 0  )
        # return np.sum(self.side_codes[access_type]) == 0

    def count_blocks_to(self):
        count = 0
        for key in self.blocks_to:
            count += len(self.blocks_to[key])
        return count 
        
    def encode(self, access_type, max_len, prec_dim = 2, recompute=False, select_idx=[]):

        if self.side_codes[access_type] is None or recompute == True:
            code = np.zeros((max_len, prec_dim))

            if prec_dim == 2:
                code[ self.blocked_by['move'], 0 ] = 1
                code[ self.blocked_by[access_type + '-left'], 1 ] = 1
                code[ self.blocked_by[access_type + '-right'], 1 ] = 1
            else:
                code[ self.blocked_by['move'], 0 ] = 1
                code[ self.blocked_by[access_type + '-left'], 1 ] = 1
                code[ self.blocked_by[access_type + '-right'], 2 ] = 1
            
            self.side_codes[access_type] = code

        ret = self.side_codes[access_type]
        if len(select_idx) > 0:
            ret = ret[select_idx]
            if len(select_idx) > max_len:
                # example: [1,2,4,5,-1-1-1]
                ret[ max_len: ] = 0
        
        return ret


class PrecedenceGraph(object):
    def __init__(self) -> None:
        self.nodes = []
        self.node_num = 0
    
    def reset(self):
        self.nodes.clear()
        self.node_num = 0
    
    def add_node(self, node_id, size, pos, rot=None):
        node = BlockNode(node_id, size, pos, rot)
        self.nodes.append(node)
        self.node_num += 1

    def remove_node(self, node_index):
        node = self.nodes[node_index]
        # self.nodes.remove(node)
        for n in self.nodes:
            self.remove_blocked(n, node)
            

    def block_to(self, a_index, b_index, block_to_type: str):
        a = self.nodes[a_index]
        b = self.nodes[b_index]        
        a.blocks_to[block_to_type].append(b.index)
        b.blocked_by[block_to_type].append(a.index)

    def remove_blocked(self, a, b):
        # NOTE remove node_b data in node_a
        for key in a.blocked_by:
            if b.index in a.blocked_by[key]:
                a.blocked_by[key].remove(b.index)
                b.blocks_to[key].remove(a.index)
        
        for axis in ['x', 'y', 'z']:
            a.side_codes[axis][ b.index ] = 0
                
    def _encode(self, access_types = ['x', 'y', 'z'], prec_dim=2):        
        all_node_code = []

        for access_type in access_types:
            node_code = np.zeros((self.node_num, self.node_num, prec_dim))
            for node in self.nodes:
                node_code[node.index] = node.encode( access_type, self.node_num, prec_dim)
            all_node_code.append(node_code)

        return all_node_code

    def _compute_side_spaces(self, node:BlockNode, node_type, gripper_width, container_size=None):
        '''
        compute 6 side space of box + top space on the top
        '''
        if node_type == 'fake':
            max_z = container_size[2]

            bx, by, bz = node.size
            x, y, z = node.pos
            x = int(x)
            y = int(y)

            bottom_z = z
            # bottom_z = np.floor(z + bz/2)

            x_left = x - gripper_width
            if x == 0:
                node.side_spaces['x-left'] = None
            elif x_left < 0:
                node.side_spaces['x-left'] = [ 0, y, bottom_z, x, by, max_z ]
            else:
                node.side_spaces['x-left'] = [ x_left, y, bottom_z, gripper_width, by, max_z ]

            x_right = x + bx + gripper_width
            if x == container_size[0] - 1:
                node.side_spaces['x-right'] = None
            if x_right > container_size[0]:
                node.side_spaces['x-right'] = [ x + bx, y, bottom_z, container_size[0] - x - bx, by, max_z ]
            else:
                node.side_spaces['x-right'] = [ x + bx, y, bottom_z, gripper_width, by, max_z ]

            y_left = y - gripper_width
            if y == 0:
                node.side_spaces['y-left'] = None
            elif y_left < 0:
                node.side_spaces['y-left'] = [ x, 0, bottom_z, bx, y, max_z ]
            else:
                node.side_spaces['y-left'] = [ x, y_left, bottom_z, bx, gripper_width, max_z ]

            y_right = y + by + gripper_width
            if y == container_size[1] - 1:
                node.side_spaces['y-right'] = None
            elif y_right > container_size[1]:
                node.side_spaces['y-right'] = [ x, y + by, bottom_z, bx, container_size[1] - y - by, max_z ]
            else:
                node.side_spaces['y-right'] = [ x, y + by, bottom_z, bx, gripper_width, max_z ]

            node.side_spaces['z-left'] = None
            node.side_spaces['z-right'] = [ x, y, z + bz, bx, by, max_z ]

            node.side_spaces['move'] = [ x, y, z + bz, bx, by, max_z ]

        elif node_type == 'sim':

            pos = node.pos
            rot = node.rot
            size = node.size
            
            xaixs = rot[:3,0]
            yaixs = rot[:3,1]
            zaixs = rot[:3,2]
            
            gripper_offset = gripper_width / 2

            node.side_spaces['x-left'] = BlockNode(-1, [gripper_width, size[1], size[2]], pos + xaixs * (size[0] / 2 + gripper_offset) , rot )
            node.side_spaces['x-right'] = BlockNode(-1, [gripper_width, size[1], size[2]], pos - xaixs * (size[0] / 2 + gripper_offset) , rot )
            node.side_spaces['y-left'] = BlockNode(-1, [size[0], gripper_width, size[2]], pos + yaixs * (size[1] / 2 + gripper_offset) , rot )
            node.side_spaces['y-right'] = BlockNode(-1, [size[0], gripper_width, size[2]], pos - yaixs * (size[1] / 2 + gripper_offset) , rot )
            node.side_spaces['z-left'] = BlockNode(-1, [size[0], size[1], gripper_width], pos + zaixs * (size[2] / 2 + gripper_offset) , rot )
            node.side_spaces['z-right'] = BlockNode(-1, [size[0], size[1], gripper_width], pos - zaixs * (size[2] / 2 + gripper_offset) , rot )

            node.side_spaces['move'] = BlockNode(-1, size, pos + [0,0, gripper_width] , rot )

    def _intersect(self, node_type,  space1, space2):
        '''
        node_type == 'fake'
            space1 is a list [pos, size]
        node_type == 'sim'
            sapce1 is a node
        '''
        
        if space1 is None or space2 is None: return False
        
        if node_type == 'fake':
            space1 = list(space1.pos) + list(space1.size)

            
            x1, y1, z1 = space1[:3]
            w1, l1, h1 = space1[3:6]
            
            x2, y2, z2 = space2[:3]
            w2, l2, h2 = space2[3:6]

            minx = min(x1, x2)
            miny = min(y1, y2)
            minz = min(z1, z2)

            maxx = max(x1 + w1, x2 + w2)
            maxy = max(y1 + l1, y2 + l2)
            maxz = max(z1 + h1, z2 + h2)
            
            if (maxx - minx < w1 + w2) and (maxy - miny < l1 + l2) and (maxz - minz < h1 + h2):
                return True
            else:
                return False

        elif node_type == 'sim':
            s1 = space1.size
            p1 = space1.pos
            r1 = space1.rot
            
            s2 = space2.size
            p2 = space2.pos
            r2 = space2.rot
            pose2 = np.eye(4)
            pose2[:3,:3] = r2
            pose2[:3,3] = p2

            points1 = POINTS_INSIDE_BOX * s1 / 2 * 0.999 # smaller a little  
            points1 = np.einsum( 'ij,aj->ai', r1, points1 ) + p1

            inv_pose2 = np.linalg.inv(pose2)
            rot = inv_pose2[:3,:3]
            trans = inv_pose2[:3,3]
            points1_under_axis2 = np.einsum( 'ij,aj->ai', rot, points1 ) + trans

            def is_point_inside( points, cube_size ):
                abs_points = abs(points)
                sub = cube_size/2.0 - abs_points
                return (( sub > 0).sum(axis=1) == 3).sum() > 0

            is_inside = is_point_inside( points1_under_axis2, s2 )
            return is_inside

        else:
            assert 'only node_type in {fake, sim}'

    def build_graph( self, node_type, gripper_width, container_size=None, my_scene=None ):
        '''
            node_type: string { 'fake', 'sim', 'belt' }
                1. fake means node.pos is its left_bottom corner, use grid to represent the sapce
                2. sim means node.pos / node.rot is real pose, box in real 3D world
                3. belt means node follow a fixed order
        '''
        
        if node_type in {'fake', 'sim', 'belt'}:
            for node in self.nodes:
                self._compute_side_spaces(node, node_type, gripper_width, container_size)
        
        side_keys = self.nodes[0].blocks_to.keys()

        # get around space of each box/node
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                if i == j: continue
                node_i = self.nodes[i]
                node_j = self.nodes[j]
                for key in side_keys:
                    # if node_type == 'fake':
                    if self._intersect(node_type, node_i, node_j.side_spaces[key]):
                        self.block_to(node_i.index, node_j.index, key)

        if my_scene is not None:
            for i in range(len(self.nodes)):
                node_i = self.nodes[i]
                
                size = node_i.size.copy()
                rot = node_i.rot.copy()
                position = node_i.pos.copy()

                for key in side_keys:
                    if key == 'move': continue
                    
                    if node_i.side_spaces[key].pos[2] <= -0.11:
                        self.block_to(node_i.index, node_i.index, key)

                    else:
                        if 'x' in key:
                            axis_id = 0
                        elif 'y' in key:
                            axis_id = 1
                        elif 'z' in key:
                            axis_id = 2

                        if 'left' in key:
                            dir_sign = 1
                        elif 'right' in key:
                            dir_sign = -1

                        axis = rot[:,axis_id]
                        final_pos = position + dir_sign * axis * size[axis_id] / 2

                        if axis_id == 0:
                            final_rot = rot @ Rotation.from_rotvec( dir_sign * np.pi/2 * np.array([0,1,0])).as_matrix()
                        elif axis_id == 1:
                            final_rot = rot @ Rotation.from_rotvec( dir_sign * np.pi/2 * np.array([1,0,0])).as_matrix()
                        else:
                            final_rot = rot
                        
                        final_pose = np.eye(4)
                        final_pose[:3,:3] = final_rot
                        final_pose[:3,3] = final_pos

                        node_i.side_spaces[key].final_pose = final_pose

                        # check final dir
                        final_rot_z = final_rot[:,2]
                        min_z_index = np.argmax( np.abs(final_rot_z) )

                        if min_z_index != 2:
                            self.block_to(node_i.index, node_i.index, key)
                            pass
                            # no vertical !!!
                            # if abs(final_rot_z[min_z_index]) > 0.9:
                            #     self.block_to(node_i.index, node_i.index, key)

                            # if final_pos[2] < -0.05:
                            #     self.block_to(node_i.index, node_i.index, key)

                        else:
                            valid = my_scene.robot.check_valid_mat(final_pose)
                            if not valid:
                                self.block_to(node_i.index, node_i.index, key)
                    


        for i in range(len(self.nodes)):
            node = self.nodes[i]
            for axis in ['x', 'y', 'z']:
                code = node.encode( axis, self.node_num )

class Factory(object):
    def __init__(self, fact_type, data_type, targe_container_size=None, gripper_width=10, require_box_num=None, data_folder=None, simulation_scene=None) -> None:
        '''
        fact_type: string { 'box', 'tap_fake', 'tap_sim', 'tap_belt', 'conveyor_fake', 'conveyor_sim' }
        data_type: string { 'rand', 'ppsg', 'fix' }
            1. random
            2. perfect packing guarantee packing
        target_container_size: list [w, l, h]
        '''

        assert fact_type in [ 'box', 'tap_fake', 'tap_sim', 'tap_belt', 'conveyor_fake', 'conveyor_sim' ], "fact_type only support { 'box', 'tap_fake', 'tap_sim', 'tap_belt', 'conveyor_fake', 'conveyor_sim' }"
        assert data_type in [ 'rand', 'ppsg', 'fix' ], "data_type only support {'rand', 'ppsg', 'fix' }"
        
        self.data_type = data_type
        self.fact_type = fact_type
        self.node_type = fact_type.split('_')[-1]

        self.data_folder = data_folder

        if self.data_folder is not None:
            load_fact(self.data_folder)

        self.select_list = []
        self.box_list = []
        self.remove_list = []

        self.require_box_num = require_box_num

        self.simulation_scene = simulation_scene

        self.size_scale = 0


        self.gripper_width = gripper_width
        self.precedence_graph = None

        if data_type == 'ppsg' or fact_type in ['tap_fake', 'conveyor_fake']:
            assert targe_container_size is not None, "target_container_size not found"
        
        if targe_container_size is not None:
            self.targe_container_size = targe_container_size
            w, l, h = self.targe_container_size
            h = max(w, l)
            self.ppsg_container_size = [ w, l, h]

        if fact_type in ['tap_fake', 'tap_sim']:
            source_scale_rate = 1.4

            if fact_type == 'tap_sim':
                source_scale_rate = 1.9

            source_container_size = [ int(targe_container_size[i] * source_scale_rate) for i in range(3) ]
            source_container_size[2] = 10000
            self.source_container_size = source_container_size
            self.source_container = Container(source_container_size, 'C', \
                                same_height_threshold=0, min_ems_width=0, \
                                stable_rule='none', stable_scale_factor=0, use_bridge=False, world_type='ideal', container_type='single', pack_type='last' )
        elif fact_type in [ 'tap_belt' ]:
            source_container_size = [ int(targe_container_size[i]) for i in range(3) ]
            source_container_size[2] = 100000
            self.source_container_size = source_container_size
            self.source_container = Container(source_container_size, 'C', \
                                same_height_threshold=0, min_ems_width=0, \
                                stable_rule='none', stable_scale_factor=0, use_bridge=False, world_type='ideal', container_type='single', pack_type='last' )
            
        else:
            self.source_container = None
        self.precedence_graph = PrecedenceGraph()


    def new_order(self, box_range, box_num, box_list=None, prec_graph=None):
        assert box_list is None or type(box_list) == list, "box_list must be a list"

        if box_list is None:
            if self.data_type == 'rand':
                # if self.min_height_diff > 0:
                # rand_boxes = np.random.randint(box_range[0], box_range[1]+1, (box_num, 3) )
                # rand_boxes[:,2] = np.random.randint(10, 30, box_num)
                # box_list = list(rand_boxes)

                size_list = [ i for i in range(box_range[0], box_range[1]+1) ]
                mu = 0.5
                sigma = 0.16
                prob_x = np.linspace(mu - 3*sigma, mu + 3*sigma, len(size_list))
                prob_blocks = np.exp( - (prob_x-mu)**2 / (2*sigma**2) ) / (np.sqrt(2*np.pi) * sigma)
                prob_blocks = prob_blocks / np.sum(prob_blocks)
                box_list = list(np.random.choice(size_list, (box_num, 3), p=prob_blocks ))

                # box_list = list(np.random.randint(box_range[0], box_range[1]+1, (box_num, 3) ))
            
            elif self.data_type == 'ppsg':
                box_list, _ = ppsg_cut(box_num, box_range, self.ppsg_container_size)
                # rand rot z
                for i in range(box_num):
                    if np.random.rand() > 0.5:
                        box_list[i] = box_list[i][[1,0,2]]

            elif self.data_type == 'fix':
                candidate_num = 5
                # candidate_boxes = np.random.randint(box_range[0], box_range[1]+1, (candidate_num, 3) )
                
                size_list = [ i for i in range(box_range[0], box_range[1]+1) ]
                mu = 0.5
                sigma = 0.16
                prob_x = np.linspace(mu - 3*sigma, mu + 3*sigma, len(size_list))
                prob_blocks = np.exp( - (prob_x-mu)**2 / (2*sigma**2) ) / (np.sqrt(2*np.pi) * sigma)
                prob_blocks = prob_blocks / np.sum(prob_blocks)
                candidate_boxes = np.random.choice(size_list, (candidate_num, 3), p=prob_blocks )

                candiate_ids = [i for i in range(candidate_num)]
                sample_ids = np.random.choice(candiate_ids, box_num, replace=True)
                box_list = list(candidate_boxes[sample_ids])

        self.box_list = box_list
        
        # vol = 0
        # for b in box_list:
        #     vol += b[0] * b[1] * b[2]
        # print( vol / ( self.targe_container_size[0] * self.targe_container_size[1] * self.targe_container_size[2] ) )

        self.precedence_graph.reset()
        if self.source_container is not None:
            self.source_container.clear_container()

        if self.fact_type == 'box':
            if prec_graph is None:
                pos_list = []
                for box_i, box in enumerate(self.box_list):
                    pos = [0,0,0]
                    self.precedence_graph.add_node(box_i, box, pos)
                self.precedence_graph.build_graph(self.node_type, self.gripper_width)
            else:
                self.precedence_graph = prec_graph
            pass

        elif self.fact_type == 'tap_fake':
            if prec_graph is None:
                pos_list = []
                for box_i, box in enumerate(self.box_list):
                    pos = left_bottom(self.source_container, box, find_ems=False, ret_ems=True)
                    
                    # this only work for fixorder
                    # pos = self.source_container.add_new_box(box, None, 0, False, real_pos=[0,0,0,0])
                    # self.source_container.update_ems()
                    
                    pos_list.append(pos)
                    self.precedence_graph.add_node(box_i, box, pos)
                self.precedence_graph.build_graph(self.node_type, self.gripper_width, self.source_container.container_size)
                self.pos_list = pos_list
            else:
                self.precedence_graph = prec_graph

        elif self.fact_type == 'tap_sim':
            pos_list = []
            self.size_scale = 6
            for box_i, box in enumerate(self.box_list):
                pos = left_bottom(self.source_container, box + [self.size_scale, self.size_scale, 0], ret_ems=True)
                pos_list.append(pos)
            self.pos_list = pos_list

        elif self.fact_type == 'tap_belt':
            if prec_graph is None:
                pos_list = []
                for box_i, box in enumerate(self.box_list):
                    # this only work for fixorder
                    pos = self.source_container.add_new_box(box, None, 0, False, real_pos=[0,0,0,0])
                    self.source_container.update_ems()

                    pos_list.append(pos)
                    self.precedence_graph.add_node(box_i, box, pos)
                self.precedence_graph.build_graph(self.node_type, self.gripper_width, self.source_container.container_size)
                self.pos_list = pos_list
                
            else:
                self.precedence_graph = prec_graph


        if prec_graph is not None:
            self.precedence_graph = prec_graph

    def update_order(self, box_poses, box_sizes, boxes):
        self.new_order(None, None, list(boxes))

        self.precedence_graph.reset()
        for bi in range(len(box_sizes)):
            pose = box_poses[bi]
            size = box_sizes[bi]
            self.precedence_graph.add_node(bi, size, pose[:3,3], pose[:3,:3])
        self.precedence_graph.build_graph('sim', self.gripper_width, my_scene=self.simulation_scene)



    def load_order(self, index):
        assert self.data_folder is not None, "data_folder must exist"
        self.box_list = copy.deepcopy(data_box_list[index])

        if self.precedence_graph is not None:
            self.precedence_graph = copy.deepcopy(data_prec_list[index])

    def remove_box(self, box_id, for_test):

        if len(self.select_list) > 0:
            real_id = self.select_list[box_id]
        else:
            real_id = box_id

        if for_test:
            self.box_list.pop(real_id)
        else:
            self.remove_list.append(real_id)
            if self.fact_type in ['tap_fake', 'tap_belt']:
                self.precedence_graph.remove_node(real_id)

    def is_order_finish(self):
        return len(self.box_list) == 0 or ( len(self.box_list) == len(self.remove_list) )

    def reset(self) :
        self.box_list.clear()
        self.remove_list.clear()
        self.select_list.clear()

        if self.fact_type in ['tap_fake', 'tap_sim', 'tap_belt']:
            self.precedence_graph.reset()
            self.source_container.clear_container(with_reward=True)
        # TODO else

    def update_select_list(self, rotate_axes=['z']):
        box_num = len(self.box_list)
        self.select_list.clear()

        if self.require_box_num is None:
            self.select_list = [ i for i in range( box_num ) ]
        elif self.require_box_num >= box_num:
            self.select_list = [ i for i in range( box_num ) ]
            self.select_list += [ -1 ] * (self.require_box_num - box_num)
        else:
            # select based on precedence
            if self.fact_type in ['tap_fake', 'tap_belt']:
                # find valid node
                for node in self.precedence_graph.nodes:
                    if node.index in self.remove_list: continue
                    for axis in rotate_axes:
                        if node.can_access(axis):
                            self.select_list.append(node.index)
                            break
                    
                    if len(self.select_list) == self.require_box_num:
                        break
                
                block_types = self.precedence_graph.nodes[0].blocks_to.keys()

                enough = len(self.select_list) == self.require_box_num
                
                for node_index in self.select_list:
                    if enough: break
                    if node_index in self.remove_list: continue

                    for block_type in block_types:
                        if enough: break
                        for blocked_index in self.precedence_graph.nodes[node_index].blocks_to[block_type]:
                            if blocked_index not in self.select_list:
                                self.select_list.append(blocked_index)
                            if len(self.select_list) == self.require_box_num:
                                enough = True
                                break

            else:
                no_remove_list = []
                for i in range(box_num):
                    if i not in self.remove_list:
                        no_remove_list.append(i)
                    if len(no_remove_list) == self.require_box_num:
                        break

                self.select_list = no_remove_list

            if len(self.select_list) < self.require_box_num:
                num_diff = self.require_box_num - len(self.select_list)
                self.select_list += self.remove_list[ :num_diff ]
                        
            assert len(self.select_list) == self.require_box_num, "wrong select_list"

    def get_box_state(self, rotate_axes):
        '''
            rotate_axes: dict/list {'x', 'y', 'z'}
        '''
        self.update_select_list(rotate_axes)
        
        valid_mask = []
        access_mask = []
        box_states = []
        prec_states = []

        # NOTE for greedy
        node_counts = []

        for rot in range(2):
            for axis in rotate_axes:
                # for box_i, box in enumerate(self.box_list):
                    # if box_i not in self.select_list: continue

                for box_i in self.select_list:
                    box = self.box_list[box_i]
                    if box_i == -1:
                        box = [-1, -1, -1]

                    count = 0
                    if self.precedence_graph is not None:
                    # if self.fact_type in { 'tap_fake', 'tap_sim' }:
                    # if True:
                        node = self.precedence_graph.nodes[box_i]
                        code = node.encode( axis, self.precedence_graph.node_num, select_idx=self.select_list )
                        prec_states.append(code)
                        count = node.count_blocks_to()

                    
                    if axis == 'x':
                        tmp_box = [ box[1], box[2], box[0] ]
                    elif axis == 'y':
                        tmp_box = [ box[0], box[2], box[1] ]
                    elif axis == 'z':
                        tmp_box = box

                    if rot == 0:
                        box_states.append( [ tmp_box[0], tmp_box[1], tmp_box[2] ] )
                    elif rot == 1:
                        box_states.append( [ tmp_box[1], tmp_box[0], tmp_box[2] ] )

                    if box_i == -1 or box_i in self.remove_list:
                        valid_mask.append(False)
                        node_counts.append(-1)
                    else:
                        valid_mask.append(True)
                        node_counts.append(count)
                    
                    accessible = True
                    if box_i == -1:
                        accessible = False
                    else:
                        if len(prec_states) > 0:
                            accessible = self.precedence_graph.nodes[box_i].can_access(axis)
                    access_mask.append(accessible)
                    if accessible == False:
                        node_counts[-1] = -1

        # node_counts = np.array(node_counts)
        # max_count = np.max(node_counts)
        # valid_mask = np.array(valid_mask)
        # valid_mask[ node_counts != max_count ] = False

        if len(prec_states) == 0:
            # Keep precedence shape stable even when no precedence graph exists
            # so observation-space checks and encoder reshapes stay consistent.
            prec_states = np.zeros((len(valid_mask), len(self.select_list), 2), dtype=int)

        box_states = np.array(box_states, dtype=np.float32)
        valid_mask = np.array(valid_mask, dtype=bool)
        access_mask = np.array(access_mask, dtype=bool)

        prec_states = np.array(prec_states, dtype=int)
        if len(prec_states) > 1:
            prec_states = prec_states.reshape(-1, len(self.select_list), 2)

        return box_states, prec_states, valid_mask, access_mask

    def save_order(self, index):
        assert self.data_folder is not None, "data_folder must exist"
        os.makedirs(self.data_folder, exist_ok=True)

        box_path = os.path.join(self.data_folder, f"{index}_box.npy")
        prec_path = os.path.join(self.data_folder, f"{index}_pre.npy")

        np.save(box_path, self.box_list)
        if self.precedence_graph is not None:
            np.save(prec_path, self.precedence_graph)

        