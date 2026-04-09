#coding=utf-8
from os import fdopen
import numpy as np
from scipy.spatial import qhull
from matplotlib import pyplot as plt
from ur5_pybullet_env.utils import utils
import pyvista
import pymesh
import copy
from ur5_pybullet_env.utils import pybullet_utils
from scipy.spatial.transform import Rotation as R
import transform as tf
import itertools

pbox_origin = pymesh.load_mesh("./datasets/robotScene/assets/cube.obj")
# import config
x_ = np.linspace(-1., 1., 20)
y_ = np.linspace(-1., 1., 20)
z_ = np.linspace(-1., 1., 20)

x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')

box_inside_points = np.concatenate([[x,y,z]]).reshape(3, -1).transpose(1,0)

def get_box(block_size, pose=None):
    """
    Args:
        block_sizes: Dict{'x', 'y', 'z'}
        pose: Array [4, 4] / [3, 4]
    Returns:
        pbox: pymesh.Mesh
    """
    if pose is None:
        pose = np.identity(4)

    block_size = np.array( [ block_size['x'], block_size['y'], block_size['z'] ] )
    x, y, z = block_size / 2.0

    
    v = pbox_origin.vertices.copy()
    f = pbox_origin.faces
    v[:,0] = x * v[:,0]
    v[:,1] = y * v[:,1]
    v[:,2] = z * v[:,2]

    rot = pose[:3,:3]
    trans = pose[:3,3]
    v = np.einsum( 'ij,aj->ai', rot, v ) + trans

    f_cell = np.ones((len(f), 4)).astype('int') * 3
    f_cell[:, 1:] = f
    mesh = pyvista.PolyData(v, f_cell)

    pbox = pymesh.form_mesh(v, f)
    return pbox, mesh

def is_point_inside( points, cube_size ):
    abs_points = abs(points)
    sub = cube_size/2.0 - abs_points
    return (( sub > 0).sum(axis=1) == 3).sum() > 0
    
    for point in points:
        if (cube_size/2.0 > abs(point)).all():
            return True
    return False

def same_block_size( a_size, b_size, unit_height ):
    '''
    
    Args:
        a_size: np.array / list [3]
        b_size: np.array / list [3]
    
    Returns:
        ret: bool
    '''
    a = np.round(np.sort(a_size) / unit_height)
    b = np.round(np.sort(b_size) / unit_height)
    # a = np.round(np.sort(a_size) / unit_height / unit_length) * unit_length
    # b = np.round(np.sort(b_size) / unit_height / unit_length) * unit_length

    # 2^k = 
    for i in range(3):
        if abs(a[i] - b[i]) > 1e-3:
            return False
    return True 
    # return (a == b).all()


def get_grasp_poses(block_position, block_rotation, block_size):

    init_x_vec = [1, 0, 0]
    init_y_vec = [0, 1, 0]
    init_z_vec = [0, 0, 1]
    rot_x_vec = tf.transform_by_matrix([init_x_vec], block_rotation, True)[0]
    rot_y_vec = tf.transform_by_matrix([init_y_vec], block_rotation, True)[0]
    rot_z_vec = tf.transform_by_matrix([init_z_vec], block_rotation, True)[0]

    z_point_pos = block_position + rot_z_vec * block_size['z'] / 2
    z_point_neg = block_position - rot_z_vec * block_size['z'] / 2
    z_rot_pos = tf.vec_euler_to_matrix( rot_x_vec, np.deg2rad(180) ) @ block_rotation
    z_rot_neg = block_rotation

    y_point_pos = block_position + rot_y_vec * block_size['y'] / 2
    y_point_neg = block_position - rot_y_vec * block_size['y'] / 2
    y_rot_pos = tf.vec_euler_to_matrix( rot_x_vec, np.deg2rad(-90) ) @ block_rotation
    y_rot_neg = tf.vec_euler_to_matrix( rot_x_vec, np.deg2rad(90) ) @ block_rotation

    x_point_pos = block_position + rot_x_vec * block_size['x'] / 2
    x_point_neg = block_position - rot_x_vec * block_size['x'] / 2
    x_rot_pos = tf.vec_euler_to_matrix( rot_y_vec, np.deg2rad(90) ) @ block_rotation
    x_rot_neg = tf.vec_euler_to_matrix( rot_y_vec, np.deg2rad(-90) ) @ block_rotation

    z_pos = tf.combine_to_matrix( z_point_pos, z_rot_pos )[:3]
    z_neg = tf.combine_to_matrix( z_point_neg, z_rot_neg )[:3]
    y_pos = tf.combine_to_matrix( y_point_pos, y_rot_pos )[:3]
    y_neg = tf.combine_to_matrix( y_point_neg, y_rot_neg )[:3]
    x_pos = tf.combine_to_matrix( x_point_pos, x_rot_pos )[:3]
    x_neg = tf.combine_to_matrix( x_point_neg, x_rot_neg )[:3]

    return z_pos, z_neg, y_pos, y_neg, x_pos, x_neg

def get_grasp_direct(block_id:int, rotation_order:list, precedences_list:list):
    '''
    Args:
        block_id: int
        rot_type: list
            int(block_id / blocks_num)
        precedences_list: list
            np.array(n x n)
    Returns:
        direct: str
    '''
    # precedences_list检查是否碰撞
    if rotation_order[-1] == 2:
        direct = 'up'
        if precedences_list[0].sum(axis=0)[block_id] > 0:
            direct = 'down'
    elif rotation_order[-1] == 1:
        direct = 'left'
        if precedences_list[2].sum(axis=0)[block_id] > 0:
            direct = 'right'
    elif rotation_order[-1] == 0:
        direct = 'forward'
        if precedences_list[4].sum(axis=0)[block_id] > 0:
            direct = 'backward'
    return direct

class Block(object):
    def __init__( self, length, width, height, pose, is_main=True, basic_unit = 0.02):

        self.length = length
        self.width = width
        self.height = height

        self.size = {}
        self.size['x'] = length
        self.size['y'] = width
        self.size['z'] = height
        
        self.pose = pose
        self.center = pose[:3,3]

        # generate side block, but side block would not generate
        self.mesh, self.poly_mesh = get_box( self.size, self.pose )

        self.corners = self.mesh.vertices

        self.basic_unit = basic_unit

        if is_main:
            # direction vector
            self._calc_direct_vector()
            # side block
            self._generate_side_block( self.basic_unit * 2 )

            z_pos, z_neg, y_pos, y_neg, x_pos, x_neg = get_grasp_poses(self.pose[:3,3], self.pose[:3,:3], self.size)
            # TODO
            self.grasp_poses = {}
            self.grasp_poses['up'] = z_pos
            self.grasp_poses['down'] = z_neg
            self.grasp_poses['left'] = y_pos
            self.grasp_poses['right'] = y_neg
            self.grasp_poses['forward'] = x_pos
            self.grasp_poses['backward'] = x_neg
    
    def _calc_direct_vector(self):
        self.vectors = {}
        self.vectors['x'] = self.pose[:3,0]
        self.vectors['y'] = self.pose[:3,1]
        self.vectors['z'] = self.pose[:3,2]

    def _generate_side_block(self, gripper_height):
        """
        Add self.side_blocks

        Args:
            gripper_height: float
        """
        signs = [1, -1]
        axes = ['z', 'y', 'x']
        
        blocks_list = []

        for axis in axes:
            for sign in signs:
                vector = self.vectors[axis] * sign

                center = self.center + vector * (gripper_height + self.size[axis]) / 2.0
                # calculate corner of block
                size = copy.deepcopy(self.size)
                size[axis] = gripper_height
                
                pose = self.pose.copy()
                pose[:3,3] = center

                scale_size = self.basic_unit * 0.1
                block = Block( size['x'] - scale_size, size['y'] - scale_size, size['z'] - scale_size, pose, False, self.basic_unit )

                blocks_list.append(block)
        
        blocks_dict = {}
        blocks_dict['up'] = blocks_list[0]
        blocks_dict['down'] = blocks_list[1]

        blocks_dict['left'] = blocks_list[2]
        blocks_dict['right'] = blocks_list[3]
        
        blocks_dict['forward'] = blocks_list[4]
        blocks_dict['backward'] = blocks_list[5]
        self.side_blocks = blocks_dict
    
    def _is_interset(self, a_mesh, b_mesh, a_poly, b_poly, a_data, b_data):
        """
        Args:
            a_mesh: pymesh.Mesh
            b_mesh: pymesh.Mesh
        Returns:
            ret: bool
        """
        def plot(points, color=[1,0,0]):
            for edges in points:
                pybullet_utils.p.addUserDebugLine( edges[0], edges[-1], color, 10, lifeTime=60  )        

        a_size, a_pose = a_data
        b_size, b_pose = b_data
        a_size = np.array([ a_size['x'], a_size['y'], a_size['z'] ])
        b_size = np.array([ b_size['x'], b_size['y'], b_size['z'] ])

        a_edge_points = box_inside_points * a_size / 2
        
        rot = a_pose[:3,:3]
        trans = a_pose[:3,3]
        a_edge_points = np.einsum( 'ij,aj->ai', rot, a_edge_points ) + trans

        a_points = a_edge_points.reshape(-1, 8, 3)
        # for edges in a_points:
        #     pybullet_utils.p.addUserDebugLine( edges[0], edges[-1], [1,0,0], 10, lifeTime=60  )

        inv_b_pose = np.linalg.inv(b_pose)
        rot = inv_b_pose[:3,:3]
        trans = inv_b_pose[:3,3]
        a_edge_points_at_b = np.einsum( 'ij,aj->ai', rot, a_edge_points ) + trans

        points_ab = a_edge_points_at_b.reshape(-1, 8, 3)
        
        is_inside = is_point_inside( a_edge_points_at_b, b_size )
        return is_inside

        rot = b_pose[:3,:3]
        trans = b_pose[:3,3]
        b_edge_points = box_inside_points * b_size / 2
        b_edge_points = np.einsum( 'ij,aj->ai', rot, b_edge_points ) + trans

        b_points = b_edge_points.reshape(-1, 8, 3)
        
        b_edge_points_at_b = box_inside_points * b_size / 2
        points_bb = b_edge_points_at_b.reshape(-1, 8, 3)

        # for edges in b_points:
        #     pybullet_utils.p.addUserDebugLine( edges[0], edges[-1], [1,0,0], 10, lifeTime=60  )

        # poly_inter = a_poly.boolean_intersection(b_poly)
        # return len(poly_inter.points) > 0

        intersect = pymesh.boolean(a_mesh, b_mesh, "intersection")
        
        a_volume = abs(a_mesh.volume)
        b_volume = abs(b_mesh.volume)
        min_volume = min(a_volume, b_volume)
        interset_rate = intersect.volume / min_volume
        # return intersect.volume > 0
        return interset_rate > 1e-3

    def is_upon(self, block):
        """
        Args:
            block: Block
        Returns:
            ret: bool
        """
        a_center = self.center
        b_center = block.center

        if b_center[2] > a_center[2]:
            return False

        # move the a down to the same z as self, check intersection
        a_pose = self.pose.copy()
        a_pose[2,3] = b_center[2]

        down_a_mesh, poly_a_mesh = get_box( self.size, a_pose )
        b_mesh = block.mesh
        return self._is_interset( down_a_mesh, b_mesh, poly_a_mesh, block.poly_mesh, [ self.size, a_pose ], [ block.size, block.pose ] )

    def is_blocked_by(self, block, checked_side):
        """"
        Return True if the 'checked_side' of self.block is blocked by others
        
        Args:
            block: Block
            checked_side: string
                "up", "down", "left", "right", "forward", "backward"
        Returns:
            ret: bool
        
        """
        side_block = self.side_blocks[checked_side]
        return self._is_interset( side_block.mesh, block.mesh, side_block.poly_mesh, block.poly_mesh, [ side_block.size, side_block.pose ], [ block.size, block.pose ] )
    
    def __str__(self):
        return "size: [%.3f, %.3f, %.3f]\n    center: %s\n" % ( 
            self.length, self.width, self.height,
            self.center )


def save_blocks(blocks_list, save_dir):
    import os

    for s in ["up", "down", "left", "right", "forward", "backward"]:
        poses = []
        sizes = []
        centers = []
        for block in blocks_list:
            poses.append(block.side_blocks[s].pose)

            block_size = block.side_blocks[s].size
            sizes.append([block_size['x'], block_size['y'], block_size['z']])
            centers.append(block.side_blocks[s].center)
        poses = np.array(poses).reshape(len(blocks_list), -1)
        sizes = np.array(sizes).reshape(len(blocks_list), -1)
        centers = np.array(centers).reshape(len(blocks_list), -1)
        np.savetxt( os.path.join(save_dir, "%s_pose.txt" % s), poses)
        np.savetxt( os.path.join(save_dir, "%s_size.txt" % s), sizes)
        np.savetxt( os.path.join(save_dir, "%s_position.txt" % s), centers)
    


def calc_move_precedence(blocks_list, blocks_num):
    """
    Compute the top/move precedence of the blockes
    
    Args:
        blocks_list: List[Block]
    Returns:
        precedence: np.array[n, n],
            precedence[a,b]=1 means b is blocked by a or a blocks b
    """
    n = len(blocks_list)
    # precedence = np.ones((blocks_num,blocks_num))
    precedence = np.eye(blocks_num)
    precedence[:, :n] = 0
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if blocks_list[i].is_upon( blocks_list[j] ):
                precedence[i][j] = 1
    return precedence

def calc_side_precedences(blocks_list, precedence_sides, blocks_num):
    """
    Compute the forward and backward precedence of the blockes

    Args:
        blocks_list: List[Block]
        precedence_sides: List[string], 'up', 'down', 'left', 'right', 'forward', 'backward'
            exmaple: precedence_sides = [ 'up', 'dowm' ]
    Returns:
        precedence_0: np.array [n x n]
            precedence_0[a,b]=1 means the 'precedence_sides[0] size' of b is blocked by a or a blocks b
        precedence_1: np.array [n x n]
            precedence_1[a,b]=1 means the 'precedence_sides[1] size' of b is blocked by a or a blocks b
        ...
        precedence_x: np.array [n x n]
            precedence_x[a,b]=1 means the 'precedence_sides[x] size' of b is blocked by a or a blocks b
    """
    n = len(blocks_list)
    pre_len = len(precedence_sides)

    pre_list = []
    for i in range(pre_len):
        # pre = np.ones((blocks_num,blocks_num))
        pre = np.eye(blocks_num)
        pre[:, :n] = 0
        pre_list.append(pre)
        
    for p, pre_side in enumerate(precedence_sides):
        for i in range(n):
            for j in range(n):
                if i == j: continue
                pre_list[p][j][i] = blocks_list[i].is_blocked_by( blocks_list[j], pre_side )

    return pre_list

def to_size(block_size_dict):
    return np.array( [ block_size_dict['x'], block_size_dict['y'], block_size_dict['z'] ] )

# TODO
def match_previous_blocks(blocks_mask, pre_blocks_mask, blocks_list, pre_blocks_list):
    
    new_dict = {}
    match_c2p_dict = {}
    match_p2c_dict = {}

    rot_dict = {}
    rate_dict = {}
    prec_dict = {}

    remove_ids = []
    
    if pre_blocks_mask is None:
        return match_c2p_dict, new_dict, rot_dict, rate_dict, prec_dict, remove_ids

    block_ids = list(np.unique(blocks_mask)[1:])
    pre_block_ids = list(np.unique(pre_blocks_mask)[1:]) # 10 obj max
    
    max_blocks_num = config.TAP.blocks_num

    if len(pre_block_ids) > max_blocks_num:
        for i in pre_block_ids[max_blocks_num:]:
            pre_blocks_mask[ pre_blocks_mask == i ] = 0
        pre_block_ids = pre_block_ids[:max_blocks_num]

    all_orders = [
        [0,1,2],
        [0,1,2],
        [0,1,2],
        [0,1,2],

        [1,2,0],
        [1,2,0],
        [1,2,0],
        [1,2,0],

        [2,0,1],
        [2,0,1],
        [2,0,1],
        [2,0,1],

        [0,2,1],
        [0,2,1],
        [0,2,1],

        [2,1,0],
        [2,1,0],
        [2,1,0],

        [1,0,2],
        [1,0,2],
        [1,0,2],
    ]
    all_rates = [
        [1.0,1.0,1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0,  1.0],
        
        [1.0,1.0,1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0,  1.0],

        [1.0,1.0,1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0,  1.0],
        
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0,  -1.0],

        [-1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0,  -1.0],

        [-1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0,  -1.0],

        [-1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0,  -1.0],

    ]
    # precedence_sides = [ ['up', 'down'], ['left', 'right'], ['forward', 'backward'] ]    

    for b_i, b in enumerate(block_ids):
        b_mask = (blocks_mask == b)
        pre_b_in_mask = pre_blocks_mask[ b_mask ]

        current_area_size = np.sum(b_mask)

        if len(pre_b_in_mask) > 0:

            pre_ids = np.unique(pre_b_in_mask)
            max_i = None
            pre_area_size = 0
            
            for pre_i in pre_ids:
                pre_b_count = np.sum(pre_b_in_mask == pre_i)
                if pre_area_size < pre_b_count:
                    pre_area_size = pre_b_count
                    max_i = pre_i

            if max_i != 0:
                pre_area_size = np.sum(pre_blocks_mask == max_i)

            if max_i != 0 and pre_area_size > current_area_size * 0.9 and current_area_size > pre_area_size * 0.9:                
                pre_b_i = pre_block_ids.index(max_i) 
                
                # 检查方向是否变化
                if pre_b_i >= len(pre_blocks_list):
                    pre_block = pre_blocks_list[0]
                    pre_b_i = 0
                else:
                    pre_block = pre_blocks_list[pre_b_i]
                block = blocks_list[b_i]
                
                if same_block_size( to_size(block.size) , to_size(pre_block.size), config.TAP.unit_height, tolerant_num=tolerant_num):

                    match_c2p_dict[ b_i ] = pre_b_i
                    match_p2c_dict[ pre_b_i ] = b_i 


                    pre_pose = pre_block.pose
                    current_pose = block.pose

                    pre_xyz = tf.matrix_to_xyz(pre_pose)
                    cur_xyz = tf.matrix_to_xyz(current_pose)

                    rot_order = all_orders[0]
                    rot_rate = all_rates[1]
                    for o_i in range(len(all_orders)):
                        o = all_orders[o_i]
                        r = all_rates[o_i]
                        
                        same_num = 0
                        for axis in range(3):
                            if tf.vec_cos_theta( cur_xyz[axis], r[axis] * pre_xyz[ o[axis] ] ) > 0.5:
                                same_num += 1
                        if same_num == 3:
                            rot_order = o
                            rot_rate = r
                            break
                    
                    rot_dict[ b_i ] = rot_order
                    rate_dict[ b_i ] = rot_rate
                    base_precedence_sides = [ [4,5], [2, 3], [0,1] ] # 这里其实是对应 x y z

                    final_prec_list = []
                    for o_i, rot_o in enumerate(rot_order[::-1]): # 因为side我之前是按照 z y x的顺序，这里得逆
                        prec = base_precedence_sides[rot_o]
                        if rot_rate[2-o_i] == -1:
                            prec[0], prec[1] = prec[1], prec[0]
                        final_prec_list += prec
                    
                    prec_dict[b_i] = list(final_prec_list)

            elif max_i != 0:
                # new_dict[ b_i ] = pre_block_ids.index(max_i)
                pre_b_i = pre_block_ids.index(max_i)

                if pre_b_i not in new_dict:
                    new_dict[pre_b_i] = []

                new_dict[pre_b_i].append(b_i)

        
    for pre_b_i, pre_b in enumerate(pre_block_ids):
        if pre_b_i in match_p2c_dict:
            continue
        remove_ids.append(pre_b_i)

    return match_c2p_dict, new_dict, rot_dict, rate_dict, prec_dict, remove_ids


def update_precedence(scene, blocks_list, blocks_mask, pre_blocks_mask, pre_blocks_list, pre_precedences_list, precedence_sides, precedences_list, unit_height, cam_pos=None, check_arm=True):
    """
    Update the precedence of the blockes with robot

    Args:
        blocks_list: List[Block]
        precedence_sides: List[string], 'up', 'down', 'left', 'right', 'forward', 'backward'
            exmaple: precedence_sides = [ 'up', 'dowm' ]
        precedences_list: List[np.array( blocks_num * blocks_num )]
            precedence of the blockes
        
    Returns:
        precedences_list: List[np.array( blocks_num * blocks_num )]
            precedence of the blockes
        
    """

    # 检查变化
    match_dict, new_dict, rot_dict, rate_dict, prec_dict, remove_ids = match_previous_blocks(blocks_mask, pre_blocks_mask, blocks_list, pre_blocks_list)
    blocks_num = len(blocks_list)

    # if blocks_num <= 3:
    #     match_dict = {}

    if pre_blocks_mask is not None:
        pre_blocks_num = len(pre_blocks_list)

    scene.collision_off_obstacles(scene.robot.ee.body, 0)
    scene.collision_off_obstacles(scene.robot.ee.body, -1)

    for p_i, pre_side in enumerate(precedence_sides):
        # check each side of each block
        print(pre_side)
        # if pre_side == 'up': continue
        for b_i, b in enumerate(blocks_list):
            # if precedences_list[p_i][b_i, b_i] == 1: continue
            if precedences_list[p_i].sum(axis=0)[b_i] > 0: continue
            if b.side_blocks[pre_side].pose[2,3] < 0: # 将朝下低于地面的方向设置为不可抓取
                precedences_list[p_i][b_i, b_i] = 1
            
            else:
                # 让机器人检查每个方向可否抓取
                if check_arm:

                    grasp_pose = b.grasp_poses[pre_side]
                    rot = np.identity(4)
                    # 这里旋转是因为识别方向和抓取执行方向是相反的
                    rot[:3,:3] = R.from_euler( 'XYZ', [ np.pi, 0, 0 ] ).as_matrix()
                    grasp_pose = grasp_pose @ rot

                    # 判断和相机视角的朝向过滤
                    side_dir = grasp_pose[:3, 2]
                    side_center = grasp_pose[:3, 3]
                    cam_to_side = side_center - cam_pos

                    if tf.vec_cos_theta( cam_to_side, side_dir ) > 0:
                        precedences_list[p_i][b_i, b_i] = 1

                    else:
                        if b_i in match_dict:
                            pre_b_i = match_dict[b_i]
                            pre_p_i = prec_dict[b_i][p_i]

                            if pre_precedences_list[ pre_p_i ][:, pre_b_i][:pre_blocks_num].sum() > 0:
                                # 看更新的物体是不是阻挡了这个东西
                                blocked_ids = np.where(pre_precedences_list[ pre_p_i ][:, pre_b_i][:pre_blocks_num] > 0)[0]
                                has_remove = False
                                for i in blocked_ids:
                                    if i in remove_ids:
                                        has_remove = True
                                        # can_reach, collision_with = scene.check_reachable( grasp_pose, allowed_time=0, with_ee_body=False )
                                        can_reach, collision_with = scene.check_reachable( grasp_pose, allowed_time=0.05, with_ee_body=False )
                                        if not can_reach:
                                            precedences_list[p_i][b_i, b_i] = 1
                                        break
                                
                                if not has_remove:
                                    precedences_list[p_i][b_i, b_i] = pre_precedences_list[ pre_p_i ][pre_b_i][pre_b_i]

                            # if pre_precedences_list[ pre_p_i ][pre_b_i][pre_b_i] == 0 and pre_precedences_list[ pre_p_i ][:, pre_b_i][:blocks_num].sum() > 0:
                            #     # 看更新的物体是不是阻挡了这个东西
                            #     blocked_ids = np.where(pre_precedences_list[ pre_p_i ][:, pre_b_i][:blocks_num] > 0)[0]
                            #     for i in blocked_ids:
                            #         if i in remove_ids:
                            #             if not scene.check_reachable( grasp_pose, allowed_time=0, with_ee_body=False ):
                            #                 precedences_list[p_i][b_i, b_i] = 1
                            #             break
                            # else:
                            #     precedences_list[p_i][b_i, b_i] = pre_precedences_list[ pre_p_i ][pre_b_i][pre_b_i]


                        else:
                            can_reach, collision_with = scene.check_reachable( grasp_pose, allowed_time=0.05, with_ee_body=False )
                            if not can_reach:
                                print('b_' + str(b_i) + ' ' + pre_side)
                                precedences_list[p_i][b_i, b_i] = 1


    # scene.collision_on_obstacles(scene.robot.ee.body)
    scene.collision_on_obstacles(scene.robot.ee.body, 0)
    scene.collision_on_obstacles(scene.robot.ee.body, -1)

    return precedences_list

def convert_block_array(blocks_list, unit_height, unit_length):
    """MC
    Args:
        blocks_list: List[Block]
        unit_height: float
    Returns:
        ret: np.array [n x 3]
    """
    blocks_arr = [
        ( 
            np.round(b.length / unit_height / unit_length) * unit_length, 
            np.round(b.width / unit_height / unit_length) * unit_length, 
            np.round(b.height / unit_height / unit_length) * unit_length 
        ) for b in blocks_list
    ]
    return np.array(blocks_arr)

def draw_blocks_polygon( blocks_list, names, image_name=None ):
    plt.cla()
    plt.axis('equal')
    plt.grid()
    polygons = []
    for i, block in enumerate(blocks_list):
        points = np.array(block.polygon.boundary.coords[:])
        poly, = plt.plot(points[:,0], points[:,1], '-o', label=names[i] )
        polygons.append( poly )

    plt.legend( handles=polygons)
    # plt.show()
    if image_name is None:
        plt.savefig('./poly.png')
    else:
        plt.savefig(image_name)

def draw_side_blocks_polygon( block, block_name, image_name ):
    
    sides = [ block.side_blocks[s] for s in block.side_blocks ]
    names = [ block_name + '-' + s for s in block.side_blocks ]
    
    sides.append(block)
    names.append(block_name)

    draw_blocks_polygon(sides, names, image_name)