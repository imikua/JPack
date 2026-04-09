import os
import numpy as np
import matplotlib as mpl
import pybullet as p

from ur5_pybullet_env.ur5_envs.envs import Env, colors
from ur5_pybullet_env.ur5_envs import cameras
from ur5_pybullet_env.ur5_envs import models

from ur5_pybullet_env.utils import pybullet_utils
from ur5_pybullet_env.utils import utils
from .block import Block, same_block_size
import matplotlib.pyplot as plt

def init_cam():
    width = 640
    height = 480

    ratio = width / height

    # vision sensor > object properties > common
    focal_dist = 2.0

    if (ratio > 1):
        fov_x = np.deg2rad(60.0)
        fov_y = 2 * np.math.atan(np.math.tan(fov_x / 2) / ratio)
    else:
        fov_y = np.deg2rad(60.0)
        fov_x = 2 * np.math.atan(np.math.tan(fov_y / 2) / ratio)

    # fov = np.deg2rad(60.0)

    focal_x = width / (focal_dist * np.math.tan(fov_x / 2))
    focal_y = height / (focal_dist * np.math.tan(fov_y / 2))

    ins = np.identity(3)
    ins[0, 0] = focal_x
    ins[1, 1] = focal_y
    ins[0, 2] = width / 2
    ins[1, 2] = height / 2

    return list(np.reshape(ins, (-1)))

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to a rotation matrix.
    """
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx
    return R


def quaternion_to_rotation_matrix(x, y, z, w):
    """
    Convert a quaternion (x, y, z, w) to a rotation matrix.
    """
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])
    return R


def plot_directions(quaternions):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for quat in quaternions:
        x, y, z, w = quat
        R = quaternion_to_rotation_matrix(x, y, z, w)
        direction = R @ np.array([1, 0, 0])

        ax.quiver(0, 0, 0, direction[0], direction[1], direction[2], length=1.0)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# def plot_directions(euler_angles):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     for angles in euler_angles:
#         roll, pitch, yaw = angles
#         R = euler_to_rotation_matrix(roll, pitch, yaw)
#         direction = R @ np.array([1, 0, 0])
#
#         ax.quiver(0, 0, 0, direction[0], direction[1], direction[2], length=1.0)
#
#     ax.set_xlim([-1, 1])
#     ax.set_ylim([-1, 1])
#     ax.set_zlim([-1, 1])
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     plt.show()

class Pack_Env(Env):
    def __init__(self, assets_root=models.get_data_path(), disp=False, shared_memory=False, hz=240,
                 use_egl=False) -> None:
        super().__init__(assets_root, disp=disp, shared_memory=shared_memory, hz=hz, use_egl=use_egl)

        self.blocks_size = []
        self.sizes = []
        self.packed_positions = []

    def reset(self, show_gui=True):
        super().reset(show_gui=show_gui)
        self.blocks_size = []
        self.sizes = []
        self.packed_positions = []

    def load_scene(self, save_path, scene_id, block_unit, offset=np.array([0, 0, 0])):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        data_path = os.path.join(save_path, str(scene_id))

        poses_mat = np.load(os.path.join(data_path, "poses.npy"), allow_pickle=True)
        sizes = np.load(os.path.join(data_path, "sizes.npy"), allow_pickle=True)

        tex_path = os.path.join(data_path, "textures.npy")
        self.textures = None
        if os.path.exists(tex_path):
            self.textures = np.load(tex_path, allow_pickle=True)

        poses = []
        self.box_ids = []
        self.box_urdf = []

        # for i in range(len(poses_mat)):
        for i in range(1):
            mat = poses_mat[i]
            size = sizes[i]
            pose = pybullet_utils.mat_to_pose(mat)

            pose[0] += offset
            poses.append(pose)

            if len(poses_mat) <= len(colors):
                color = colors[i]
            else:
                color = colors[np.random.randint(len(colors))]

            color = pybullet_utils.color_random(color)
            box_id, obj_urdf = self.add_box(pose, size, color, use_tex=True)

            self.box_ids.append(box_id)
            self.box_urdf.append(obj_urdf)

            if self.textures is not None:
                p.changeVisualShape(box_id, -1, textureUniqueId=self.tex_uids[self.textures[i]])

            if self.recorder is not None:
                self.recorder.register_object(box_id, obj_urdf)

            if i > 1:
                break

        block_sizes = np.array(sizes / block_unit)
        block_sizes = np.ceil(block_sizes)

        self.refresh_boxes()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        pybullet_utils.simulate_step(1000)

        return block_sizes, poses_mat, sizes

    def load_box(self, box_size, unit, generate_origin):
        cube_extents = box_size * unit
        cube_half_extents = np.array(cube_extents) / 2
        targetFLB = generate_origin

        cube_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=cube_half_extents)

        mass = 1
        # if self.visual:
        cube_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=cube_half_extents)
        cube_body = p.createMultiBody(baseMass=mass,
                                      baseCollisionShapeIndex=cube_collision_shape,
                                      baseVisualShapeIndex=cube_visual_shape)
        # else:
        #     cube_body = p.createMultiBody(baseMass=mass,
        #                               baseCollisionShapeIndex=cube_collision_shape)


        p.resetBasePositionAndOrientation(cube_body, targetFLB + cube_half_extents, [0, 0, 0, 1])
        pick_pose = p.getBasePositionAndOrientation(cube_body)
        self.robot.add_object_to_list('rigid', cube_body)
        self.obstacles.append(cube_body)
        p.changeDynamics(cube_body, -1, mass=mass)

        pick_pose = [list(i) for i in pick_pose]
        pick_pose[0][-1] += cube_half_extents[-1]

        return cube_body, pick_pose

    def refresh_boxes(self):
        if self.textures is not None:
            for i, box_id in enumerate(self.box_ids):
                p.changeVisualShape(box_id, -1, textureUniqueId=self.tex_uids[self.textures[i]])

    def add_wall(self, target_origin, target_width, block_gap, block_unit, wall_width, wall_height, only_base=False):
        block_width = target_width * (block_unit + block_gap * 2) + wall_width * 2
        half_width = block_width / 2.0
        half_height = wall_height / 2.0

        tx = target_origin[0]
        ty = target_origin[1]
        target_center = np.array([tx + target_width * block_unit / 2.0, ty + target_width * block_unit / 2.0, 0])
        target_center += [wall_width, wall_width, 0]

        wall_color = (0.79, 0.196, 0.198, 1)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        container_offset = np.array([0, 0, 0.025])
        wall_1 = None
        if not only_base:
            wall_1, wall_1_urdf = self.add_box(
                (target_center + container_offset + [0, half_width, half_height], (0, 0, 0, 1)),
                (block_width + wall_width, wall_width, wall_height), color=wall_color, category='fixed', mass=0)
            wall_2, wall_2_urdf = self.add_box(
                (target_center + container_offset - [0, half_width, -half_height], (0, 0, 0, 1)),
                (block_width + wall_width, wall_width, wall_height), color=wall_color, category='fixed', mass=0)
            wall_3, wall_3_urdf = self.add_box(
                (target_center + container_offset + [half_width, 0, half_height], (0, 0, 0, 1)),
                (wall_width, block_width + wall_width, wall_height), color=wall_color, category='fixed', mass=0)
            wall_4, wall_4_urdf = self.add_box(
                (target_center + container_offset - [half_width, 0, -half_height], (0, 0, 0, 1)),
                (wall_width, block_width + wall_width, wall_height), color=wall_color, category='fixed', mass=0)
        wall_5, wall_5_urdf = self.add_box((target_center + container_offset, (0, 0, 0, 1)),
                                           (block_width, block_width, wall_width), color=(0.9, 0.2, 0.2, 1),
                                           category='fixed', mass=0)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        if self.recorder is not None:
            if wall_1 is not None:
                self.recorder.register_object(wall_1, wall_1_urdf)
                self.recorder.register_object(wall_2, wall_2_urdf)
                self.recorder.register_object(wall_3, wall_3_urdf)
                self.recorder.register_object(wall_4, wall_4_urdf)
            self.recorder.register_object(wall_5, wall_5_urdf)

        if wall_1 is not None:
            return [wall_1, wall_2, wall_3, wall_4, wall_5]
        else:
            return [wall_5]
        # return [ container_id ]

    def pick_block(self, pick_pose):
        pick_pose[1] = [pick_pose[1][3], pick_pose[1][0], pick_pose[1][1], pick_pose[1][2]]
        pick_pose = utils.multiply(pick_pose, ((0, 0, 0), utils.eulerXYZ_to_quatXYZW((np.pi, 0, 0))))

        max_count = 1
        while True:
            max_count -= 1
            pick_success = self.pick(pick_pose)
            if pick_success:
                break
            if max_count <= 0:
                break

        while not self.robot.is_static:
            pybullet_utils.simulate_step(1)
            if self.recorder is not None:
                self.recorder.add_keyframe()
        return pick_success, pick_pose

    def place_at(self, position, rotation, size, offset, block_unit, state='place',
                 rot_right=False, block_gap=0, pos_offset=None, rot_offset=None, time_limit = True, cube_id = None):

        # 加点偏移保证间隔
        position = np.array(position) * (block_unit + block_gap)

        size = np.round(size / block_unit) * (block_unit + block_gap)

        real_position = [
            offset[0] + position[0] + size[0] / 2,
            offset[1] + position[1] + size[1] / 2,
            offset[2] + position[2] + size[2] / 2
        ]
        FLB_position = [
            offset[0] + position[0],
            offset[1] + position[1],
            offset[2] + position[2] - size[2] / 2
        ]
        pose = np.identity(4)
        pose[:3, 3] = real_position

        if pos_offset is not None:
            rot_pos_offset = rotation @ pos_offset

        if rot_offset is not None:
            rotation = rotation @ rot_offset

        rot = np.identity(4)
        rot[:3, :3] = rotation

        pose = pose @ rot
        pose = pose[:3]

        if pos_offset is not None:
            pose[:3, 3] += rot_pos_offset

        if state in ['test', 'move']:
            pose[:3, 3] += [0, 0, 0.1 + size[-1]]

        pose_mat = pose.copy()
        pose = pybullet_utils.mat_to_pose(pose)

        # self.robot.place(pose)
        if state in ['test', 'move']:
            move_success = self.move_ee_to(pose)
        else:
            max_count = 1
            while True:
                max_count -= 1
                height_pose = [list(i) for i in pose]
                height_pose[0][-1] += 0.2
                move_success = self.move_ee_to(height_pose)
                move_success = self.place(pose, [0, 0, 0.05 + size[-1] / 2])
                if move_success:
                    break
                if max_count <= 0:
                    break

        while not self.robot.is_static:
            pybullet_utils.simulate_step(1)
            if self.recorder is not None:
                self.recorder.add_keyframe()

        # pybullet_utils.simulate_step(20)
        for _ in range(20):
            pybullet_utils.simulate_step(1)
            if self.recorder is not None:
                self.recorder.add_keyframe()

        p.resetBasePositionAndOrientation(cube_id, FLB_position + size / 2, [0, 0, 0, 1])
        p.changeDynamics(cube_id, -1, mass = 0.0)
        return move_success, pose_mat

    def remove_current_block(self, block_size, block_unit, height_offset, remove_position):
        attach_obj_ids = self.robot.ee.attach_obj_ids

        offset = remove_position.copy()
        offset[2] += height_offset

        self.place_at([0, 0, 0], np.identity(3), block_size * block_unit, offset, block_unit, 'place')

        remove_success = True

        if attach_obj_ids is not None:
            pybullet_utils.p.resetBasePositionAndOrientation(attach_obj_ids, [-10, -10, 5], [0, 0, 0, 1])
        else:
            remove_success = False

        currj = self.robot.get_current_joints()
        currj[1] -= 0.3
        self.robot.movej(currj)

        return remove_success

    def new_container(self, target_origin, target_width, block_gap, block_unit, wall_width, wall_height,
                      offset=np.array([0.3, 0, 0]), only_base=False):
        for c in self.containers:
            for obj in c:
                pose = p.getBasePositionAndOrientation(obj)
                pose = [list(i) for i in pose]
                p.resetBasePositionAndOrientation(obj, pose[0] + offset, pose[1])
        walls = self.add_wall(target_origin, target_width, block_gap, block_unit, wall_width, wall_height, only_base)

        self.containers.append([])
        self.containers[-1] += walls

