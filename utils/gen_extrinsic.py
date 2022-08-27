import torch
import numpy as np

def trans_x(x:float) -> torch.Tensor:
    return torch.Tensor([[1., 0., 0., x], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

def trans_y(y:float) -> torch.Tensor:
    return torch.Tensor([[1., 0., 0., 0.], [0., 1., 0., y], [0., 0., 1., 0.], [0., 0., 0., 1.]])

def trans_z(z:float) -> torch.Tensor:
    return torch.Tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., z], [0., 0., 0., 1.]])

def rot_x(alpha:float) -> torch.Tensor:
    return torch.Tensor([[1., 0., 0., 0.], [0., np.cos(alpha), np.sin(alpha), 0.], [0., -np.sin(alpha), np.cos(alpha), 0.], [0., 0., 0., 1.]])

def rot_y(beta:float) -> torch.Tensor:
    return torch.Tensor([[np.cos(beta), 0., -np.sin(beta), 0.], [0., 1., 0., 0.], [np.sin(beta), 0., np.cos(beta), 0.], [0., 0., 0., 1.]])

def rot_z(gamma:float) -> torch.Tensor:
    return torch.Tensor([[np.cos(gamma), np.sin(gamma), 0., 0.], [-np.sin(gamma), np.cos(gamma), 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

def flip_x() -> torch.Tensor:
    return torch.Tensor([[-1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

def flip_y() -> torch.Tensor:
    return torch.Tensor([[1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

def flip_z() -> torch.Tensor:
    return torch.Tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., -1., 0.], [0., 0., 0., 1.]])


def spherical_pose(radius:float, beta:float, gamma:float, camera_coordinate_type:str, pose_type:str) -> torch.Tensor:
    '''
    from world:
       z
       |
       |
       |___y
      /
     /
    /
    x
    radius as meter moving towards positive axis, angle as radian looking towards positive axis with counter-clockwise
    '''
    pose = rot_z(gamma) @ rot_y(beta) @ trans_x(radius)
    if camera_coordinate_type == 'right_up_out':
        pose = pose @ rot_z(-np.pi / 2) @ rot_x(-np.pi / 2)
    elif camera_coordinate_type == 'right_up_in':
        pose = pose @ rot_z(-np.pi / 2) @ rot_x(-np.pi / 2) @ flip_z()
    elif camera_coordinate_type == 'right_down_out':
        pose = pose @ rot_z(-np.pi / 2) @ rot_x(np.pi / 2) @ flip_z()
    elif camera_coordinate_type == 'right_down_in':
        pose = pose @ rot_z(-np.pi / 2) @ rot_x(np.pi / 2)
    elif camera_coordinate_type == 'left_up_out':
        pose = pose @ rot_z(np.pi / 2) @ rot_x(-np.pi / 2) @ flip_z()
    elif camera_coordinate_type == 'left_up_in':
        pose = pose @ rot_z(np.pi / 2) @ rot_x(-np.pi / 2)
    elif camera_coordinate_type == 'left_down_out':
        pose = pose @ rot_z(np.pi / 2) @ rot_x(np.pi / 2)
    elif camera_coordinate_type == 'left_down_in':
        pose = pose @ rot_z(np.pi / 2) @ rot_x(np.pi / 2) @ flip_z()
    else:
        raise NotImplementedError
    if pose_type == 'c2w':
        pass
    elif pose_type == 'w2c':
        pose = pose.inverse()
    else:
        raise NotImplementedError
    return pose

def translation_pose(z:float, x:float, y:float, camera_coordinate_type:str, pose_type:str) -> torch.Tensor:
    '''
    from world:
       z
       |
       |
       |___y
      /
     /
    /
    x
    x, y, z as meter moving towards positive axis
    '''
    pose = trans_z(z) @ trans_x(x) @ trans_y(y)
    if camera_coordinate_type == 'right_up_out':
        pose = pose
    elif camera_coordinate_type == 'right_up_in':
        pose = pose @ flip_z()
    elif camera_coordinate_type == 'right_down_out':
        pose = pose @ flip_y()
    elif camera_coordinate_type == 'right_down_in':
        pose = pose @ flip_y() @ flip_z()
    elif camera_coordinate_type == 'left_up_out':
        pose = pose @ flip_x() @ flip_y()
    elif camera_coordinate_type == 'left_up_in':
        pose = pose @ flip_x() @ flip_y() @ flip_z()
    elif camera_coordinate_type == 'left_down_out':
        pose = pose @ flip_x()
    elif camera_coordinate_type == 'left_down_in':
        pose = pose @ flip_x() @ flip_z()
    else:
        raise NotImplementedError
    if pose_type == 'c2w':
        pass
    elif pose_type == 'w2c':
        pose = pose.inverse()
    else:
        raise NotImplementedError
    return pose

def spherical_pose_colmap(radius:float, beta:float, gamma:float, camera_coordinate_type:str, pose_type:str) -> torch.Tensor:
    '''
    from world:
       z
       |
       |
       |___y
      /
     /
    /
    x
    radius as meter moving towards positive axis, angle as radian looking towards positive axis with counter-clockwise
    COLMAP's coordinate is not z up in world coordinate, this function is actually a customized version
    '''
    pose = rot_x(-np.pi / 2) @ trans_z(-1) @ rot_z(gamma) @ rot_y(beta) @ trans_x(radius)
    if camera_coordinate_type == 'right_up_out':
        pose = pose @ rot_z(-np.pi / 2) @ rot_x(-np.pi / 2)
    elif camera_coordinate_type == 'right_up_in':
        pose = pose @ rot_z(-np.pi / 2) @ rot_x(-np.pi / 2) @ flip_z()
    elif camera_coordinate_type == 'right_down_out':
        pose = pose @ rot_z(-np.pi / 2) @ rot_x(np.pi / 2) @ flip_z()
    elif camera_coordinate_type == 'right_down_in':
        pose = pose @ rot_z(-np.pi / 2) @ rot_x(np.pi / 2)
    elif camera_coordinate_type == 'left_up_out':
        pose = pose @ rot_z(np.pi / 2) @ rot_x(-np.pi / 2) @ flip_z()
    elif camera_coordinate_type == 'left_up_in':
        pose = pose @ rot_z(np.pi / 2) @ rot_x(-np.pi / 2)
    elif camera_coordinate_type == 'left_down_out':
        pose = pose @ rot_z(np.pi / 2) @ rot_x(np.pi / 2)
    elif camera_coordinate_type == 'left_down_in':
        pose = pose @ rot_z(np.pi / 2) @ rot_x(np.pi / 2) @ flip_z()
    else:
        raise NotImplementedError
    if pose_type == 'c2w':
        pass
    elif pose_type == 'w2c':
        pose = pose.inverse()
    else:
        raise NotImplementedError
    return pose


def surround_poses(frame_num:int, radius:float, camera_coordinate_type:str, pose_type:str) -> list:
    poses = []
    for i in range(frame_num):
        gamma = -2 * np.pi * i / frame_num
        poses.append(spherical_pose(radius, 0, gamma, camera_coordinate_type, pose_type))
    return poses

def surround_poses_colmap(frame_num:int, radius:float, camera_coordinate_type:str, pose_type:str) -> list:
    '''
    COLMAP's coordinate is not z up in world coordinate, this function is actually a customized version
    '''
    poses = []
    for i in range(frame_num):
        gamma = -2 * np.pi * i / frame_num
        poses.append(spherical_pose_colmap(radius, 0, gamma, camera_coordinate_type, pose_type))
    return poses


def spiral_poses(frame_num:int, radius:float, camera_coordinate_type:str, pose_type:str) -> list:
    poses = []
    for i in range(frame_num):
        beta = 0.5 * np.pi * i / frame_num
        gamma = -2 * np.pi * i / frame_num
        poses.append(spherical_pose(radius, beta, gamma, camera_coordinate_type, pose_type))
    return poses

def spiral_poses_colmap(frame_num:int, radius:float, camera_coordinate_type:str, pose_type:str) -> list:
    '''
    COLMAP's coordinate is not z up in world coordinate, this function is actually a customized version
    '''
    poses = []
    for i in range(frame_num):
        beta = 0.5 * np.pi * i / frame_num
        gamma = -2 * np.pi * i / frame_num
        poses.append(spherical_pose_colmap(radius, beta, gamma, camera_coordinate_type, pose_type))
    return poses


def circle_poses(frame_num:int, distance:float, radius:float, camera_coordinate_type:str, pose_type:str) -> list:
    poses = []
    for i in range(frame_num):
        theta = 2 * np.pi * i / frame_num
        poses.append(translation_pose(distance, radius * np.cos(theta), radius * np.sin(theta), camera_coordinate_type, pose_type))
    return poses
