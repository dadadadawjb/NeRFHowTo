import numpy as np
import torch
import imageio
from abc import ABC, abstractmethod
import os
import json
import random
import matplotlib.pyplot as plt

class Dataset(ABC):
    def __init__(self) -> None:
        self.training_indices = []
        self.validation_indices = []
        self.testing_indices = []

    @abstractmethod
    def get_image(self, index:int) -> tuple:
        # (image_coordinate_type, image): tuple(str, torch.Tensor((height, width, 3)D))
        pass

    @abstractmethod
    def get_camera(self, index:int) -> tuple:
        # (camera_coordinate_type, camera_model, pose_type, camera_intrinsic, camera_extrinsic): tuple(str, str, str, dict, torch.Tensor((4, 4)D))
        pass


class HelloWorldDataset(Dataset):
    def __init__(self, images:np.ndarray, poses:np.ndarray, focal:float) -> None:
        super(HelloWorldDataset, self).__init__()
        self._images = torch.from_numpy(images)
        self._poses = torch.from_numpy(poses)
        self._focal = float(focal)

        self.training_indices = list(range(100))                        # 100
        self.validation_indices = list(range(0, 100, 10))               # 10
        self.testing_indices = list(range(100, self._images.shape[0]))  # 6
        self.show_index = 101
    
    def get_image(self, index:int) -> tuple:
        return ('right_down', self._images[index])
    
    def get_camera(self, index:int) -> tuple:
        camera_intrinsic = {
            'fx': self._focal, 
            'fy': self._focal, 
            'cx': self._images.shape[2] / 2, 
            'cy': self._images.shape[1] / 2
        }
        return ('right_up_out', 'pinhole', 'c2w', camera_intrinsic, self._poses[index])


class BlenderDataset(Dataset):
    def __init__(self, metas:dict) -> None:
        super(BlenderDataset, self).__init__()
        self._metas = metas
        self._training_start = 0
        self._training_end = len(metas["train"]["frames"]) - 1
        # [_training_start, _training_end]
        self._validation_start = self._training_end + 1
        self._validation_end = self._validation_start + len(metas["val"]["frames"]) - 1
        # [_validation_start, _validation_end]
        self._testing_start = self._validation_end + 1
        self._testing_end = self._testing_start + len(metas["test"]["frames"]) - 1
        # [_testing_start, _testing_end]

        self.training_indices = list(range(self._training_start, self._training_end + 1))
        self.validation_indices = list(range(self._validation_start, self._validation_end + 1))
        self.testing_indices = list(range(self._testing_start, self._testing_end + 1))
        self.show_index = 2
    
    def get_image(self, index:int) -> tuple:
        if index in self.training_indices:
            image_path = self._metas["train"]["frames"][index - self._training_start]["file_path"]
        elif index in self.validation_indices:
            image_path = self._metas["val"]["frames"][index - self._validation_start]["file_path"]
        elif index in self.testing_indices:
            image_path = self._metas["test"]["frames"][index - self._testing_start]["file_path"]
        else:
            raise ValueError("Index out of range")
        image_origin = np.array(imageio.imread(image_path)).astype(np.float32) / 255.0
        image = np.zeros((image_origin.shape[0], image_origin.shape[1], 3), dtype='float32')
        r, g, b, a = image_origin[:, :, 0], image_origin[:, :, 1], image_origin[:, :, 2], image_origin[:, :, 3]
        image[:, :, 0] = r * a + (1.0 - a) * 1.0
        image[:, :, 1] = g * a + (1.0 - a) * 1.0
        image[:, :, 2] = b * a + (1.0 - a) * 1.0
        image = torch.from_numpy(image)
        return ('right_down', image)
    
    def get_camera(self, index:int) -> tuple:
        if index in self.training_indices:
            image_path = self._metas["train"]["frames"][index - self._training_start]["file_path"]
            camera_extrinsic = self._metas["train"]["frames"][index - self._training_start]["transform_matrix"]
            camera_angle_x = self._metas["train"]["camera_angle_x"]
        elif index in self.validation_indices:
            image_path = self._metas["val"]["frames"][index - self._validation_start]["file_path"]
            camera_extrinsic = self._metas["val"]["frames"][index - self._validation_start]["transform_matrix"]
            camera_angle_x = self._metas["val"]["camera_angle_x"]
        elif index in self.testing_indices:
            image_path = self._metas["test"]["frames"][index - self._testing_start]["file_path"]
            camera_extrinsic = self._metas["test"]["frames"][index - self._testing_start]["transform_matrix"]
            camera_angle_x = self._metas["val"]["camera_angle_x"]
        else:
            raise ValueError("Index out of range")
        image_origin = np.array(imageio.imread(image_path))
        focal = 0.5 * image_origin.shape[1] / np.tan(0.5 * camera_angle_x)
        camera_intrinsic = {
            'fx': focal, 
            'fy': focal, 
            'cx': image_origin.shape[1] / 2, 
            'cy': image_origin.shape[0] / 2
        }
        camera_extrinsic = torch.from_numpy(np.array(camera_extrinsic, dtype='float32'))
        return ('right_up_out', 'pinhole', 'c2w', camera_intrinsic, camera_extrinsic)


class LLFFDataset(Dataset):
    def __init__(self, image_paths:list, poses:np.ndarray, focals:list) -> None:
        super(LLFFDataset, self).__init__()
        self._image_paths = image_paths
        self._poses = torch.from_numpy(poses)
        self._focals = focals

        self.training_indices = list(range(len(image_paths)))
        self.validation_indices = list(range(int(0.6 * len(image_paths)), int(0.8 * len(image_paths))))
        self.testing_indices = list(range(int(0.8 * len(image_paths)), len(image_paths)))
        self.show_index = 0
    
    def get_image(self, index:int) -> tuple:
        image = imageio.imread(self._image_paths[index])
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)
        return ('right_down', image)
    
    def get_camera(self, index:int) -> tuple:
        image = np.array(imageio.imread(self._image_paths[index]))
        camera_intrinsic = {
            'fx': self._focals[index], 
            'fy': self._focals[index], 
            'cx': image.shape[1] / 2, 
            'cy': image.shape[0] / 2
        }
        camera_extrinsic = torch.cat([self._poses[index], torch.Tensor([[0., 0., 0., 1.]])], dim=0)
        return ('right_up_out', 'pinhole', 'c2w', camera_intrinsic, camera_extrinsic)


class COLMAPDataset(Dataset):
    def __init__(self, image_paths:list, poses:list, camera_intrinsic:dict, height:int, width:int) -> None:
        super(COLMAPDataset, self).__init__()
        self._image_paths = image_paths
        self._poses = poses
        self._camera_intrinsic = camera_intrinsic
        self._height = height
        self._width = width

        # here we do not care about the actual performance
        self.training_indices = list(range(len(image_paths)))
        self.validation_indices = random.sample(self.training_indices, 10)
        self.testing_indices = random.sample(self.training_indices, 10)
        self.show_index = 123
    
    def get_image(self, index:int) -> tuple:
        image = np.array(imageio.imread(self._image_paths[index])).astype(np.float32) / 255.0
        image = torch.from_numpy(image)
        return ('right_down', image)
    
    def get_camera(self, index:int) -> tuple:
        camera_extrinsic = torch.from_numpy(self._poses[index])
        return ('right_down_in', 'pinhole', 'w2c', self._camera_intrinsic, camera_extrinsic)


def load_helloworld(dataset_path:str) -> HelloWorldDataset:
    data = np.load(dataset_path)
    images = data["images"]
    poses = data["poses"]
    focal = data["focal"]

    dataset = HelloWorldDataset(images, poses, focal)
    return dataset

def observe_helloworld(dataset_path:str, log_path:str) -> dict:
    data = np.load(dataset_path)
    images = data["images"]
    poses = data["poses"]
    focal = data["focal"]

    info = {}
    info["examples_num"] = images.shape[0]
    info["all_examples_same_size"] = "True"
    info["image_height"] = images.shape[1]
    info["image_width"] = images.shape[2]
    info["all_examples_same_image_coordinate_type"] = "True"
    info["image_coordinate_type"] = "right_down"
    info["all_examples_same_camera_coordinate_type"] = "True"
    info["camera_coordinate_type"] = "right_up_out"
    info["all_examples_same_camera_model"] = "True"
    info["camera_model"] = "pinhole"
    info["all_examples_same_pose_type"] = "True"
    info["pose_type"] = "c2w"
    info["all_examples_same_camera_intrinsic"] = "True"
    info["camera_intrinsic"] = "fx: {}, fy: {}, cx: {}, cy: {}".format(focal, focal, images.shape[2] / 2, images.shape[1] / 2)
    info["camera_extrinsics"] = "see extrinsics.png"

    # use the 'right_up_out' and 'c2w' convention
    dirs = np.stack([np.sum([0, 0, -1] * pose[:3, :3], axis=-1) for pose in poses])
    origins = poses[:, :3, -1]
    ax = plt.figure(figsize=(12, 8)).add_subplot(projection='3d')
    ax.quiver(0, 0, 0, 1, 0, 0, length=1.0, color='r', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 1, 0, length=1.0, color='g', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, 1, length=1.0, color='b', arrow_length_ratio=0.1)
    _ = ax.quiver(origins[..., 0].flatten(), origins[..., 1].flatten(), origins[..., 2].flatten(), 
        dirs[..., 0].flatten(), dirs[..., 1].flatten(), dirs[:, 2].flatten(), length=0.5, normalize=True)
    plt.savefig(os.path.join(log_path, 'extrinsics.png'))

    return info


def load_blender(dataset_path:str) -> BlenderDataset:
    splits = ['train', 'val', 'test']
    metas = {}
    for split in splits:
        with open(os.path.join(dataset_path, f'transforms_{split}.json'), 'r') as f:
            metas[split] = json.load(f)
            for i in range(len(metas[split]["frames"])):
                metas[split]["frames"][i]["file_path"] = os.path.join(dataset_path, metas[split]["frames"][i]["file_path"] + ".png")
    
    dataset = BlenderDataset(metas)
    return dataset

def observe_blender(dataset_path:str, log_path:str) -> dict:
    splits = ['train', 'val', 'test']
    metas = {}
    for split in splits:
        with open(os.path.join(dataset_path, f'transforms_{split}.json'), 'r') as f:
            metas[split] = json.load(f)
    
    poses_list = []
    for split in splits:
        for i in range(len(metas[split]["frames"])):
            pose = torch.from_numpy(np.array(metas[split]["frames"][i]["transform_matrix"]))
            poses_list.append(pose)
    poses = torch.stack(poses_list, dim=0)
    poses = poses.numpy()
    
    info = {}
    info["examples_num"] = len(metas["train"]["frames"]) + len(metas["val"]["frames"]) + len(metas["test"]["frames"])
    info["all_examples_same_size"] = "not necessary"
    info["all_examples_same_image_coordinate_type"] = "True"
    info["image_coordinate_type"] = "right_down"
    info["all_examples_same_camera_coordinate_type"] = "True"
    info["camera_coordinate_type"] = "right_up_out"
    info["all_examples_same_camera_model"] = "True"
    info["camera_model"] = "pinhole"
    info["all_examples_same_pose_type"] = "True"
    info["pose_type"] = "c2w"
    info["all_examples_same_camera_intrinsic"] = "not necessary"
    info["camera_extrinsics"] = "see extrinsics.png"

    # use the 'right_up_out' and 'c2w' convention
    dirs = np.stack([np.sum([0, 0, -1] * pose[:3, :3], axis=-1) for pose in poses])
    origins = poses[:, :3, -1]
    ax = plt.figure(figsize=(12, 8)).add_subplot(projection='3d')
    ax.quiver(0, 0, 0, 1, 0, 0, length=1.0, color='r', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 1, 0, length=1.0, color='g', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, 1, length=1.0, color='b', arrow_length_ratio=0.1)
    _ = ax.quiver(origins[..., 0].flatten(), origins[..., 1].flatten(), origins[..., 2].flatten(), 
        dirs[..., 0].flatten(), dirs[..., 1].flatten(), dirs[:, 2].flatten(), length=0.5, normalize=True)
    plt.savefig(os.path.join(log_path, 'extrinsics.png'))

    return info


def load_llff(dataset_path:str) -> LLFFDataset:
    image_paths = [os.path.join(dataset_path, "images_8", name) for name in sorted(os.listdir(os.path.join(dataset_path, "images_8"))) if name.endswith(".png")]
    
    # something messy
    def _normalize(x):
        return x / np.linalg.norm(x)
    def _viewmatrix(z, up, pos):
        vec2 = _normalize(z)
        vec1_avg = up
        vec0 = _normalize(np.cross(vec1_avg, vec2))
        vec1 = _normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, pos], 1)
        return m
    def _poses_avg(poses):
        hwf = poses[0, :3, -1:]
        center = poses[:, :3, 3].mean(0)
        vec2 = _normalize(poses[:, :3, 2].sum(0))
        up = poses[:, :3, 1].sum(0)
        c2w = np.concatenate([_viewmatrix(vec2, up, center), hwf], 1)
        return c2w
    def _recenter_poses(poses):
        poses_ = poses + 0
        bottom = np.reshape([0., 0., 0., 1.], [1, 4])
        c2w = _poses_avg(poses)
        c2w = np.concatenate([c2w[:3, :4], bottom], -2)
        bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
        poses = np.concatenate([poses[:, :3, :4], bottom], -2)
        poses = np.linalg.inv(c2w) @ poses
        poses_[:, :3, :4] = poses[:, :3, :4]
        poses = poses_
        return poses
    poses_bounds = np.load(os.path.join(dataset_path, "poses_bounds.npy"))
    poses = poses_bounds[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_bounds[:, -2:].transpose([1, 0])
    sh = imageio.imread(image_paths[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1.0 / 8
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    poses[:, :3, 3] *= 1.0 / (0.75 * bds.min())
    bds *= 1.0 / (0.75 * bds.min())
    poses = _recenter_poses(poses)
    focals = poses[:, 2, -1].tolist()
    poses = poses[:, :3, :4]

    dataset = LLFFDataset(image_paths, poses, focals)
    return dataset

def observe_llff(dataset_path:str, log_path:str) -> dict:
    image_paths = [os.path.join(dataset_path, "images_8", name) for name in sorted(os.listdir(os.path.join(dataset_path, "images_8"))) if name.endswith(".png")]

    # something messy
    def _normalize(x):
        return x / np.linalg.norm(x)
    def _viewmatrix(z, up, pos):
        vec2 = _normalize(z)
        vec1_avg = up
        vec0 = _normalize(np.cross(vec1_avg, vec2))
        vec1 = _normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, pos], 1)
        return m
    def _poses_avg(poses):
        hwf = poses[0, :3, -1:]
        center = poses[:, :3, 3].mean(0)
        vec2 = _normalize(poses[:, :3, 2].sum(0))
        up = poses[:, :3, 1].sum(0)
        c2w = np.concatenate([_viewmatrix(vec2, up, center), hwf], 1)
        return c2w
    def _recenter_poses(poses):
        poses_ = poses + 0
        bottom = np.reshape([0., 0., 0., 1.], [1, 4])
        c2w = _poses_avg(poses)
        c2w = np.concatenate([c2w[:3, :4], bottom], -2)
        bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
        poses = np.concatenate([poses[:, :3, :4], bottom], -2)
        poses = np.linalg.inv(c2w) @ poses
        poses_[:, :3, :4] = poses[:, :3, :4]
        poses = poses_
        return poses
    poses_bounds = np.load(os.path.join(dataset_path, "poses_bounds.npy"))
    poses = poses_bounds[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_bounds[:, -2:].transpose([1, 0])
    sh = imageio.imread(image_paths[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1.0 / 8
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    poses[:, :3, 3] *= 1.0 / (0.75 * bds.min())
    bds *= 1.0 / (0.75 * bds.min())
    poses = _recenter_poses(poses)
    # focals = poses[:, 2, -1].tolist()
    poses = poses[:, :3, :4]

    info = {}
    info["examples_num"] = len(image_paths)
    info["all_examples_same_size"] = "not necessary"
    info["all_examples_same_image_coordinate_type"] = "True"
    info["image_coordinate_type"] = "right_down"
    info["all_examples_same_camera_coordinate_type"] = "True"
    info["camera_coordinate_type"] = "right_up_out"
    info["all_examples_same_camera_model"] = "True"
    info["camera_model"] = "pinhole"
    info["all_examples_same_pose_type"] = "True"
    info["pose_type"] = "c2w"
    info["all_examples_same_camera_intrinsic"] = "not necessary"
    info["camera_extrinsics"] = "see extrinsics.png"

    # use the 'right_up_out' and 'c2w' convention
    dirs = np.stack([np.sum([0, 0, -1] * pose[:3, :3], axis=-1) for pose in poses])
    origins = poses[:, :3, -1]
    ax = plt.figure(figsize=(12, 8)).add_subplot(projection='3d')
    ax.quiver(0, 0, 0, 1, 0, 0, length=0.01, color='r', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 1, 0, length=0.01, color='g', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, 1, length=0.01, color='b', arrow_length_ratio=0.1)
    _ = ax.quiver(origins[..., 0].flatten(), origins[..., 1].flatten(), origins[..., 2].flatten(), 
        dirs[..., 0].flatten(), dirs[..., 1].flatten(), dirs[:, 2].flatten(), length=0.01, normalize=True)
    plt.savefig(os.path.join(log_path, 'extrinsics.png'))

    return info


def load_colmap(dataset_path:str) -> COLMAPDataset:
    def _quaternion_to_rotation_matrix(q:np.ndarray) -> np.ndarray:
        # qx, qy, qz, qw
        return np.array(
            [[1.0 - 2 * (q[1] ** 2 + q[2] ** 2), 2 * (q[0] * q[1] - q[2] * q[3]), 2 * (q[0] * q[2] + q[1] * q[3])], 
             [2 * (q[0] * q[1] + q[2] * q[3]), 1.0 - 2 * (q[0] ** 2 + q[2] ** 2), 2 * (q[1] * q[2] - q[0] * q[3])], 
             [2 * (q[0] * q[2] - q[1] * q[3]), 2 * (q[1] * q[2] + q[0] * q[3]), 1.0 - 2 * (q[0] ** 2 + q[1] ** 2)]], 
             dtype=q.dtype
        )

    camera_path = os.path.join(dataset_path, "sparse", "cameras.txt")
    camera_num = 0
    with open(camera_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("#"):
                continue
            else:
                camera_num += 1
                _camera_id, camera_model, width, height, fx, fy, cx, cy = line.split()
                assert camera_model == "PINHOLE"
                width, height = int(width), int(height)
                fx, fy, cx, cy = float(fx), float(fy), float(cx), float(cy)
    if camera_num != 1:
        raise NotImplementedError
    
    images_path = os.path.join(dataset_path, "sparse", "images.txt")
    image_paths = []
    poses = []
    with open(images_path, "r") as f:
        lines = f.readlines()
        index = 0
        _bottom = np.array([0., 0., 0., 1.], dtype=np.float32).reshape([1, 4])
        for line in lines:
            if line.startswith("#"):
                continue
            else:
                index += 1
                if index % 2 == 1:
                    _image_id, qw, qx, qy, qz, tx, ty, tz, _camera_id, image_name = line.split()
                    qw, qx, qy, qz = float(qw), float(qx), float(qy), float(qz)
                    tx, ty, tz = float(tx), float(ty), float(tz)
                    image_paths.append(os.path.join(dataset_path, "images", image_name))
                    rot_matrix = _quaternion_to_rotation_matrix(np.array([qx, qy, qz, qw], dtype=np.float32))
                    t = np.array([tx, ty, tz], dtype=np.float32).reshape([3, 1])
                    w2c = np.concatenate([np.concatenate([rot_matrix, t], axis=1), _bottom], axis=0)
                    poses.append(w2c)
                else:
                    continue
    
    camera_intrinsic = {
        'fx': fx, 
        'fy': fy, 
        'cx': cx, 
        'cy': cy
    }
    dataset = COLMAPDataset(image_paths, poses, camera_intrinsic, height, width)
    return dataset

def observe_colmap(dataset_path:str, log_path:str) -> dict:
    def _quaternion_to_rotation_matrix(q:np.ndarray) -> np.ndarray:
        # qx, qy, qz, qw
        return np.array(
            [[1.0 - 2 * (q[1] ** 2 + q[2] ** 2), 2 * (q[0] * q[1] - q[2] * q[3]), 2 * (q[0] * q[2] + q[1] * q[3])], 
             [2 * (q[0] * q[1] + q[2] * q[3]), 1.0 - 2 * (q[0] ** 2 + q[2] ** 2), 2 * (q[1] * q[2] - q[0] * q[3])], 
             [2 * (q[0] * q[2] - q[1] * q[3]), 2 * (q[1] * q[2] + q[0] * q[3]), 1.0 - 2 * (q[0] ** 2 + q[1] ** 2)]], 
             dtype=q.dtype
        )
    
    camera_path = os.path.join(dataset_path, "sparse", "cameras.txt")
    camera_num = 0
    with open(camera_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("#"):
                continue
            else:
                camera_num += 1
                _camera_id, camera_model, width, height, fx, fy, cx, cy = line.split()
                assert camera_model == "PINHOLE"
                width, height = int(width), int(height)
                fx, fy, cx, cy = float(fx), float(fy), float(cx), float(cy)
    if camera_num != 1:
        raise NotImplementedError
    
    images_path = os.path.join(dataset_path, "sparse", "images.txt")
    image_paths = []
    poses = []
    with open(images_path, "r") as f:
        lines = f.readlines()
        index = 0
        _bottom = np.array([0., 0., 0., 1.], dtype=np.float32).reshape([1, 4])
        for line in lines:
            if line.startswith("#"):
                continue
            else:
                index += 1
                if index % 2 == 1:
                    _image_id, qw, qx, qy, qz, tx, ty, tz, _camera_id, image_name = line.split()
                    qw, qx, qy, qz = float(qw), float(qx), float(qy), float(qz)
                    tx, ty, tz = float(tx), float(ty), float(tz)
                    image_paths.append(os.path.join(dataset_path, "images", image_name))
                    rot_matrix = _quaternion_to_rotation_matrix(np.array([qx, qy, qz, qw], dtype=np.float32))
                    t = np.array([tx, ty, tz], dtype=np.float32).reshape([3, 1])
                    w2c = np.concatenate([np.concatenate([rot_matrix, t], axis=1), _bottom], axis=0)
                    poses.append(w2c)
                else:
                    continue
    
    camera_intrinsic = {
        'fx': fx, 
        'fy': fy, 
        'cx': cx, 
        'cy': cy
    }
    
    info = {}
    info["examples_num"] = len(image_paths)
    info["all_examples_same_size"] = "True"
    info["image_height"] = height
    info["image_width"] = width
    info["all_examples_same_image_coordinate_type"] = "True"
    info["image_coordinate_type"] = "right_down"
    info["all_examples_same_camera_coordinate_type"] = "True"
    info["camera_coordinate_type"] = "right_down_in"
    info["all_examples_same_camera_model"] = "True"
    info["camera_model"] = "pinhole"
    info["all_examples_same_pose_type"] = "True"
    info["pose_type"] = "w2c"
    info["all_examples_same_camera_intrinsic"] = "True"
    info["camera_intrinsic"] = "fx: {}, fy: {}, cx: {}, cy: {}".format(camera_intrinsic["fx"], camera_intrinsic["fy"], camera_intrinsic["cx"], camera_intrinsic["cy"])
    info["camera_extrinsics"] = "see extrinsics.png"

    # use the 'right_down_in' and 'w2c' convention
    dirs = np.stack([np.sum([0, 0, 1] * pose[:3, :3].T, axis=-1) for pose in poses])
    origins = np.stack([np.sum(-1 * pose[:3, -1] * pose[:3, :3].T, axis=-1) for pose in poses])
    ax = plt.figure(figsize=(12, 8)).add_subplot(projection='3d')
    # show x, y, z axis
    ax.quiver(0, 0, 0, 1, 0, 0, length=1.0, color='r', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 1, 0, length=1.0, color='g', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, 1, length=1.0, color='b', arrow_length_ratio=0.1)
    _ = ax.quiver(origins[..., 0].flatten(), origins[..., 1].flatten(), origins[..., 2].flatten(), 
        dirs[..., 0].flatten(), dirs[..., 1].flatten(), dirs[:, 2].flatten(), length=0.5, normalize=True)
    plt.savefig(os.path.join(log_path, 'extrinsics.png'))

    return info
