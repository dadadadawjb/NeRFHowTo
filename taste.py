import torch
import numpy as np
import os
import tqdm
import matplotlib.pyplot as plt

from utils.config import config_parse
from utils.dataset import load_helloworld, load_blender, load_llff, load_colmap
from models.positional_encoding import FourierFeatureMapping
from models.neural_radiance_field import MLP
from models.pipeline import forward_pipeline
from utils.gen_extrinsic import surround_poses, spiral_poses, circle_poses, surround_poses_colmap, spiral_poses_colmap
from utils.gen_intrinsic import jitter_fx_pinhole, jitter_fy_pinhole, jitter_fxy_pinhole

if __name__ == '__main__':
    args = config_parse()
    print(args.expname)
    if not os.path.exists(os.path.join(args.log_path, args.expname)):
        print("experiment has not been trained")
        exit(-1)
    else:
        if not os.path.exists(os.path.join(args.log_path, args.expname, 'taste')):
            os.mkdir(os.path.join(args.log_path, args.expname, 'taste'))
        else:
            pass
    taste_type = input("what type of taste? ('extrinsic' or 'intrinsic') ")
    assert taste_type in ['extrinsic', 'intrinsic']
    if not os.path.exists(os.path.join(args.log_path, args.expname, 'taste', taste_type)):
        os.mkdir(os.path.join(args.log_path, args.expname, 'taste', taste_type))
    else:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # initialize dataset
    print("start initializing dataset")
    if args.dataset_type == 'helloworld':
        dataset = load_helloworld(args.data_path)
    elif args.dataset_type == 'blender':
        dataset = load_blender(args.data_path)
    elif args.dataset_type == 'llff':
        dataset = load_llff(args.data_path)
    elif args.dataset_type == 'colmap':
        dataset = load_colmap(args.data_path)
    else:
        raise NotImplementedError
    print("finish initializing dataset")

    # initialize models
    print("start initializing models")
    x_embedder = FourierFeatureMapping(3, args.x_freq_num, args.x_freq_type)
    d_embedder = FourierFeatureMapping(3, args.d_freq_num, args.d_freq_type)
    coarse_nerf = MLP(x_embedder.output_size, d_embedder.output_size, 
        args.width1, args.depth1, args.width2, args.depth2, args.width3, args.depth3).to(device)
    fine_nerf = MLP(x_embedder.output_size, d_embedder.output_size, 
        args.width1, args.depth1, args.width2, args.depth2, args.width3, args.depth3).to(device)
    coarse_nerf.load_state_dict(torch.load(os.path.join(args.log_path, args.expname, "train", 'coarse_nerf.pth')))
    fine_nerf.load_state_dict(torch.load(os.path.join(args.log_path, args.expname, "train", 'fine_nerf.pth')))
    coarse_nerf.eval()
    fine_nerf.eval()
    print("finish initializing models")

    # taste
    print("start tasting")
    if taste_type == 'extrinsic':
        generate_type = input("what type of poses to generate? ('surround', 'spiral', 'circle') ")
        if not os.path.exists(os.path.join(args.log_path, args.expname, 'taste', taste_type, generate_type)):
            os.mkdir(os.path.join(args.log_path, args.expname, 'taste', taste_type, generate_type))
        else:
            print(f"experiment has already been tasted with type {taste_type}+{generate_type}")
            exit(-1)
        if generate_type == 'surround':
            frame_num = int(input("how many frames to generate? "))
            radius = float(input("radius of the generated poses ball? (you can use `observe_dataset.py` to observe appropriate radius) "))
            # use dataset's show example as other default settings
            image_coordinate_type, image = dataset.get_image(dataset.show_index)
            height, width = image.shape[0], image.shape[1]
            camera_coordinate_type, camera_model, pose_type, camera_intrinsic, _ = dataset.get_camera(dataset.show_index)
            # custom
            if args.dataset_type == 'colmap':
                camera_extrinsics = surround_poses_colmap(frame_num, radius, camera_coordinate_type, pose_type)
            else:
                camera_extrinsics = surround_poses(frame_num, radius, camera_coordinate_type, pose_type)
        elif generate_type == 'spiral':
            frame_num = int(input("how many frames to generate? "))
            radius = float(input("radius of the generated poses ball? (you can use `observe_dataset.py` to observe appropriate radius) "))
            # use dataset's show example as other default settings
            image_coordinate_type, image = dataset.get_image(dataset.show_index)
            height, width = image.shape[0], image.shape[1]
            camera_coordinate_type, camera_model, pose_type, camera_intrinsic, _ = dataset.get_camera(dataset.show_index)
            # custom
            if args.dataset_type == 'colmap':
                camera_extrinsics = spiral_poses_colmap(frame_num, radius, camera_coordinate_type, pose_type)
            else:
                camera_extrinsics = spiral_poses(frame_num, radius, camera_coordinate_type, pose_type)
        elif generate_type == 'circle':
            frame_num = int(input("how many frames to generate? "))
            distance = float(input("distance from the center of the generated poses circle? (you can use `observe_dataset.py` to observe appropriate distance) "))
            radius = float(input("radius of the generated poses circle? (you can use `observe_dataset.py` to observe appropriate radius) "))
            # use dataset's show example as other default settings
            image_coordinate_type, image = dataset.get_image(dataset.show_index)
            height, width = image.shape[0], image.shape[1]
            camera_coordinate_type, camera_model, pose_type, camera_intrinsic, _ = dataset.get_camera(dataset.show_index)
            camera_extrinsics = circle_poses(frame_num, distance, radius, camera_coordinate_type, pose_type)
        else:
            raise NotImplementedError
    elif taste_type == 'intrinsic':
        generate_type = input("what type of camera to generate? ('fx', 'fy', 'fxy') ")
        if not os.path.exists(os.path.join(args.log_path, args.expname, 'taste', taste_type, generate_type)):
            os.mkdir(os.path.join(args.log_path, args.expname, 'taste', taste_type, generate_type))
        else:
            print(f"experiment has already been tasted with type {taste_type}+{generate_type}")
            exit(-1)
        if generate_type == 'fx':
            frame_num = int(input("how many frames to generate? "))
            ratio_min = float(input("minimum ratio of the generated camera's fx to jitter? "))
            ratio_max = float(input("maximum ratio of the generated camera's fx to jitter? "))
            # use dataset's show example as other default settings
            image_coordinate_type, image = dataset.get_image(dataset.show_index)
            height, width = image.shape[0], image.shape[1]
            camera_coordinate_type, camera_model, pose_type, camera_intrinsic, camera_extrinsic = dataset.get_camera(dataset.show_index)
            assert camera_model == 'pinhole'
            camera_intrinsics = jitter_fx_pinhole(frame_num, ratio_min, ratio_max, camera_intrinsic)
        elif generate_type == 'fy':
            frame_num = int(input("how many frames to generate? "))
            ratio_min = float(input("minimum ratio of the generated camera's fy to jitter? "))
            ratio_max = float(input("maximum ratio of the generated camera's fy to jitter? "))
            # use dataset's show example as other default settings
            image_coordinate_type, image = dataset.get_image(dataset.show_index)
            height, width = image.shape[0], image.shape[1]
            camera_coordinate_type, camera_model, pose_type, camera_intrinsic, camera_extrinsic = dataset.get_camera(dataset.show_index)
            assert camera_model == 'pinhole'
            camera_intrinsics = jitter_fy_pinhole(frame_num, ratio_min, ratio_max, camera_intrinsic)
        elif generate_type == 'fxy':
            frame_num = int(input("how many frames to generate? "))
            ratio_min = float(input("minimum ratio of the generated camera's fx and fy to jitter? "))
            ratio_max = float(input("maximum ratio of the generated camera's fx and fy to jitter? "))
            # use dataset's show example as other default settings
            image_coordinate_type, image = dataset.get_image(dataset.show_index)
            height, width = image.shape[0], image.shape[1]
            camera_coordinate_type, camera_model, pose_type, camera_intrinsic, camera_extrinsic = dataset.get_camera(dataset.show_index)
            assert camera_model == 'pinhole'
            camera_intrinsics = jitter_fxy_pinhole(frame_num, ratio_min, ratio_max, camera_intrinsic)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    # each iteration taste one camera setting
    with torch.no_grad():
        if taste_type == 'extrinsic':
            for i, camera_extrinsic in tqdm.tqdm(enumerate(camera_extrinsics)):
                camera_extrinsic = camera_extrinsic.to(device)

                coarse_rgb_map, coarse_depth_map, coarse_disp_map, coarse_acc_map, fine_rgb_map, fine_depth_map, fine_disp_map, fine_acc_map = forward_pipeline(height, width, 
                    image_coordinate_type, camera_coordinate_type, camera_model, pose_type, camera_intrinsic, camera_extrinsic, None, 
                    args.z_near, args.z_far, args.samples_num, args.samples_num_prime, x_embedder, d_embedder, coarse_nerf, fine_nerf, args.chunk_size, args.white_bkgd)
                
                # clamp to [0., 1.] for possible numerical error
                fine_rgb_map = fine_rgb_map.detach().cpu().numpy()
                fine_rgb_map = np.maximum(np.minimum(fine_rgb_map, np.ones_like(fine_rgb_map)), np.zeros_like(fine_rgb_map))
                plt.imsave(os.path.join(args.log_path, args.expname, 'taste', taste_type, generate_type, '{:03d}_rgb.png'.format(i)), fine_rgb_map)
                plt.imsave(os.path.join(args.log_path, args.expname, 'taste', taste_type, generate_type, '{:03d}_d.png'.format(i)), fine_depth_map.detach().cpu().numpy())
        elif taste_type == 'intrinsic':
            for i, camera_intrinsic in tqdm.tqdm(enumerate(camera_intrinsics)):
                camera_extrinsic = camera_extrinsic.to(device)

                coarse_rgb_map, coarse_depth_map, coarse_disp_map, coarse_acc_map, fine_rgb_map, fine_depth_map, fine_disp_map, fine_acc_map = forward_pipeline(height, width, 
                    image_coordinate_type, camera_coordinate_type, camera_model, pose_type, camera_intrinsic, camera_extrinsic, None, 
                    args.z_near, args.z_far, args.samples_num, args.samples_num_prime, x_embedder, d_embedder, coarse_nerf, fine_nerf, args.chunk_size, args.white_bkgd)
                
                # clamp to [0., 1.] for possible numerical error
                fine_rgb_map = fine_rgb_map.detach().cpu().numpy()
                fine_rgb_map = np.maximum(np.minimum(fine_rgb_map, np.ones_like(fine_rgb_map)), np.zeros_like(fine_rgb_map))
                plt.imsave(os.path.join(args.log_path, args.expname, 'taste', taste_type, generate_type, '{:03d}_rgb.png'.format(i)), fine_rgb_map)
                plt.imsave(os.path.join(args.log_path, args.expname, 'taste', taste_type, generate_type, '{:03d}_d.png'.format(i)), fine_depth_map.detach().cpu().numpy())
        else:
            raise NotImplementedError
    print("finish tasting")
