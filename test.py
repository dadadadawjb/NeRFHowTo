import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import tqdm

from utils.config import config_parse
from utils.dataset import load_helloworld, load_blender, load_llff, load_colmap
from models.positional_encoding import FourierFeatureMapping
from models.neural_radiance_field import MLP
from models.pipeline import forward_pipeline
from utils.eval import calculate_psnr_image

if __name__ == '__main__':
    args = config_parse()
    print(args.expname)
    if not os.path.exists(os.path.join(args.log_path, args.expname)):
        print("experiment has not been trained")
        exit(-1)
    else:
        if not os.path.exists(os.path.join(args.log_path, args.expname, 'test')):
            os.mkdir(os.path.join(args.log_path, args.expname, 'test'))
        else:
            print("experiment has already been tested")
            exit(-1)
    log_file = open(os.path.join(args.log_path, args.expname, 'test', 'log.txt'), 'w')
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

    # test
    print("start testing")
    test_psnr = []
    # each iteration test one (image, camera) pair
    with torch.no_grad():
        for iteration, index in tqdm.tqdm(enumerate(dataset.testing_indices)):
            image_coordinate_type, image = dataset.get_image(index)
            camera_coordinate_type, camera_model, pose_type, camera_intrinsic, camera_extrinsic = dataset.get_camera(index)
            image = image.to(device)
            camera_extrinsic = camera_extrinsic.to(device)

            coarse_rgb_map, coarse_depth_map, coarse_disp_map, coarse_acc_map, fine_rgb_map, fine_depth_map, fine_disp_map, fine_acc_map = forward_pipeline(image.shape[0], image.shape[1], 
                image_coordinate_type, camera_coordinate_type, camera_model, pose_type, camera_intrinsic, camera_extrinsic, None, 
                args.z_near, args.z_far, args.samples_num, args.samples_num_prime, x_embedder, d_embedder, coarse_nerf, fine_nerf, args.chunk_size, args.white_bkgd)
            
            psnr = calculate_psnr_image(fine_rgb_map, image)
            test_psnr.append(psnr)
            if args.verbose:
                print("iteration: {}, psnr: {}".format(iteration, psnr))
            # clamp to [0., 1.] for possible numerical error
            fine_rgb_map = fine_rgb_map.detach().cpu().numpy()
            fine_rgb_map = np.maximum(np.minimum(fine_rgb_map, np.ones_like(fine_rgb_map)), np.zeros_like(fine_rgb_map))
            plt.imsave(os.path.join(args.log_path, args.expname, 'test', '{:03d}_show.png'.format(iteration)), fine_rgb_map)
    
    print("average test psnr: {}".format(sum(test_psnr) / len(test_psnr)))
    print("test psnr: {}".format(test_psnr), file=log_file)
    print("average test psnr: {}".format(sum(test_psnr) / len(test_psnr)), file=log_file)
    log_file.close()
    print("finish testing")
