import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import tqdm
import matplotlib.pyplot as plt

from utils.config import config_parse
from utils.dataset import load_helloworld, load_blender, load_llff, load_colmap
from models.positional_encoding import FourierFeatureMapping
from models.neural_radiance_field import MLP
from models.pipeline import forward_pipeline
from utils.eval import calculate_psnr_mse
from utils.draw import draw_line

if __name__ == '__main__':
    args = config_parse()
    print(args.expname)
    if not os.path.exists(os.path.join(args.log_path, args.expname)):
        os.mkdir(os.path.join(args.log_path, args.expname))
    else:
        if not os.path.exists(os.path.join(args.log_path, args.expname, 'train')):
            os.mkdir(os.path.join(args.log_path, args.expname, 'train'))
        else:
            print("experiment has already been trained")
            exit(-1)
    with open(os.path.join(args.log_path, args.expname, 'train', 'args.txt'), 'w') as f:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            f.write('{} = {}\n'.format(arg, attr))
    log_file = open(os.path.join(args.log_path, args.expname, 'train', 'log.txt'), 'w')
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
    model_params = list(coarse_nerf.parameters()) + list(fine_nerf.parameters())
    optimizer = optim.Adam(model_params, lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    loss_fn = nn.functional.mse_loss
    print("finish initializing models")

    # loop
    print("start training")
    loss_list = []      # training along iterations
    psnr_list = []      # validation along epochs
    for epoch in tqdm.trange(args.epochs):
        coarse_nerf.train()
        fine_nerf.train()
        train_loss = 0.0
        # each iteration train one (image, camera) pair
        for iteration, index in enumerate(dataset.training_indices):
            image_coordinate_type, image = dataset.get_image(index)
            camera_coordinate_type, camera_model, pose_type, camera_intrinsic, camera_extrinsic = dataset.get_camera(index)
            height, width = image.shape[0], image.shape[1]
            if args.rays_num != -1:
                image = image.reshape((-1, 3))
                random_slice = torch.randperm(image.shape[0])[:args.rays_num]
                image = image[random_slice]
            else:
                random_slice = None
            image = image.to(device)
            camera_extrinsic = camera_extrinsic.to(device)

            coarse_rgb_map, coarse_depth_map, coarse_disp_map, coarse_acc_map, fine_rgb_map, fine_depth_map, fine_disp_map, fine_acc_map = forward_pipeline(height, width, 
                image_coordinate_type, camera_coordinate_type, camera_model, pose_type, camera_intrinsic, camera_extrinsic, random_slice, 
                args.z_near, args.z_far, args.samples_num, args.samples_num_prime, x_embedder, d_embedder, coarse_nerf, fine_nerf, args.chunk_size, args.white_bkgd)
            
            loss = loss_fn(fine_rgb_map, image)
            loss_list.append(loss.item())
            train_loss += loss.item()
            if args.verbose:
                print("iteration: {}, loss: {}".format(iteration, loss.item()))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_loss /= len(dataset.training_indices)

        coarse_nerf.eval()
        fine_nerf.eval()
        val_psnr = 0.0
        # each iteration validate one (image, camera) pair
        with torch.no_grad():
            for iteration, index in enumerate(dataset.validation_indices):
                image_coordinate_type, image = dataset.get_image(index)
                camera_coordinate_type, camera_model, pose_type, camera_intrinsic, camera_extrinsic = dataset.get_camera(index)
                height, width = image.shape[0], image.shape[1]
                if args.val_accelerate:
                    image = image.reshape((-1, 3))
                    random_slice = torch.randperm(image.shape[0])[:args.rays_num]
                    image = image[random_slice]
                else:
                    random_slice = None
                image = image.to(device)
                camera_extrinsic = camera_extrinsic.to(device)

                coarse_rgb_map, coarse_depth_map, coarse_disp_map, coarse_acc_map, fine_rgb_map, fine_depth_map, fine_disp_map, fine_acc_map = forward_pipeline(height, width, 
                    image_coordinate_type, camera_coordinate_type, camera_model, pose_type, camera_intrinsic, camera_extrinsic, random_slice, 
                    args.z_near, args.z_far, args.samples_num, args.samples_num_prime, x_embedder, d_embedder, coarse_nerf, fine_nerf, args.chunk_size, args.white_bkgd)
                
                loss = loss_fn(fine_rgb_map, image)
                psnr = calculate_psnr_mse(loss)
                val_psnr += psnr
                if args.verbose:
                    print("iteration: {}, psnr: {}".format(iteration, psnr))
        val_psnr /= len(dataset.validation_indices)
        psnr_list.append(val_psnr)

        # show time
        with torch.no_grad():
            image_coordinate_type, image = dataset.get_image(dataset.show_index)
            camera_coordinate_type, camera_model, pose_type, camera_intrinsic, camera_extrinsic = dataset.get_camera(dataset.show_index)
            image = image.to(device)
            camera_extrinsic = camera_extrinsic.to(device)

            coarse_rgb_map, coarse_depth_map, coarse_disp_map, coarse_acc_map, fine_rgb_map, fine_depth_map, fine_disp_map, fine_acc_map = forward_pipeline(image.shape[0], image.shape[1], 
                image_coordinate_type, camera_coordinate_type, camera_model, pose_type, camera_intrinsic, camera_extrinsic, None, 
                args.z_near, args.z_far, args.samples_num, args.samples_num_prime, x_embedder, d_embedder, coarse_nerf, fine_nerf, args.chunk_size, args.white_bkgd)
            
            loss = loss_fn(fine_rgb_map, image)
            show_psnr = calculate_psnr_mse(loss)
            if args.verbose:
                print("show psnr: {}".format(show_psnr))
            # clamp to [0., 1.] for possible numerical error
            fine_rgb_map = fine_rgb_map.detach().cpu().numpy()
            fine_rgb_map = np.maximum(np.minimum(fine_rgb_map, np.ones_like(fine_rgb_map)), np.zeros_like(fine_rgb_map))
            plt.imsave(os.path.join(args.log_path, args.expname, 'train', '{:03d}_show.png'.format(epoch)), fine_rgb_map)

        print(f"train loss: {train_loss:.4f}, val psnr: {val_psnr:.4f}, show psnr: {show_psnr:.4f}, lr: {scheduler.get_last_lr()[0]:.6f}")
        print(f"train loss: {train_loss:.4f}, val psnr: {val_psnr:.4f}, show psnr: {show_psnr:.4f}, lr: {scheduler.get_last_lr()[0]:.6f}", file=log_file)
        scheduler.step()
    print("finish training")

    # save results
    print("start saving results")
    torch.save(coarse_nerf.state_dict(), os.path.join(args.log_path, args.expname, 'train', 'coarse_nerf.pth'))
    torch.save(fine_nerf.state_dict(), os.path.join(args.log_path, args.expname, 'train', 'fine_nerf.pth'))
    draw_line(loss_list, 1, title='train_loss', xlabel='iteration', ylabel='loss', 
                path=os.path.join(args.log_path, args.expname, 'train', 'train_loss.png'), ylimit=False)
    draw_line(psnr_list, 1, title='val_psnr', xlabel='epoch', ylabel='psnr', 
                path=os.path.join(args.log_path, args.expname, 'train', 'val_psnr.png'), ylimit=False)
    log_file.close()
    print("finish saving results")
