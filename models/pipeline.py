import torch

from .ray_casting import generate_rays
from .ray_sampling import stratified_sample, hierarchical_sample
from .positional_encoding import FourierFeatureMapping
from .neural_radiance_field import MLP
from .ray_marching import volume_rendering
from utils.chunking import get_chunks

def forward_pipeline(height:int, width:int, image_coordinate_type:str, camera_coordinate_type:str, camera_model:str, pose_type:str, 
    camera_intrinsic:dict, camera_extrinsic:torch.Tensor, random_slice:torch.Tensor, z_near:float, z_far:float, samples_num:int, samples_num_prime:int, 
    x_embedder:FourierFeatureMapping, d_embedder:FourierFeatureMapping, coarse_nerf:MLP, fine_nerf:MLP, 
    chunk_size:int, white_bkgd:bool) -> tuple:
    """
    Params:
        height, width: the height and width of the output maps
        image_coordinate_type: the type of the coordinate convention of the output maps
        camera_coordinate_type: the type of the coordinate convention of the used camera
        camera_model: the type of the used camera model
        camera_intrinsic: the intrinsic of the used camera
        camera_extrinsic: the extrinsic of the used camera
        random_slice: the random slice of sampled pixels rays, None for whole pixels rays
        z_near, z_far: the accessible space z value range for all rays to be sampled
        samples_num: the num of samples along each ray for stratified sampling
        samples_num_prime: the num of samples along each ray for hierarchical sampling
        x_embedder: the used positional encoder for x
        d_embedder: the used positional encoder for d
        coarse_nerf: the used coarse neural radiance field
        fine_nerf: the used fine neural radiance field
        chunk_size: the size of the chunk to be used for parallel processing
        white_bkgd: whether to set white background
    ----------
    Return:
        coarse_rgb_map: torch.Tensor((height, width, 3)D), rgb coarse map
        coarse_depth_map: torch.Tensor((height, width)D), depth coarse map
        coarse_disp_map: torch.Tensor((height, width)D), disparity coarse map
        coarse_acc_map: torch.Tensor((height, width)D), accumulated alpha coarse map
        fine_rgb_map: torch.Tensor((height, width, 3)D), rgb fine map
        fine_depth_map: torch.Tensor((height, width)D), depth fine map
        fine_disp_map: torch.Tensor((height, width)D), disparity fine map
        fine_acc_map: torch.Tensor((height, width)D), accumulated alpha fine map
    ----------
    Note:
        one camera pose to one maps
        use same named variables in two pass to avoid out-of-memory issue
    """
    # ray casting
    rays_o, rays_d = generate_rays(height, width, image_coordinate_type, camera_coordinate_type, camera_model, pose_type, 
        camera_intrinsic, camera_extrinsic)
    if random_slice is None:
        _rays_d_mixed = rays_d[..., None, :].expand((height, width, samples_num, 3)).reshape((-1, 3))
        _rays_d_combined_mixed = rays_d[..., None, :].expand((height, width, samples_num+samples_num_prime, 3)).reshape((-1, 3))
    else:
        rays_o = rays_o.reshape((-1, 3))[random_slice]
        rays_d = rays_d.reshape((-1, 3))[random_slice]
        _rays_d_mixed = rays_d[..., None, :].expand((rays_d.shape[0], samples_num, 3)).reshape((-1, 3))
        _rays_d_combined_mixed = rays_d[..., None, :].expand((rays_d.shape[0], samples_num+samples_num_prime, 3)).reshape((-1, 3))

    # ray sampling stratified
    pts, z_vals = stratified_sample(rays_o, rays_d, z_near, z_far, samples_num)
    _pts_mixed = pts.reshape((-1, 3))

    # positional encoding
    pts_encoded = x_embedder(_pts_mixed)
    dirs_encoded = d_embedder(_rays_d_mixed)

    # chunking
    pts_chunks = get_chunks(pts_encoded, chunk_size)
    dirs_chunks = get_chunks(dirs_encoded, chunk_size)

    # coarse neural radiance field querying
    coarse_raws = []
    for pts_chunk, dirs_chunk in zip(pts_chunks, dirs_chunks):
        coarse_raws.append(coarse_nerf(pts_chunk, dirs_chunk))
    if random_slice is None:
        coarse_raws = torch.cat(coarse_raws, dim=0).reshape((height, width, samples_num, 4))
    else:
        coarse_raws = torch.cat(coarse_raws, dim=0).reshape((rays_d.shape[0], samples_num, 4))

    # ray marching
    coarse_rgb_map, coarse_depth_map, coarse_disp_map, coarse_acc_map, coarse_weights = volume_rendering(z_vals, coarse_raws, white_bkgd)

    # ray sampling hierarchical
    pts_combined, z_vals_combined, z_vals_prime = hierarchical_sample(rays_o, rays_d, z_vals, coarse_weights, samples_num_prime)
    _pts_combined_mixed = pts_combined.reshape((-1, 3))

    # positional encoding
    pts_combined_encoded = x_embedder(_pts_combined_mixed)
    dirs_combined_encoded = d_embedder(_rays_d_combined_mixed)

    # chunking
    pts_combined_chunks = get_chunks(pts_combined_encoded, chunk_size)
    dirs_combined_chunks = get_chunks(dirs_combined_encoded, chunk_size)

    # fine neural radiance field querying
    fine_raws = []
    for pts_combined_chunk, dirs_combined_chunk in zip(pts_combined_chunks, dirs_combined_chunks):
        fine_raws.append(fine_nerf(pts_combined_chunk, dirs_combined_chunk))
    if random_slice is None:
        fine_raws = torch.cat(fine_raws, dim=0).reshape((height, width, samples_num+samples_num_prime, 4))
    else:
        fine_raws = torch.cat(fine_raws, dim=0).reshape((rays_d.shape[0], samples_num+samples_num_prime, 4))

    # ray marching
    fine_rgb_map, fine_depth_map, fine_disp_map, fine_acc_map, fine_weights = volume_rendering(z_vals_combined, fine_raws, white_bkgd)

    return (coarse_rgb_map, coarse_depth_map, coarse_disp_map, coarse_acc_map, fine_rgb_map, fine_depth_map, fine_disp_map, fine_acc_map)
