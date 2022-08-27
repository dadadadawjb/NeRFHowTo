import torch

def volume_rendering(z_vals:torch.Tensor, field_raws:torch.Tensor, white_bkgd:bool) -> tuple:
    """
    Params:
        z_vals: torch.Tensor((height, width, samples_num)D) / torch.Tensor((rays_num, samples_num)D), the z values of samples in each ray
        field_raws: torch.Tensor((height, width, samples_num, 4)D) / torch.Tensor((rays_num, samples_num, 4)D), the raw output of neural radiance field
        white_bkgd: whether to set white background
    ----------
    Return:
        rgb_map: torch.Tensor((height, width, 3)D) / torch.Tensor((rays_num, 3)D), rgb map
        depth_map: torch.Tensor((height, width)D) / torch.Tensor((rays_num, )D), depth map
        disp_map: torch.Tensor((height, width)D) / torch.Tensor((rays_num, )D), disparity map
        acc_map: torch.Tensor((height, width)D) / torch.Tensor((rays_num, )D), accumulated alpha map
        weight: torch.Tensor((height, width, samples_num)D) / torch.Tensor((rays_num, samples_num)D), sampled points' contribution weights
    """
    # quadrature estimation
    delta = z_vals[..., 1:] - z_vals[..., :-1]      # (height, width, samples_num-1) / (rays_num, samples_num-1)
    delta = torch.cat([delta, 1e10 * torch.ones_like(delta[..., 0:1])], dim=-1) # (height, width, samples_num) / (rays_num, samples_num)

    # from density to transparency
    alpha = 1.0 - torch.exp(-field_raws[..., -1] * delta)   # (height, width, samples_num) / (rays_num, samples_num)

    # accumulated transmittance
    T = torch.cumprod(1.0 - alpha + 1e-5, dim=-1)   # tricky, (height, width, samples_num) / (rays_num, samples_num)
    T = torch.roll(T, 1, -1)
    T[..., 0] = 1.0

    # weight
    weight = T * alpha  # (height, width, samples_num) / (rays_num, samples_num)

    # rgb map
    rgb_map = torch.sum(weight[..., None] * field_raws[..., :-1], dim=-2)   # (height, width, 3) / (rays_num, 3)
    # depth map
    depth_map = torch.sum(weight * z_vals, dim=-1)  # (height, width) / (rays_num, )
    # disparity map
    disp_map = 1.0 / torch.max(1e-5 * torch.ones_like(depth_map), depth_map / torch.sum(weight, dim=-1))    # (height, width) / (rays_num, )
    # accumulated alpha map
    acc_map = torch.sum(weight, dim=-1) # (height, width) / (rays_num, )
    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])
    
    return (rgb_map, depth_map, disp_map, acc_map, weight)
