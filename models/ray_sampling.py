import torch

def stratified_sample(rays_o:torch.Tensor, rays_d:torch.Tensor, z_near:float, z_far:float, samples_num:int) -> tuple:
    """
    Params:
        rays_o: torch.Tensor((height, width, 3)D) / torch.Tensor((rays_num, 3)D), each pixel's ray's origin in world coordinate
        rays_d: torch.Tensor((height, width, 3)D) / torch.Tensor((rays_num, 3)D), each pixel's ray's unit direction in world coordinate
        z_near: the accessible space nearest z value for all rays
        z_far: the accessible space farthest z value for all rays
        samples_num: the num of samples along each ray
    ----------
    Return:
        pts: torch.Tensor((height, width, samples_num, 3)D) / torch.Tensor((rays_num, samples_num, 3)D), sampled points' coordinates in world coordinate
        z_vals: torch.Tensor((height, width, samples_num)D) / torch.Tensor((rays_num, samples_num)D), sampled points' z values in each ray
    ----------
    Note:
        [z_near, z_far]
    """
    # uniform sample
    t_vals = torch.linspace(0.0, 1.0, samples_num).to(rays_o.device)
    z_vals = z_near * (1.0 - t_vals) + z_far * t_vals

    # perturb
    z_mids = 0.5 * (z_vals[1:] + z_vals[:-1])
    z_uppers = torch.cat([z_mids, z_vals[-1:]], dim=-1)
    z_lowers = torch.cat([z_vals[:1], z_mids], dim=-1)
    t_rand = torch.rand([samples_num]).to(rays_o.device)
    z_vals = z_lowers * (1.0 - t_rand) + z_uppers * t_rand
    z_vals = z_vals.expand(list(rays_o.shape[:-1]) + [samples_num])

    pts = rays_o[..., None, :] + z_vals[..., :, None] * rays_d[..., None, :]
    return (pts, z_vals)


def hierarchical_sample(rays_o:torch.Tensor, rays_d:torch.Tensor, z_vals:torch.Tensor, weights:torch.Tensor, samples_num_prime:int) -> tuple:
    """
    Params:
        rays_o: torch.Tensor((height, width, 3)D) / torch.Tensor((rays_num, 3)D), each pixel's ray's origin in world coordinate
        rays_d: torch.Tensor((height, width, 3)D) / torch.Tensor((rays_num, 3)D), each pixel's ray's unit direction in world coordinate
        z_vals: torch.Tensor((height, width, samples_num)D) / torch.Tensor((rays_num, samples_num)D), sampled points' z values in each ray
        weights: torch.Tensor((height, width, samples_num)D) / torch.Tensor((rays_num, samples_num)D), sampled points' contribution weights in each ray
        samples_num_prime: the num of new samples along each ray
    ----------
    Return:
        pts_combined: torch.Tensor((height, width, samples_num+samples_num_prime, 3)D) / torch.Tensor((rays_num, samples_num+samples_num_prime, 3)D), old+new sampled points' coordinates in world coordinate
        z_vals_combined: torch.Tensor((height, width, samples_num+samples_num_prime)D) / torch.Tensor((rays_num, samples_num+samples_num_prime)D), old+new sampled points' z values in each ray
        z_vals_prime: torch.Tensor((height, width, samples_num_prime)D) / torch.Tensor((rays_num, samples_num_prime)D), new sampled points' z values in each ray
    """
    def _sample_pdf(bins:torch.Tensor, _weights:torch.Tensor, _samples_num_prime:int) -> torch.Tensor:
        """
        Params:
            bins: torch.Tensor((..., samples_num-1)D), use samples' middle points to represent each bin
            _weights: torch.Tensor((..., samples_num-1-1)D), use samples' weights to represent each bin's weight
            _samples_num_prime: the num of new samples
        ----------
        Return:
            new_samples: torch.Tensor((..., _samples_num_prime)D), new samples
        ----------
        Note:
            a bit of tricky
            samples:        |---|---|---|---|---|---|
            bins:             |---|---|---|---|---|
            _weights:           |---|---|---|---|
        """
        # get pdf
        pdf = (_weights + 1e-5) / torch.sum(_weights + 1e-5, dim=-1, keepdims=True)   # (..., samples_num-1-1)

        # pdf2cdf
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)                  # (..., samples_num-1)

        # sample position
        u = torch.linspace(0.0, 1.0, _samples_num_prime).to(cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [_samples_num_prime])                       # (..., _samples_num_prime)
        u = u.contiguous()
        indices = torch.searchsorted(cdf, u, right=True)                                # (..., _samples_num_prime)
        indices_below = torch.clamp(indices-1, min=0)
        indices_above = torch.clamp(indices, max=cdf.shape[-1]-1)
        indices = torch.stack([indices_below, indices_above], dim=-1)                   # (..., _samples_num_prime, 2)

        # sample
        _matched_shape = list(indices.shape[:-1]) + [cdf.shape[-1]]                     # (..., _samples_num_prime, samples_num-1)
        cdf = torch.gather(cdf.unsqueeze(-2).expand(_matched_shape), dim=-1, index=indices)
        bins = torch.gather(bins.unsqueeze(-2).expand(_matched_shape), dim=-1, index=indices)
        denom = cdf[..., 1] - cdf[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t_vals = (u - cdf[..., 0]) / denom
        new_samples = bins[..., 0] * (1.0 - t_vals) + bins[..., 1] * t_vals             # (..., _samples_num_prime)

        return new_samples

    # sample according to pdf
    z_mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])                         # (height, width, samples_num-1) / (rays_num, samples_num-1)
    z_vals_prime = _sample_pdf(z_mids, weights[..., 1:-1], samples_num_prime)   # (height, width, samples_num_prime) / (rays_num, samples_num_prime)
    z_vals_prime = z_vals_prime.detach()

    # combine
    z_vals_combined, _ = torch.sort(torch.cat([z_vals, z_vals_prime], dim=-1), dim=-1)
    pts_combined = rays_o[..., None, :] + z_vals_combined[..., :, None] * rays_d[..., None, :]
    return (pts_combined, z_vals_combined, z_vals_prime)
