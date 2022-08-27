import torch
import torch.nn.functional as F

def generate_rays(height:int, width:int, image_coordinate_type:str, camera_coordinate_type:str, camera_model:str, pose_type:str, 
    camera_intrinsic:dict, camera_extrinsic:torch.Tensor) -> tuple:
    """
    Params:
        height, width: image's height and width in pixel
        image_coordinate_type: either 'right_down' or 'right_up', denotes two common coordinate systems, recommend for 'right_down', 'right_up' may be not correct, 
            'right_down' as origin at the left-up corner with +x axis pointing right and +y axis pointing down, 
            'right_up' as origin at the left_down corner with +x axis pointing right and +y axis pointing up
        camera_coordinate_type: '{right/left}_{up/down}_{in/out}', denotes all 8 types of poses of camera related to image, recommend for 'right_up_out', 
            same convention as `image_coordinate_type` as xyz order
        camera_model: only 'pinhole' now, correspond to `camera_intrinsic`
        pose_type: either 'c2w' or 'w2c', correspond to `camera_extrinsic`, recommend for 'c2w'
        camera_intrinsic: if 'pinhole' then 'fx', 'fy', 'cx' and 'cy'
        camera_extrinsic: torch.Tensor((4, 4)D), [R,t|0,1] format
    ----------
    Return:
        rays_o: torch.Tensor((height, width, 3)D), each pixel's ray's origin in world coordinate
        rays_d: torch.Tensor((height, width, 3)D), each pixel's ray's unit direction in world coordinate
    """
    # TODO: clean but not elegant
    image_coordinate_type = image_coordinate_type.split('_')
    assert len(image_coordinate_type) == 2
    x_image_coordinate_type, y_image_coordinate_type = image_coordinate_type[0], image_coordinate_type[1]
    camera_coordinate_type = camera_coordinate_type.split('_')
    assert len(camera_coordinate_type) == 3
    x_camera_coordinate_type, y_camera_coordinate_type, z_camera_coordinate_type = camera_coordinate_type[0], camera_coordinate_type[1], camera_coordinate_type[2]

    # generate pixel coordinate
    '''(i, j) in right_down convention
    (1, 1), (2, 1), ..., (width, 1)
    (1, 2), (2, 2), ..., (width, 2)
    ...
    (1, height), (2, height), ..., (width, height)
    '''
    '''(i, j) in right_up convention
    (1, height), (2, height), ..., (width, height)
    ...
    (1, 2), (2, 2), ..., (width, 2)
    (1, 1), (2, 1), ..., (width, 1)
    '''
    if x_image_coordinate_type == 'right' and y_image_coordinate_type == 'down':
        i, j = torch.meshgrid(
            torch.arange(width, dtype=torch.float32).to(camera_extrinsic.device), 
            torch.arange(height, dtype=torch.float32).to(camera_extrinsic.device), 
            indexing='ij')
        i, j = i.transpose(-1, -2), j.transpose(-1, -2)
    elif x_image_coordinate_type == 'right' and y_image_coordinate_type == 'up':
        i, j = torch.meshgrid(
            torch.arange(width, dtype=torch.float32).to(camera_extrinsic.device), 
            torch.arange(height, dtype=torch.float32).to(camera_extrinsic.device), 
            indexing='ij')
        i, j = i.transpose(-1, -2), j.transpose(-1, -2)
        j = j.flip([0,])
    else:
        raise NotImplementedError
    
    # image2camera
    x_sign = 1 if x_image_coordinate_type == x_camera_coordinate_type else -1
    y_sign = 1 if y_image_coordinate_type == y_camera_coordinate_type else -1
    z_sign = 1 if z_camera_coordinate_type == 'in' else -1
    if camera_model == 'pinhole':
        x = x_sign * (i - camera_intrinsic['cx']) / camera_intrinsic['fx']
        y = y_sign * (j - camera_intrinsic['cy']) / camera_intrinsic['fy']
        z = z_sign * torch.ones_like(i)
    else:
        raise NotImplementedError
    rays_d = torch.stack([x, y, z], dim=-1)     # (height, width, 3)

    # camera2world
    if pose_type == 'c2w':
        c2w = camera_extrinsic
    elif pose_type == 'w2c':
        c2w = torch.linalg.inv(camera_extrinsic)
    else:
        raise NotImplementedError
    rays_d = torch.sum(rays_d[..., None, :] * c2w[:3, :3], dim=-1)
    rays_d = F.normalize(rays_d, dim=-1)        # (height, width, 3)
    rays_o = c2w[:3, -1].expand(rays_d.shape)   # (height, width, 3)
    return (rays_o, rays_d)
