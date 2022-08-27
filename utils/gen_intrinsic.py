def jitter_fx_pinhole(frame_num:int, ratio_min:float, ratio_max:float, reference_intrinsic:dict) -> list:
    intrinsics = []
    for i in range(frame_num):
        ratio = ratio_min + (ratio_max - ratio_min) * i / (frame_num - 1)
        intrinsic = {
            'fx': reference_intrinsic['fx'] * ratio, 
            'fy': reference_intrinsic['fy'], 
            'cx': reference_intrinsic['cx'], 
            'cy': reference_intrinsic['cy']
        }
        intrinsics.append(intrinsic)
    return intrinsics


def jitter_fy_pinhole(frame_num:int, ratio_min:float, ratio_max:float, reference_intrinsic:dict) -> list:
    intrinsics = []
    for i in range(frame_num):
        ratio = ratio_min + (ratio_max - ratio_min) * i / (frame_num - 1)
        intrinsic = {
            'fx': reference_intrinsic['fx'], 
            'fy': reference_intrinsic['fy'] * ratio, 
            'cx': reference_intrinsic['cx'], 
            'cy': reference_intrinsic['cy']
        }
        intrinsics.append(intrinsic)
    return intrinsics


def jitter_fxy_pinhole(frame_num:int, ratio_min:float, ratio_max:float, reference_intrinsic:dict) -> list:
    intrinsics = []
    for i in range(frame_num):
        ratio = ratio_min + (ratio_max - ratio_min) * i / (frame_num - 1)
        intrinsic = {
            'fx': reference_intrinsic['fx'] * ratio, 
            'fy': reference_intrinsic['fy'] * ratio, 
            'cx': reference_intrinsic['cx'], 
            'cy': reference_intrinsic['cy']
        }
        intrinsics.append(intrinsic)
    return intrinsics
