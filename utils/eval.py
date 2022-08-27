import torch

def calculate_psnr_mse(mse:torch.Tensor) -> float:
    return (-10.0 * torch.log10(mse)).item()

def calculate_psnr_image(image1:torch.Tensor, image2:torch.Tensor) -> float:
    mse = torch.mean((image1 - image2) ** 2)
    return calculate_psnr_mse(mse)
