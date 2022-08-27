import torch

def get_chunks(input:torch.Tensor, chunk_size:int) -> list:
    """
    Params:
        input: torch.Tensor((N, feature_num)D)
        chunk_size: the size of each chunk
    ----------
    Return:
        chunks: list((chunk_size)D), each as torch.Tensor((chunk_size, feature_num)D)
    ----------
    Note:
        to avoid out-of-memory issue, each chunk as an original batch
    """
    return [input[i:i+chunk_size] for i in range(0, input.shape[0], chunk_size)]
