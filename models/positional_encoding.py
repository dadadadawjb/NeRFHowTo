import torch
import torch.nn as nn

class FourierFeatureMapping(nn.Module):
    def __init__(self, input_size:int, freq_num:int, freq_type:str) -> None:
        super().__init__()
        self.input_size = input_size
        self.freq_num = freq_num
        self.output_size = input_size * (1 + 2 * freq_num)
        self.embed_fns = []

        # self
        self.embed_fns.append(lambda x: x)
        # frequency expansion
        if freq_type == "log":
            # 2^0, 2^1, 2^2, ..., 2^(freq_num-1)
            freq_bands = 2.0 ** torch.linspace(0.0, freq_num-1, freq_num)
        elif freq_type == "linear":
            # 2^0, 2^0+delta, 2^0+2delta, ..., 2^(freq_num-1)
            freq_bands = torch.linspace(2.0**0.0, 2.0**(freq_num-1), freq_num)
        else:
            raise NotImplementedError
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(freq * x))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(freq * x))
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return torch.cat([fn(x) for fn in self.embed_fns], dim=-1)
