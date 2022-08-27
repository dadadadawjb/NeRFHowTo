import torch
import torch.nn as nn

class MLP(nn.Module):
    """
                          \sigma
                          ^
                          |
    x -> layers1 -> layers2 -> layers3 -> rgb
                    ^          ^
                    |          |
                    x          d
    """
    def __init__(self, x_input_size:int, d_input_size:int, 
        width1:int, depth1:int, width2:int, depth2:int, width3:int, depth3:int) -> None:
        super().__init__()
        self.x_input_size = x_input_size
        self.d_input_size = d_input_size

        self.layers1 = nn.ModuleList(
            [nn.Linear(x_input_size, width1)] + 
            [nn.Linear(width1, width1) for _ in range(depth1-1)]
        )
        self.layers2 = nn.ModuleList(
            [nn.Linear(width1 + x_input_size, width2)] + 
            [nn.Linear(width2, width2) for _ in range(depth2-1)]
        )
        self.sigma_layer = nn.Linear(width2, 1)
        self.sigma_constraint = nn.ReLU()
        self.layers3 = nn.ModuleList(
            [nn.Linear(width2 + d_input_size, width3)] + 
            [nn.Linear(width3, width3) for _ in range(depth3-1)]
        )
        self.rgb_layer = nn.Linear(width3, 3)
        self.rgb_constraint = nn.Sigmoid()
        self.activation = nn.ReLU()
    
    def forward(self, x:torch.Tensor, d:torch.Tensor) -> torch.Tensor:
        x_copy = x
        for layer in self.layers1:
            x = self.activation(layer(x))
        
        x = torch.cat([x, x_copy], dim=-1)
        for layer in self.layers2:
            x = self.activation(layer(x))
        
        sigma = self.sigma_constraint(self.sigma_layer(x))

        x = torch.cat([x, d], dim=-1)
        for layer in self.layers3:
            x = self.activation(layer(x))
        
        rgb = self.rgb_constraint(self.rgb_layer(x))

        return torch.cat([rgb, sigma], dim=-1)
