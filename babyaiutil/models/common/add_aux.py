import torch
import torch.nn as nn

class AddAuxLayer(nn.Module):
    def __init__(self, input_dim, aux_dim):
        super().__init__()
        self.project = nn.Linear(aux_dim, input_dim)
        self.combine = nn.Linear(input_dim * 2, input_dim)

    def forward(self, x, aux):
        # Expand aux so that dims are the same as x
        # we assume that aux and x both have batch
        # dims
        aux = self.project(aux)[
            tuple(
                [slice(None, None)]
                + [None] * (x.dim() - aux.dim())
                + [slice(None, None)]
            )
        ].expand_as(x)

        return self.combine(torch.cat([x, aux], dim=-1))
