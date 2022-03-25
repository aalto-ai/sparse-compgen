import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def unfold_neighbourhood(image, kernel_size, pad_value):
    # N x H x W x C => N x C x H x W
    channels = image.permute(0, 3, 1, 2)
    padded = F.pad(channels, (1, 1, 1, 1), value=pad_value)

    # N x C x Hp x Wp => N x Hp x Wp x C
    hwc_padded = padded.permute(0, 2, 3, 1)

    # N x Hp x Wp x C => N x H x Wp x C x H_u
    unfolded_h = hwc_padded.unfold(1, 3, 1)

    # N x H x Wp x C x H_u => N x H x W x C x H_u x W_u
    unfolded_w = unfolded_h.unfold(2, 3, 1)

    # N x H x W x C x H_u x W_u => N x H x W x H_u x W_u x C
    return unfolded_w.permute(0, 1, 2, 4, 5, 3)


class SpatialMinMaxNormalization(nn.Module):
    def forward(self, x):
        reshaped = x.flatten(-3)
        minimum = reshaped.min(dim=-1)[0][..., None].detach()
        maximum = reshaped.max(dim=-1)[0][..., None].detach()
        scale = maximum - minimum + 10e-5

        normalized = (reshaped - minimum) / scale

        return normalized.view(x.shape)


class Readjustment(nn.Module):
    def __init__(self):
        super().__init__()
        self.adjust_bias = nn.Parameter(torch.tensor(0.0))
        self.norm = SpatialMinMaxNormalization()

    def forward(self, x):
        return self.norm(torch.relu(x - self.adjust_bias))


class Masking(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.linear = nn.Linear(emb_dim, 1, bias=False)

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        return torch.clamp(x @ self.linear.weight.transpose(-2, -1), 0, 1)


class SigmoidMasking(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.linear = nn.Linear(emb_dim, 1, bias=True)

    def forward(self, x):
        return self.linear(x).sigmoid()


class Gamma(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self):
        return torch.sigmoid(self.weight)


class MVProp2D(nn.Module):
    def __init__(self, emb_dim, n_iterations=10, hard_gather_map=False):
        super().__init__()
        self.gamma = Gamma()
        self.gather = SigmoidMasking(emb_dim)
        self.norm = Readjustment()
        self.value_flow_logits = torch.tensor(
            [[-10.0, 10.0, -10.0], [10.0, 10.0, 10.0], [-10.0, 10.0, -10.0]],
            dtype=torch.float,
        )
        self.hard_gather_map = hard_gather_map
        self.n_iterations = n_iterations

    def forward(self, image, rewarding_states):
        B, H, W, E = image.shape

        # B x H x W value propagation mask
        if self.hard_gather_map:
            gather_map = (self.gather(image).detach() > 0.5).float()
        else:
            gather_map = self.gather(image)

        # B x H x W x F => # B x H x W x 1
        reward_locations = self.norm(rewarding_states)

        # B x H x W values
        value_map = reward_locations
        value_flow = torch.sigmoid(self.value_flow_logits).to(value_map.device)

        gamma_eval = self.gamma()

        for i in range(self.n_iterations):
            # B x H x W x H_u x W_u values
            # gamma = gamma_eval if i > 0 else 1

            value_map_unfolded = unfold_neighbourhood(
                value_map, kernel_size=3, pad_value=0.0
            ).squeeze(-1)
            value_map_candidate = gather_map * (
                (gamma_eval * value_map_unfolded) * value_flow
            ).flatten(3).max(dim=-1)[0].unsqueeze(-1)

            # Concatenate with the current value map, pick
            # the max value that we've currently observed
            value_map = (
                torch.cat([value_map_candidate, value_map], dim=-1)
                .max(dim=-1)[0]
                .unsqueeze(-1)
            )

        # Mask one more time, which should mask out the
        # rewards from the value map
        value_map = torch.relu(value_map - reward_locations)
        return gather_map, value_map, reward_locations
