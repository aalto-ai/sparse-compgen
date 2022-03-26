import torch
import torch.nn as nn

from torchvision.models import resnet
from ..img_mask import ConvAttention
from .mvprop import unfold_neighbourhood


class MinMaxNorm(nn.Module):
    def forward(self, x):
        minimum = x.min(dim=-1)[0][..., None].detach()
        maximum = x.max(dim=-1)[0][..., None].detach()
        scale = maximum - minimum + 10e-5

        normalized = (x - minimum) / scale

        return normalized.reshape(x.shape)


class ResNetValueExtractor(nn.Module):
    def __init__(self, img_emb_dim, direction_emb_dim, n_actions):
        super().__init__()
        self.dir_emb = nn.Embedding(4, direction_emb_dim)
        self.resnet = resnet.resnet18(pretrained=False, num_classes=n_actions)
        self.resnet.conv1 = nn.Conv2d(
            img_emb_dim + direction_emb_dim + 1, 64, kernel_size=3
        )

    def forward(self, directions, image_vectors, reward_map):
        embedded_direction = self.dir_emb(directions)
        in_vectors = torch.cat(
            [
                embedded_direction[..., None, None, :].expand(
                    *image_vectors.shape[:-1], embedded_direction.shape[-1]
                ),
                image_vectors,
                reward_map,
            ],
            dim=-1,
        )
        in_channels = in_vectors.transpose(-1, -3).transpose(-2, -1)

        return self.resnet(in_channels)


class BigConvolutionValueMapExtractor(nn.Module):
    def __init__(self, img_emb_dim, n_component, dir_emb_dim, n_outputs):
        super().__init__()
        total_emb_dim = img_emb_dim * n_component + dir_emb_dim + 2
        self.dir_emb = nn.Embedding(4, dir_emb_dim)
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=total_emb_dim,
                out_channels=total_emb_dim * 4,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=total_emb_dim * 4,
                out_channels=total_emb_dim * 2,
                kernel_size=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=total_emb_dim * 2, out_channels=n_outputs, kernel_size=1
            ),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, direction, image_vectors, value_map, reward_map):
        embedded_direction = self.dir_emb(direction)

        in_vectors = torch.cat(
            [
                embedded_direction[..., None, None, :].expand(
                    *image_vectors.shape[:-1], embedded_direction.shape[-1]
                ),
                image_vectors,
                value_map,
                reward_map,
            ],
            dim=-1,
        )
        in_channels = in_vectors.transpose(-1, -3).transpose(-2, -1)

        conv_output = self.net(in_channels)

        # Return just the channels and treat them as features
        return conv_output[..., 0, 0]


class ConvolutionMaskValueMapExtractor(nn.Module):
    def __init__(self, img_emb_dim, n_component, dir_emb_dim, n_outputs):
        super().__init__()
        total_emb_dim = img_emb_dim * n_component  #  + dir_emb_dim
        self.dir_emb = nn.Embedding(4, dir_emb_dim)
        self.att = ConvAttention(total_emb_dim, [2, 2])
        self.value_norm = MinMaxNorm()
        self.linear = nn.Sequential(
            nn.Linear(9 * 2 + dir_emb_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_outputs),
        )

    def forward(self, direction, image_vectors, value_map, reward_map):
        embedded_direction = self.dir_emb(direction)

        in_vectors = image_vectors
        in_channels = in_vectors.transpose(-1, -3).transpose(-2, -1)

        # B x C x H x W => B x H x W x C
        mask = self.att(in_channels).permute(0, 2, 3, 1)

        # unfold the value and reward maps
        # and drop channels dim, flattening last two dimensions
        # into vectors
        value_map_unfold = unfold_neighbourhood(value_map, 3, 0).squeeze(-1)
        value_map_unfold = self.value_norm(
            value_map_unfold.reshape(*value_map_unfold.shape[:-2], -1)
        )
        reward_map_unfold = unfold_neighbourhood(reward_map, 3, 0).squeeze(-1)
        reward_map_unfold = reward_map_unfold.reshape(*reward_map_unfold.shape[:-2], -1)

        # B x H x W x C => B x C
        masked_value_map_unfold = (mask * value_map_unfold).sum(dim=-2).sum(dim=-2)
        masked_reward_map_unfold = (mask * reward_map_unfold).sum(dim=-2).sum(dim=-2)

        if os.getenv("DEBUG", "0") != "0":
            import pdb

            pdb.set_trace()

        return self.linear(
            torch.cat(
                [masked_value_map_unfold, masked_reward_map_unfold, embedded_direction],
                dim=-1,
            )
        )
