import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .harness import ImageDiscriminatorHarness


def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class FiLM(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=imm_channels, kernel_size=1
        )
        # self.bn1 = nn.BatchNorm2d(imm_channels)
        self.conv2 = nn.Conv2d(
            in_channels=imm_channels, out_channels=out_features, kernel_size=1
        )
        # self.bn2 = nn.BatchNorm2d(out_features)

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y):
        if os.environ.get("DEBUG", "0") == "1":
            import pdb

            pdb.set_trace()
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        weight = self.weight(y).unsqueeze(2).unsqueeze(3)
        bias = self.bias(y).unsqueeze(2).unsqueeze(3)
        out = x * weight + bias
        return F.relu(out)


class FiLMConvEncoder(nn.Module):
    def __init__(self, attrib_offsets, emb_dim, n_words, imm_dim=128, layer_mults=None):
        super().__init__()
        layer_mults = layer_mults or [2, 1]
        len_offsets = len(attrib_offsets) - 1

        self.register_buffer(
            "attrib_offsets", torch.tensor(attrib_offsets, dtype=torch.long)
        )
        self.attrib_embeddings = nn.Embedding(attrib_offsets[-1], emb_dim)
        self.word_embeddings = nn.Embedding(n_words, emb_dim)
        self.word_gru = nn.GRU(emb_dim, emb_dim, bidirectional=True)
        self.film = FiLM(emb_dim * len_offsets, imm_dim, emb_dim * len_offsets, imm_dim)

    def forward(self, images, missions):
        mission_s = missions.transpose(0, 1)
        mission_words = self.word_embeddings(mission_s)
        mission_enc = (
            self.word_gru(
                mission_words,
                torch.zeros(
                    2,
                    mission_words.shape[1],
                    mission_words.shape[-1],
                    device=mission_words.device,
                ),
            )[1]
            .transpose(0, 1)
            .reshape(missions.shape[0], -1)
        )

        sep_image_components = self.attrib_embeddings(
            images.long() + self.attrib_offsets[: images.shape[-1]]
        )
        sep_image_components_t = sep_image_components.transpose(-2, -3).transpose(
            -3, -4
        )
        cat_image_components = sep_image_components.flatten(-2)

        filmed_cat_image_components = (
            self.film(
                cat_image_components.transpose(-1, -3).transpose(-2, -1), mission_enc
            )
            .transpose(-1, -3)
            .transpose(-2, -3)
        )

        return filmed_cat_image_components, sep_image_components_t


class FiLMDiscriminatorHarness(ImageDiscriminatorHarness):
    def __init__(self, attrib_offsets, emb_dim, n_words, lr=10e-4, layer_mults=None):
        super().__init__(attrib_offsets, emb_dim, 128, lr=lr)
        self.film_encoder = FiLMConvEncoder(
            attrib_offsets, emb_dim, n_words, imm_dim=128, layer_mults=layer_mults
        )

    def forward(self, x):
        image, mission = x
        return self.film_encoder(image, mission)
