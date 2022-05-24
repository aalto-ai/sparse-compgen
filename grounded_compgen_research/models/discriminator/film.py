import os

import torch.nn as nn
import torch.nn.functional as F

from ..interaction.film import FiLMConvEncoder
from .harness import ImageDiscriminatorHarness


class FiLMDiscriminatorHarness(ImageDiscriminatorHarness):
    def __init__(self, attrib_offsets, emb_dim, n_words, lr=10e-4, layer_mults=None):
        super().__init__(attrib_offsets, emb_dim, lr=lr)
        self.film_encoder = FiLMConvEncoder(
            attrib_offsets, emb_dim, n_words, imm_dim=128, layer_mults=layer_mults
        )

    def forward(self, x):
        image, mission = x
        return self.film_encoder(image, mission)
