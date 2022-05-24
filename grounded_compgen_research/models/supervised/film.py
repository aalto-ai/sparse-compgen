import os
import torch.nn as nn

from ..interaction.film import FiLMConvEncoder
from .harness import ImageSupervisedHarness


class FiLMConvEncoderProjection(nn.Module):
    def __init__(self, attrib_offsets, emb_dim, n_words, imm_dim=128, layer_mults=None):
        super().__init__()
        self.film_encoder = FiLMConvEncoder(
            attrib_offsets, emb_dim, n_words, imm_dim=128, layer_mults=layer_mults
        )
        self.projection = nn.Linear(imm_dim, 1)

    def forward(self, images, missions):
        filmed_cat_image_components, _, image_components = self.film_encoder(
            images, missions
        )

        # Note that image_mask re-embeds the image components
        # for its own use.
        projected_image_components = self.projection(
            filmed_cat_image_components
        ).squeeze(-1)

        if os.environ.get("DEBUG", "0") == "1":
            import pdb

            pdb.set_trace()

        return projected_image_components, image_components


class FiLMConvEncoderProjectionHarness(ImageSupervisedHarness):
    def __init__(self, attrib_offsets, emb_dim, n_words, lr=10e-4):
        super().__init__(lr=lr)
        self.encoder = FiLMConvEncoderProjection(attrib_offsets, emb_dim, n_words)

    def forward(self, x):
        image, mission = x
        return self.encoder(image, mission)
