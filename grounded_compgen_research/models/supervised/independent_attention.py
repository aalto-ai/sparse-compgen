import os

import torch.nn as nn
import torch.nn.functional as F

from ..interaction.independent_attention import Affine, IndependentAttentionModel
from .harness import ImageSupervisedHarness


class IndependentAttentionProjection(nn.Module):
    def __init__(self, attrib_offsets, emb_dim, n_words):
        super().__init__()
        self.encoder = IndependentAttentionModel(attrib_offsets, emb_dim, n_words)
        self.affine = Affine()

    def forward(self, images, missions):
        (cell_scores, image_components, attentions) = self.encoder(images, missions)

        # Note that image_mask re-embeds the image components
        # for its own use.
        projected_image_components = self.affine(
            cell_scores.reshape(cell_scores.shape[0], images.shape[1], images.shape[2])
        )

        if os.environ.get("DEBUG", "0") == "1":
            import pdb

            pdb.set_trace()

        return projected_image_components, image_components


class IndependentAttentionProjectionHarness(ImageSupervisedHarness):
    def __init__(self, attrib_offsets, emb_dim, n_words, lr=10e-4, l1_penalty=0):
        super().__init__(lr=lr)
        self.model = IndependentAttentionProjection(attrib_offsets, emb_dim, n_words)
        self.save_hyperparameters()

    def forward(self, x):
        image, mission = x
        return self.model(image, mission)

    def training_step(self, x, idx):
        loss = super().training_step(x, idx)

        l1c = (
            (
                F.normalize(self.model.encoder.attrib_embeddings.weight, dim=-1)
                @ F.normalize(self.model.encoder.word_embeddings.weight, dim=-1).T
            )
            .abs()
            .mean()
        )

        self.log("l1c", l1c, prog_bar=True)

        return loss + l1c * self.hparams.l1_penalty
