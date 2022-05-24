from typing import Tuple

import torch
import torch.nn.functional as F

from ..interaction.independent_attention import IndependentAttentionModel
from .harness import ImageDiscriminatorHarness


class IndependentAttentionDiscriminatorHarness(ImageDiscriminatorHarness):
    def __init__(self, attrib_offsets, emb_dim, n_words, lr=10e-4, l1_penalty=0):
        super().__init__(attrib_offsets, emb_dim, lr=lr, l1_penalty=l1_penalty)
        self.model = IndependentAttentionModel(attrib_offsets, emb_dim, n_words)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        image, mission = x
        return self.model(image, mission)

    def training_step(self, x, idx):
        loss = super().training_step(x, idx)

        l1c = (
            (
                F.normalize(self.model.attrib_embeddings.weight, dim=-1)
                @ F.normalize(self.model.word_embeddings.weight, dim=-1).T
            )
            .abs()
            .mean()
        )

        self.log("l1c", l1c, prog_bar=True)

        return loss + l1c * self.hparams.l1_penalty
