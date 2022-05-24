from typing import Tuple

import torch
import torch.nn.functional as F

from ..interaction.simple_attention import SimpleAttentionModel
from .harness import ImageDiscriminatorHarness


class SimpleAttentionDiscriminatorHarness(ImageDiscriminatorHarness):
    def __init__(self, attrib_offsets, emb_dim, n_words, lr=10e-4, l1_penalty=0):
        super().__init__(attrib_offsets, emb_dim, lr=lr, l1_penalty=l1_penalty)
        self.model = SimpleAttentionModel(attrib_offsets, emb_dim, n_words)
        attrib_ranges = [
            torch.arange(attrib_offsets[i], attrib_offsets[i + 1])
            for i in range(len(attrib_offsets) - 1)
        ]
        self.register_buffer(
            "attrib_ranges_expanded",
            torch.stack(
                [
                    attrib_ranges[i][
                        tuple(
                            ([None] * i)
                            + [slice(None)]
                            + [None] * (len(attrib_ranges) - 1 - i)
                        )
                    ].expand(*[len(r) for r in attrib_ranges])
                    for i in range(len(attrib_ranges))
                ],
                dim=-1,
            ),
        )

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        image, mission = x
        return self.model(image, mission)

    def training_step(self, x, idx):
        loss = super().training_step(x, idx)

        all_attrib_embedding_combos = (
            self.model.attrib_embeddings(self.attrib_ranges_expanded)
            .flatten(start_dim=-2)
            .flatten(0, -2)
        )
        in_projected_embedding_combos = self.model.in_projection(
            all_attrib_embedding_combos
        )

        l1c = (
            (
                F.normalize(in_projected_embedding_combos, dim=-1)
                @ F.normalize(self.model.word_embeddings.weight, dim=-1).T
            )
            .abs()
            .mean()
        )

        self.log("l1c", l1c, prog_bar=True)

        return loss + l1c * self.hparams.l1_penalty
