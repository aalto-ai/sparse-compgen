import torch

from ..interaction.transformer import (
    TransformerEncoderDecoderModel,
    linear_with_warmup_schedule,
)
from .harness import ImageDiscriminatorHarness


class TransformerDiscriminatorHarness(ImageDiscriminatorHarness):
    def __init__(self, attrib_offsets, emb_dim, n_words, lr=10e-4):
        super().__init__(attrib_offsets, emb_dim, lr=lr)
        self.encoder = TransformerEncoderDecoderModel(attrib_offsets, emb_dim, n_words)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": linear_with_warmup_schedule(
                    optimizer, 2000, self.trainer.max_steps, -2
                ),
                "interval": "step",
                "frequency": 1,
            },
        }

    def forward(self, x):
        image, mission = x
        return self.encoder(image, mission)
