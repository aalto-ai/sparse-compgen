import os

import torch
import torch.nn as nn

from ..discriminator.transformer import TransformerEncoderDecoderModel
from .harness import ImageSupervisedHarness


def linear_with_warmup_schedule(
    optimizer, num_warmup_steps, num_training_steps, min_lr_scale, last_epoch=-1
):
    min_lr_logscale = min_lr_scale

    def lr_lambda(current_step):
        # Scale from 0 to 1
        if current_step <= num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Scale from 1 to min_lr_scale logarithmically
        #
        # So for example, if min_lr_logscale is -3, then
        # scale goes from 0 to -3 meaning that the lr multiplier
        # goes from 1, to 1e-1 at -1, to 1e-2 at -2 to 1e-3 at -3.
        scale = min(
            1,
            float(current_step - num_warmup_steps)
            / float(num_training_steps - num_warmup_steps),
        )
        logscale = scale * min_lr_logscale
        multiplier = 10**logscale

        return multiplier

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


class TransformerEncoderDecoderProjection(nn.Module):
    def __init__(
        self,
        attrib_offsets,
        emb_dim,
        n_words,
        num_encoder_layers=1,
        num_decoder_layers=4,
    ):
        super().__init__()
        self.model = TransformerEncoderDecoderModel(
            attrib_offsets,
            emb_dim,
            n_words,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )
        self.projection = nn.Linear(emb_dim * 2, 1)

    def forward(self, images, missions):
        (
            out_img,
            image_components,
            decoder_att_weights,
            out_seq,
            self_att_masks,
            mha_masks,
        ) = self.model(images, missions)

        # Note that image_mask re-embeds the image components
        # for its own use.
        projected_image_components = self.projection(out_img).squeeze(-1)

        if os.environ.get("DEBUG", "0") == "1":
            import pdb

            pdb.set_trace()

        return projected_image_components, image_components


class TransformerEncoderDecoderProjectionHarness(ImageSupervisedHarness):
    def __init__(self, attrib_offsets, emb_dim, n_words, lr=10e-4):
        super().__init__(lr=lr)
        self.encoder = TransformerEncoderDecoderProjection(
            attrib_offsets, emb_dim, n_words
        )

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
