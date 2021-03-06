import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from ..img_mask import ImageComponentsToMask


def compute_positive_ll(scores):
    return (torch.sigmoid((scores) + 10e-7)).log()


def compute_negative_ll(scores):
    return (1 - torch.sigmoid(scores) + 10e-7).log()


def apply_mask_to_image_components(image, mask):
    return (
        (image.detach().transpose(-1, -2).transpose(-2, -3) * mask)
        .sum(dim=-1)
        .sum(dim=-1)
    )


def mask_components(image_components, mask):
    # Add component dimension to mask, since image_components has
    # a component dimension
    return apply_mask_to_image_components(image_components, mask[..., None, :, :, :])


def match_components_separately(left_components, right_components):
    return torch.clamp(
        (F.normalize(left_components, dim=-1) * F.normalize(right_components, dim=-1))
        # B x C x E => B x C
        .sum(dim=-1).relu()
        # B x C => B
        .prod(dim=-1),
        0,
        1,
    )


def soft_precision(predictions, targets):
    false_positive_mass = predictions * (1 - targets)
    true_positive_mass = predictions * targets

    precisions = true_positive_mass.view(true_positive_mass.shape[0], -1).sum(
        dim=-1
    ) / (
        true_positive_mass.view(true_positive_mass.shape[0], -1).sum(dim=-1)
        + false_positive_mass.view(false_positive_mass.shape[0], -1).sum(dim=-1)
    )

    return precisions.mean()


def soft_recall(predictions, targets):
    true_positives_mass = predictions * targets
    true_positives_mass = true_positives_mass.view(
        true_positives_mass.shape[0], -1
    ).sum(dim=-1)
    true_positives_plus_false_negatives_mass = targets.view(targets.shape[0], -1).sum(
        dim=-1
    )
    recalls = true_positives_mass / true_positives_plus_false_negatives_mass

    return recalls.mean()


class ImageDiscriminatorHarness(pl.LightningModule):
    def __init__(self, attrib_offsets, emb_dim, lr=10e-4, **kwargs):
        super().__init__()
        self.to_mask = ImageComponentsToMask(emb_dim, attrib_offsets, [2, 1])
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def forward(self, x):
        return NotImplementedError()

    def training_step(self, x, idx):
        mission, image_src, direction_src, image_tgt, direction_tgt, label = x
        _, image_components_src = self.forward((image_src[..., :2], mission))[:2]
        output_tgt, image_components_tgt = self.forward((image_tgt[..., :2], mission))[
            :2
        ]
        output_tgt = output_tgt.squeeze(-1)

        masks_src = self.to_mask(image_src[..., :2], direction_src)
        masks_tgt = self.to_mask(image_tgt[..., :2], direction_tgt)

        # Just add the log-masks, the output is already log-domain
        masked_tgt = (masks_tgt.detach().squeeze(-3) + 10e-7).log() + output_tgt

        # Need to take the exp and sum in the non-log domain for spatial-summing
        # to make any sense (we're summing over the product, or the log-sum).
        sum_tgt = masked_tgt.exp().sum(dim=-1).sum(dim=-1).log()

        loss = F.binary_cross_entropy_with_logits(sum_tgt, label.to(masks_tgt.dtype))

        # Using mse_loss here instead of BCE loss since doesn't penalize
        # overconfidence on wrong predictions as much (which is important,
        # since we will have some label noise)
        components_match = match_components_separately(
            mask_components(image_components_src.to(masks_src.dtype), masks_src),
            mask_components(image_components_tgt.to(masks_tgt.dtype), masks_tgt),
        )
        loss_img = F.mse_loss(
            components_match,
            label.to(components_match.dtype),
        )

        masks = torch.cat([masks_src, masks_tgt], dim=0)
        masks_entropy = (
            torch.special.entr(masks.flatten(0, -2).flatten(0, -2).clamp(10e-7, 1.0))
            .sum(dim=-1)
            .mean()
        )

        if os.environ.get("DEBUG", "0") == "1":
            import pdb

            pdb.set_trace()

        self.log("timg", loss_img, prog_bar=True)
        self.log("bce", loss, prog_bar=True)
        self.log("me", masks_entropy, prog_bar=True)

        return loss + loss_img

    def validation_step(self, x, idx, dataloader_idx):
        mission, image, direction, label, target = x
        output, image_components = self.forward((image[..., :2], mission))[:2]
        output = output.squeeze(-1)

        target_long = target.long()
        pos_weight = torch.tensor(
            target_long.view(-1).shape[0] / label.view(-1).shape[0]
        )

        target_output_type = target.to(output.dtype)

        bce_target = F.binary_cross_entropy_with_logits(
            output, target_output_type, pos_weight=pos_weight
        )

        if os.environ.get("DEBUG", "0") == "1":
            import pdb

            pdb.set_trace()

        precision = soft_precision(output.sigmoid(), target_output_type)
        recall = soft_recall(output.sigmoid(), target_output_type)
        f1 = 2 * (precision * recall) / (precision + recall)

        self.log("vtarget", bce_target, prog_bar=True)
        self.log("vsprec", precision, prog_bar=True)
        self.log("vsrecall", recall, prog_bar=True)
        self.log("vsf1", f1, prog_bar=True)
