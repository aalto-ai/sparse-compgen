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
    return (image.detach().permute(0, 3, 1, 2) * mask).sum(dim=-1).sum(dim=-1)


def mask_components(image_components, mask):
    return [apply_mask_to_image_components(image, mask) for image in image_components]


def match_components_separately(left_components, right_components):
    return torch.clamp(
        (
            F.normalize(torch.stack(left_components, dim=0), dim=-1)
            * F.normalize(torch.stack(right_components, dim=0), dim=-1)
        )
        .sum(dim=-1)
        .relu()
        .prod(dim=0),
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
    def __init__(self, attrib_offsets, emb_dim, out_dim, lr=10e-4, **kwargs):
        super().__init__()
        self.to_mask = ImageComponentsToMask(emb_dim, attrib_offsets, [2, 1])
        self.projection = nn.Linear(out_dim, 1)
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
        output_tgt = self.projection(output_tgt).squeeze(-1)

        masks_src = self.to_mask(image_src[..., :2], direction_src)
        masks_tgt = self.to_mask(image_tgt[..., :2], direction_tgt)

        masked_output_label = (
            masks_tgt.detach().squeeze(1) * label.float()[:, None, None]
        )
        pos_weight = torch.tensor(
            masked_output_label.view(-1).shape[0] / label.view(-1).shape[0]
        )

        loss = F.binary_cross_entropy_with_logits(
            output_tgt, masked_output_label, pos_weight=pos_weight
        )

        # Using mse_loss here instead of BCE loss since doesn't penalize
        # overconfidence on wrong predictions as much (which is important,
        # since we will have some label noise)
        loss_img = F.mse_loss(
            match_components_separately(
                mask_components(image_components_src, masks_src),
                mask_components(image_components_tgt, masks_tgt),
            ),
            label.float(),
        )

        masks = torch.cat([masks_src, masks_tgt], dim=0)
        masks_entropy = (
            torch.distributions.Categorical(
                probs=masks.reshape(-1, masks.shape[-2] * masks.shape[-1])
            )
            .entropy()
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
        output = self.projection(output).squeeze(-1)
        target_long = target.long()
        pos_weight = torch.tensor(
            target_long.view(-1).shape[0] / label.view(-1).shape[0]
        )

        bce_target = F.binary_cross_entropy_with_logits(
            output, target.float(), pos_weight=pos_weight
        )

        if os.environ.get("DEBUG", "0") == "1":
            import pdb

            pdb.set_trace()

        self.log("vtarget", bce_target, prog_bar=True)
        self.log(
            "vsprec", soft_precision(output.sigmoid(), target.float()), prog_bar=True
        )
        self.log(
            "vsrecall", soft_recall(output.sigmoid(), target.float()), prog_bar=True
        )
