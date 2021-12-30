import os

import torch
import torch.nn.functional as F
import torchmetrics as tm
import pytorch_lightning as pl


def compute_positive_ll(scores):
    return (torch.sigmoid((scores) + 10e-7)).log()


def compute_negative_ll(scores):
    return (1 - torch.sigmoid(scores) + 10e-7).log()


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


class ImageSupervisedHarness(pl.LightningModule):
    def __init__(self, lr=10e-4, **kwargs):
        super().__init__()
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def forward(self, x):
        return NotImplementedError()

    def training_step(self, x, idx):
        image, mission, target = x
        target = target.long()
        output, _ = self.forward((image[..., :2], mission))

        # Ratio of positives to negatives
        pos_weight = (1 - target).sum() / target.sum()
        loss = F.binary_cross_entropy_with_logits(
            output, target.float(), pos_weight=pos_weight
        )

        if os.environ.get("DEBUG", "0") == "1":
            import pdb

            pdb.set_trace()

        self.log("bce", loss, prog_bar=True)
        self.log(
            "tsprec", soft_precision(output.sigmoid(), target.float()), prog_bar=True
        )
        self.log(
            "tsrecall", soft_recall(output.sigmoid(), target.float()), prog_bar=True
        )

        return loss

    def validation_step(self, x, idx, dataset_idx):
        image, mission, target = x
        target = target.long()
        output, _ = self.forward((image[..., :2], mission))

        # Ratio of positives to negatives
        pos_weight = (1 - target).sum() / target.sum()
        loss = F.binary_cross_entropy_with_logits(
            output, target.float(), pos_weight=pos_weight
        )

        if os.environ.get("DEBUG", "0") == "1":
            import pdb

            pdb.set_trace()

        self.log("vtarget", loss, prog_bar=True)
        self.log(
            "vsprec", soft_precision(output.sigmoid(), target.float()), prog_bar=True
        )
        self.log(
            "vsrecall", soft_recall(output.sigmoid(), target.float()), prog_bar=True
        )
