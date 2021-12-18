import os

import torch
import torch.nn.functional as F
import torchmetrics as tm
import pytorch_lightning as pl


def compute_positive_ll(scores):
    return (torch.sigmoid((scores) + 10e-7)).log()


def compute_negative_ll(scores):
    return (1 - torch.sigmoid(scores) + 10e-7).log()


def soft_prec(preds, labels):
    return preds[labels == 1].sum() / preds.sum()


def soft_recall(preds, labels):
    return preds[labels == 1].sum() / labels.sum()


def soft_f1(preds, labels):
    prec = soft_prec(preds, labels)
    recall = soft_recall(preds, labels)

    return (2 * prec * recall) / (prec + recall)


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


class ImageDiscriminatorHarness(pl.LightningModule):
    def __init__(self, lr=10e-4, **kwargs):
        super().__init__()
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def forward(self, x):
        return NotImplementedError()

    def training_step(self, x, idx):
        mission, image_src, direction_src, image_tgt, direction_tgt, label = x
        _, masks_src, image_components_src = self.forward(
            (image_src[..., :2], mission, direction_src)
        )[:3]
        output_tgt, masks_tgt, image_components_tgt = self.forward(
            (image_tgt[..., :2], mission, direction_tgt)
        )[:3]

        loss = F.binary_cross_entropy_with_logits(output_tgt, label.float())
        loss_img = F.binary_cross_entropy(
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
        self.log(
            "tmap", tm.functional.average_precision(output_tgt, label), prog_bar=True
        )
        self.log("tf1", tm.functional.f1(output_tgt, label), prog_bar=True)
        self.log("tsf1", soft_f1(output_tgt.sigmoid(), label), prog_bar=True)
        self.log(
            "pos_ll",
            compute_positive_ll(output_tgt[label.bool()]).mean(),
            prog_bar=True,
        )
        self.log(
            "neg_ll",
            compute_negative_ll(output_tgt[~label.bool()]).mean(),
            prog_bar=True,
        )
        self.log("bce", loss, prog_bar=True)
        self.log("me", masks_entropy, prog_bar=True)

        return loss + loss_img

    def validation_step(self, x, idx):
        mission, image, direction, label, target = x
        output, _, __, predicted_states = self.forward(
            (image[..., :2], mission, direction)
        )[:4]

        loss = F.binary_cross_entropy_with_logits(output, label.float())
        bce_target = F.binary_cross_entropy_with_logits(
            predicted_states, target.float()
        )

        if os.environ.get("DEBUG", "0") == "1":
            import pdb

            pdb.set_trace()

        self.log("vmap", tm.functional.average_precision(output, label), prog_bar=True)
        self.log("vf1", tm.functional.f1(output, label), prog_bar=True)
        self.log("vsf1", soft_f1(output.sigmoid(), label), prog_bar=True)
        self.log(
            "vpos_ll", compute_positive_ll(output[label.bool()]).mean(), prog_bar=True
        )
        self.log(
            "vneg_ll", compute_negative_ll(output[~label.bool()]).mean(), prog_bar=True
        )
        self.log("vbce", loss, prog_bar=True)
        self.log("vtarget", bce_target, prog_bar=True)
        self.log(
            "vtf1",
            tm.functional.average_precision(
                predicted_states.flatten(), target.float().flatten()
            ),
            prog_bar=True,
        )

        return loss
