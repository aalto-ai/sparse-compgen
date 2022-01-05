import os

import torch
import torch.nn.functional as F
import torchmetrics as tm
import pytorch_lightning as pl

from tqdm.auto import tqdm

from .mvprop import MVProp2D
from .resnet_value import (
    BigConvolutionValueMapExtractor,
    ConvolutionMaskValueMapExtractor,
)


class VINHarness(pl.LightningModule):
    def __init__(
        self,
        interaction_module,
        offsets,
        emb_dim=32,
        k=10,
        inv_weight=10e-3,
        lr=1e-3,
        gamma=0.8,
    ):
        super().__init__()
        self.interaction_module = interaction_module
        self.mvprop = MVProp2D(emb_dim * (len(offsets) - 1), n_iterations=k)
        self.value = BigConvolutionValueMapExtractor(
            emb_dim, (len(offsets) - 1), emb_dim, 7
        )
        self.save_hyperparameters()

    def value_map_from_image(self, x):
        images, directions, missions = x

        scores, image_components = self.interaction_module(images[..., :2], missions)[
            :2
        ]
        binary_scores = scores.sigmoid()
        cat_image_components = torch.cat(image_components, dim=-1)

        B, L, H, W, C = cat_image_components.shape

        trajectory_flat_image_components = cat_image_components.reshape(
            -1, H, W, cat_image_components.shape[-1]
        )
        flat_matches = binary_scores.reshape(-1, H, W, 1)

        gather_maps, value_maps, reward_maps = self.mvprop(
            trajectory_flat_image_components, flat_matches
        )

        return cat_image_components, gather_maps, value_maps, reward_maps

    def forward(self, x):
        missions, images, directions = x

        # Something is a goal state if the product of marginals
        # is nonzero
        #
        # (B x L) x H x W
        (
            cat_image_components,
            gather_maps,
            value_maps,
            reward_maps,
        ) = self.value_map_from_image((images[..., :2], directions, missions))

        B, L, H, W, C = cat_image_components.shape
        trajectory_flat_image_components = cat_image_components.reshape(-1, H, W, C)

        values = self.value(
            directions.flatten(),
            trajectory_flat_image_components,
            value_maps,
            reward_maps,
        )

        return (
            torch.log_softmax(values.reshape(B, L, values.shape[-1]), dim=-1),
            values.reshape(B, L, values.shape[-1]),
            cat_image_components,
            gather_maps.reshape(B, L, H, W, 1),
            value_maps.reshape(B, L, H, W, 1),
            reward_maps.reshape(B, L, H, W, 1),
        )

    def training_step(self, x, step_index):
        (
            seed,
            missions,
            images_path,
            directions,
            actions,
            returns,
            masks,
        ) = x

        (
            policy_logits,
            qvalues,
            image_encodings,
            gather_maps,
            value_maps,
            reward_maps,
        ) = self((missions, images_path, directions))
        estimated_q = qvalues[masks.bool()]
        taken_actions = actions[masks.bool()]
        observed_returns = returns[masks.bool()].float()

        action_mask = torch.scatter(
            torch.zeros_like(estimated_q).long(),
            dim=-1,
            index=taken_actions.unsqueeze(-1),
            src=torch.ones_like(taken_actions.unsqueeze(-1)),
        )
        q_mse = F.mse_loss(estimated_q[action_mask.bool()], observed_returns.float())
        q_reg = F.mse_loss(
            estimated_q[~action_mask.bool()],
            torch.zeros_like(estimated_q[~action_mask.bool()]),
        )
        v_reg = F.mse_loss(value_maps, torch.zeros_like(value_maps))

        if os.environ.get("DEBUG", "0") != "0":
            import pdb

            pdb.set_trace()

        # l1_value_penalty = self.value.linear.weight.abs().mean()

        loss = q_mse + self.hparams.inv_weight * (q_reg + v_reg)
        self.log("path", q_mse, prog_bar=True)
        self.log("reg", q_reg, prog_bar=True)
        self.log("vreg", v_reg, prog_bar=True)
        # self.log("l1", l1_value_penalty, prog_bar=True)

        if os.environ.get("DEBUG", "0") == "1":
            import pdb

            pdb.set_trace()

        return loss

    def validation_step(self, x, idx, dl_idx):
        rewards = x
        success = (rewards > 0).to(torch.float).mean()

        tqdm.write(
            "val dl {} batch {} succ {} shape {}".format(
                dl_idx, idx, success.item(), x.shape
            )
        )

        self.log("vsucc", success.item(), prog_bar=True)

    def test_step(self, x, idx):
        rewards = x
        success = (rewards > 0).to(torch.float).mean()

        self.log("tsucc", success.item(), prog_bar=True)

    """
    def validation_step(self, x, step_index):
        (
            images_path,
            directions,
            actions,
            returns,
            masks,
            lengths,
            missions,
            mission_masks,
        ) = x

        image_encodings, gather_maps, value_maps, reward_maps, qvalues = self(
            (images_path, directions, missions)
        )
        estimated_q = qvalues[masks.bool()]
        taken_actions = actions[masks.bool()]
        observed_returns = returns[masks.bool()].float()

        action_mask = torch.scatter(
            torch.zeros_like(estimated_q).long(),
            dim=-1,
            index=taken_actions.unsqueeze(-1),
            src=torch.ones_like(taken_actions.unsqueeze(-1)),
        )
        q_mse = F.mse_loss(estimated_q[action_mask.bool()], observed_returns.float())

        # import pdb
        # pdb.set_trace()
        self.log("vpath", q_mse, prog_bar=True)

        if os.environ.get("DEBUG", "0") == "1":
            import pdb

            pdb.set_trace()
    """

    def configure_optimizers(self):
        import itertools

        return torch.optim.Adam(
            itertools.chain.from_iterable(
                [self.mvprop.parameters(), self.value.parameters()]
            )
        )