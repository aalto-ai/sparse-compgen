import sys

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

import pytorch_lightning as pl

from tqdm import tqdm


def compute_imitation_loss(action_logits, taken_actions):
    action_dist = Categorical(logits=F.log_softmax(action_logits, dim=-1))
    action_neg_logprobs = -action_dist.log_prob(taken_actions)

    return action_neg_logprobs.mean()


def compute_return_weighted_imitation_loss(action_logits, taken_actions, returns):
    action_dist = Categorical(logits=F.log_softmax(action_logits, dim=-1))
    action_neg_logprobs = -action_dist.log_prob(taken_actions)

    return (action_neg_logprobs * returns).mean()


def compute_conservative_policy_loss(policy_logits, taken_actions, returns):
    """Getting the policy right is more important in the high-return space."""
    return (
        F.cross_entropy(policy_logits, taken_actions, reduction="none") * returns
    ).mean()


class ImitationLearningHarness(pl.LightningModule):
    """A Lightning harness for imitation learning.

    Override the forward() method"""

    def __init__(self, lr=10e-4, entropy_bonus=10e-3):
        super().__init__()
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1000, gamma=0.9
        )

        return {"optimizer": optimizer, "scheduler": scheduler}

    def training_step(self, x, idx):
        seeds, mission, images_path, directions_path, taken_actions, returns, masks = x
        bool_masks = masks.bool()

        policy_logits, critic_values = self.forward(
            (mission, images_path, directions_path)
        )

        loss = compute_imitation_loss(
            policy_logits[bool_masks], taken_actions[bool_masks]
        )
        entropy = torch.special.entr(
            torch.softmax(policy_logits, dim=-1).clamp(10e-7, 1.0)
        ).sum(dim=-1).mean()

        self.log("tloss", loss.item(), prog_bar=True)
        self.log("tentropy", entropy.item(), prog_bar=True)
        return loss - self.hparams.entropy_bonus * entropy

    def validation_step(self, x, idx, dl_idx):
        rewards = x
        success = (rewards > 0).to(torch.float).mean()

        if sys.stdout.isatty():
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
