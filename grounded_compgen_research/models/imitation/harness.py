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


def imitation_optimizer_config(model, lr):
    optimizer = torch.optim.Adam(model.parameters(), model.hparams.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

    return {"optimizer": optimizer, "scheduler": scheduler}


def imitation_training_step(harness, x, idx):
    seeds, mission, images_path, directions_path, taken_actions, returns, masks = x
    bool_masks = masks.bool()

    past_actions = (taken_actions * (taken_actions != -1))[:, :-1]
    policy_logits, critic_values = harness(
        (mission, images_path, directions_path, past_actions)
    )

    loss = compute_imitation_loss(policy_logits[bool_masks], taken_actions[bool_masks])
    entropy = (
        torch.special.entr(torch.softmax(policy_logits, dim=-1).clamp(10e-7, 1.0))
        .sum(dim=-1)
        .mean()
    )

    harness.log("tloss", loss, prog_bar=True)
    harness.log("tentropy", entropy, prog_bar=True)
    return loss - harness.hparams.entropy_bonus * entropy


def imitation_episodic_success_validation_step(harness, x, idx, dl_idx):
    rewards = x
    success = (rewards > 0).to(torch.float).mean()

    if sys.stdout.isatty():
        tqdm.write(
            "val dl {} batch {} succ {} shape {}".format(
                dl_idx, idx, success.item(), x.shape
            )
        )

    harness.log("vsucc", success, prog_bar=True)


class ImitationLearningHarness(pl.LightningModule):
    """A Lightning harness for imitation learning.

    Override the forward() method"""

    def __init__(
        self,
        lr=10e-4,
        entropy_bonus=10e-3,
        optimizer_config_func=None,
        training_step_func=None,
        validation_step_func=None,
    ):
        super().__init__()
        self.save_hyperparameters("lr", "entropy_bonus")
        self.optimizer_config_func = optimizer_config_func or imitation_optimizer_config
        self.training_step_func = training_step_func or imitation_training_step
        self.validation_step_func = (
            validation_step_func or imitation_episodic_success_validation_step
        )

    def configure_optimizers(self):
        return self.optimizer_config_func(self, self.hparams.lr)

    def training_step(self, x, idx):
        return self.training_step_func(self, x, idx)

    def validation_step(self, x, idx, dl_idx=0):
        return self.validation_step_func(self, x, idx, dl_idx)

    def test_step(self, x, idx, dl_idx=0):
        return self.validation_step_func(self, x, idx, dl_idx)
