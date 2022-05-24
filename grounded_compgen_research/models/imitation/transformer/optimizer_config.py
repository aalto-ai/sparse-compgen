import torch.optim as optim

from .schedule import linear_with_warmup_schedule


def transformer_optimizer_config(
    harness, lr, warmup_proportion=0.14, decay_power=-2, weight_decay=0
):
    optimizer = optim.AdamW(harness.parameters(), lr=lr, weight_decay=weight_decay)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": linear_with_warmup_schedule(
                optimizer,
                harness.trainer.max_steps * warmup_proportion,
                harness.trainer.max_steps,
                -2,
            ),
            "interval": "step",
            "frequency": 1,
        },
    }
