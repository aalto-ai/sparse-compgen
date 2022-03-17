import torch.optim as optim

from .schedule import linear_with_warmup_schedule


def transformer_optimizer_config(harness, lr):
    optimizer = torch.optim.Adam(harness.parameters(), lr=lr)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": linear_with_warmup_schedule(
                optimizer, 10000, harness.trainer.max_steps, -2
            ),
            "interval": "step",
            "frequency": 1,
        },
    }