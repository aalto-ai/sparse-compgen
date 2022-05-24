import pytorch_lightning as pl


class ScheduleHparamCallback(pl.callbacks.Callback):
    def __init__(self, hparam, start, end, schedule_start, schedule_over):
        super().__init__()
        self.hparam = hparam
        self.start = start
        self.end = end
        self.schedule_start = schedule_start
        self.schedule_over = schedule_over

    def on_train_batch_start(self, trainer, mod, batch, batch_idx, unused=0):
        progress = min(
            max(batch_idx - self.schedule_start, 0) / self.schedule_over, 1.0
        )
        setattr(
            mod.hparams, self.hparam, progress * self.end + (1 - progress) * self.start
        )
