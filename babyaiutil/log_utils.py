import os
import fnmatch
import itertools

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def get_tb_logs(logger, version=None):
    directory = os.path.join(
        logger.save_dir, logger.name, "version_{}".format(version or logger.version)
    )
    acc = EventAccumulator(
        os.path.join(
            directory, fnmatch.filter(os.listdir(directory), "*events.out*")[0]
        )
    )
    acc.Reload()
    return acc


def tb_events_scalars_to_pd_dataframe(event_acc, events):
    records = list(
        itertools.chain.from_iterable(
            [
                [
                    {"step": e.step, "value": e.value, "event": event}
                    for e in event_acc.Scalars(event)
                ]
                for event in events
            ]
        )
    )

    return pd.DataFrame.from_records(records)


def quantize_df(df, step, step_col="step"):
    df[step_col] = (
        (pd.qcut(df[step_col], step, labels=False) / step) * df[step_col].max()
    ).astype(int)
    return df
