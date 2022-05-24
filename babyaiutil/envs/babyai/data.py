import os
import pickle

import h5py
from tqdm.auto import tqdm


def write_pickle(
    padded_train_trajectories,
    padded_valid_trajectories,
    words,
    word2idx,
    pickle_filename,
):
    with open(pickle_filename, "wb") as f:
        pickle.dump(
            (padded_train_trajectories, padded_valid_trajectories, words, word2idx),
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def read_pickle(pickle_filename):
    with open(pickle_filename, "rb") as f:
        return pickle.load(f)


def write_hdf5(
    padded_train_trajectories, padded_valid_trajectories, words, hdf5_filename
):
    with open(hdf5_filename, "wb") as f:
        with h5py.File(f, "w") as h5f:
            train_group = h5f.create_group("train")
            valid_group = h5f.create_group("valid")

            for key, data in tqdm(
                padded_train_trajectories.items(), desc="Writing training data"
            ):
                train_group.create_dataset(key, data=data, compression="gzip")

            for key, data in tqdm(
                padded_valid_trajectories.items(), desc="Writing validation data"
            ):
                valid_group.create_dataset(key, data=data, compression="gzip")

            h5f.attrs["words"] = " ".join(words)


def read_hdf5(hdf5_filename):
    with open(hdf5_filename, "rb") as f:
        with h5py.File(f, "r") as h5f:
            words = h5f.attrs["words"].split()
            word2idx = {w: i for i, w in enumerate(words)}

            # We read the whole thing into memory in its uncompressed
            # form, which is expensive upfront and also in terms of memory
            # usage, but works better for random access
            return (
                {k: v[:] for k, v in h5f["train"].items()},
                {k: v[:] for k, v in h5f["valid"].items()},
                words,
                word2idx,
            )


def read_data(filename):
    _, ext = os.path.splitext(filename)

    if ext in (".pt", ".pb"):
        return read_pickle(filename)

    if ext == ".h5":
        return read_hdf5

    raise RuntimeError(f"Don't know how to read extension {ext}")
