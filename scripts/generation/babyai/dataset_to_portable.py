import argparse
from babyaiutil.envs.babyai.data import read_pickle, write_hdf5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pickle_file")
    parser.add_argument("hdf5_file")
    args = parser.parse_args()

    (
        padded_train_trajectories,
        padded_valid_trajectories,
        words,
        word2idx,
    ) = read_pickle(args.pickle_file)
    write_hdf5(
        padded_train_trajectories, padded_valid_trajectories, words, args.hdf5_file
    )


if __name__ == "__main__":
    main()
