from copy import deepcopy
import numpy as np

from train_discriminator import do_experiment, parser


def main():
    p = parser()
    args = p.parse_args()

    limit_npow_2 = np.floor(np.log(args.limit) / np.log(2)).astype(int)
    limits = np.around(
        np.geomspace(2**5, 2**limit_npow_2, num=limit_npow_2 - 4)
    ).astype(int)

    for limit in limits:
        args_copy = deepcopy(args)
        args_copy.limit = int(limit)
        do_experiment(args_copy)


if __name__ == "__main__":
    main()
