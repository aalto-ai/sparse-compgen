import argparse
import pickle

from tqdm.auto import tqdm

from babyaiutil.envs.babyai.generate import (
    generate_seeds_and_action_trajectories,
    replay_actions_for_data_mp,
)
import pickle
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trajectories-per-goal", default=10000, type=int)
    parser.add_argument("--min-trajectory-length", default=7, type=int)
    parser.add_argument("--seeds-and-solutions-buffer", default=None)
    parser.add_argument(
        "--env",
        type=str,
        help="Name of environment to use",
        default="BabyAI-GoToLocal-v0",
    )
    parser.add_argument(
        "--n-goals",
        type=int,
        help="Number of unique goals in this environment",
        default=6 * 3 * 2,
    )
    parser.add_argument("--n-procs", type=int, default=None)
    parser.add_argument("output", help="Where to store the resulting bot trajectories")
    args = parser.parse_args()

    seeds_and_solutions_buffer = (
        args.seeds_and_solutions_buffer or f"seeds_and_solutions_{args.env}.pb"
    )

    (
        seeds_and_solutions,
        unique_environments_counter,
    ) = generate_seeds_and_action_trajectories(
        args.env,
        args.n_trajectories_per_goal,
        6 * 3 * 2,
        args.min_trajectory_length,
    )

    with open(args.seeds_and_solutions_buffer, "wb") as f:
        pickle.dump(seeds_and_solutions, f)

    with open(args.seeds_and_solutions_buffer, "rb") as f:
        seeds_and_solutions = pickle.load(f)

    words = sorted(
        "go to the a red blue purple green grey yellow box key ball door [act]".split()
    )
    word2idx = {w: i for i, w in enumerate(words)}

    all_data = list(
        tqdm(
            replay_actions_for_data_mp(
                args.env, word2idx, seeds_and_solutions, num_procs=args.n_procs
            ),
            total=len(seeds_and_solutions),
        )
    )

    with open(args.output, "wb") as f:
        print(word2idx)
        pickle.dump((all_data, word2idx, words), f)


if __name__ == "__main__":
    main()
