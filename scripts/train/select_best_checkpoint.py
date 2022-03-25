import argparse
import os
import sys
import yaml


def get_most_recent_version(experiment_dir):
    versions = os.listdir(os.path.join(experiment_dir, "lightning_logs"))
    return sorted(versions, key=lambda x: int(x.split("_")[1]))[-1]


def generate_experiment_name(exp, model, seed, iterations, batch_size, limit):
    return f"{exp}_s_{seed}_m_{model}_it_{iterations}_b_{batch_size}_l_{limit}"


def best_pair(items):
    return sorted(list(items), key=lambda x: x[1])[-1]


def get_best_model_for_seed(
    models_path, exp, model, seed, iterations, batch_size, limit
):
    exp_full_name = generate_experiment_name(
        exp, model, seed, iterations, batch_size, limit
    )
    model_dir = os.path.join(
        models_path,
        exp,
        model,
        exp_full_name,
    )
    version = get_most_recent_version(model_dir)
    best_k_path = os.path.join(
        model_dir, "lightning_logs", version, "checkpoints", "best_k_models.yaml"
    )

    with open(best_k_path, "r") as f:
        return best_pair(yaml.load(f, yaml.FullLoader).items())


def get_best_model_for_config(
    models_path, exp, model, seeds, iterations, batch_size, limit
):
    seed_best_checkpoints = [
        get_best_model_for_seed(
            models_path, exp, model, seed, iterations, batch_size, limit
        )
        for seed in seeds
    ]

    return best_pair(seed_best_checkpoints)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models_dir")
    parser.add_argument("--exp", type=str, help="Experiment type")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--iterations", type=int, help="Number of iterations used")
    parser.add_argument("--batch-size", type=str, help="Batch size used")
    parser.add_argument("--limit", type=int, help="Limit on training data")
    args = parser.parse_args()

    model_path, score = get_best_model_for_config(
        args.models_dir,
        args.exp,
        args.model,
        range(10),
        args.iterations,
        args.batch_size,
        args.limit,
    )

    print(f"Using {model_path} with score {score}", file=sys.stderr)
    print(model_path)


if __name__ == "__main__":
    main()
