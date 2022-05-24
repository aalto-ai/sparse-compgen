import argparse
import json

import numpy as np
import torch
from tqdm.auto import tqdm

from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX

from grounded_compgen_research.envs.babyai.data import read_data
from grounded_compgen_research.envs.babyai.render import render_integer_encoded_grid
from grounded_compgen_research.models.interaction.film import FiLMConvEncoder
from grounded_compgen_research.models.interaction.transformer import TransformerEncoderDecoderModel
from grounded_compgen_research.models.interaction.independent_attention import (
    IndependentAttentionModel,
)
from grounded_compgen_research.models.interaction.simple_attention import SimpleAttentionModel

from common import (
    trim_rows_indices_pcts,
    seed_limit_pairs_to_model_paths,
    get_checkpoint_scores_tuples_for_models,
    flatten_tree,
    to_lists,
)


WORDS = sorted(
    "go to the a red blue purple green grey yellow box key ball door [act]".split()
)
DEFAULT_MODELS = ("film", "transformer", "independent_noreg", "simple", "independent")
MODEL_NAME_TO_MODEL_CLASS = {
    "film": FiLMConvEncoder,
    "transformer": TransformerEncoderDecoderModel,
    "independent_noreg": IndependentAttentionModel,
    "simple": SimpleAttentionModel,
    "independent": IndependentAttentionModel,
}
MODEL_NAME_TO_PREFIX = {
    "film": "film_encoder.",
    "transformer": "encoder.",
    "independent_noreg": "model.",
    "simple": "model.",
    "independent": "model.",
}
DEFAULT_LIMITS = [10, 50, 100, 250, 500, 1000, 2500, 5000, 9980]


def translate_state_dict(sd, strip="encoder."):
    return {k[len(strip) :]: v for k, v in sd.items() if k.startswith(strip)}


def predict_from_model_path(
    model_cls, state_dict_key_prefix, model_path, input_frame, input_mission
):
    model = model_cls(
        [0, len(OBJECT_TO_IDX), len(OBJECT_TO_IDX) + len(COLOR_TO_IDX)],
        48,
        len(WORDS),
    )
    model.load_state_dict(
        translate_state_dict(
            torch.load(model_path, map_location=torch.device("cpu")),
            state_dict_key_prefix,
        )
    )

    return (
        model(
            torch.from_numpy(input_frame)[..., :2].long(),
            torch.from_numpy(input_mission).long(),
        )[0][..., 0]
        .sigmoid()
        .detach()
        .numpy()
    )


def predict_evaluations(
    model_cls, state_dict_key_prefix, model_paths, input_frame, input_mission
):
    return np.stack(
        [
            predict_from_model_path(
                model_cls, state_dict_key_prefix, model_path, input_frame, input_mission
            )
            for model_path in model_paths
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("logs_dir")
    parser.add_argument("models_dir")
    parser.add_argument("output_json")
    parser.add_argument("--models", type=str, nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--limits", nargs="*", type=int)
    parser.add_argument(
        "--experiment", type=str, default="discriminator_sample_nogoal_16_logsumexp"
    )
    parser.add_argument("--train-index", type=int, default=0)
    parser.add_argument("--valid-index", type=int, default=6)
    args = parser.parse_args()

    (train_trajectories, valid_trajectories, words, word2idx) = read_data

    scores = {
        m: get_checkpoint_scores_tuples_for_models(
            args.logs_dir,
            args.experiment,
            m,
            200000,
            1024,
            range(10),
            args.limits or DEFAULT_LIMITS,
        )
        for m in tqdm(args.models, desc="Computing scores")
    }
    row_indices = {
        m: trim_rows_indices_pcts(
            scores, limit_lowerbound=1, limit_upperbound=1000, pct=0.75
        )
        for m, scores in tqdm(scores.items(), desc="Computing indices")
    }
    model_paths = {
        m: seed_limit_pairs_to_model_paths(
            args.models_dir,
            args.experiment,
            m,
            200000,
            1024,
            flatten_tree(row_indices[m]),
        )
        for m in args.models
    }

    train_evaluations = {
        m: predict_evaluations(
            MODEL_NAME_TO_MODEL_CLASS[m],
            MODEL_NAME_TO_PREFIX[m],
            paths,
            train_trajectories["image_trajectories"][args.train_index][0][None],
            train_trajectories["missions"][args.train_index][None],
        )[:, 0]
        for m, paths in tqdm(model_paths.items(), desc="Training evaluations")
    }
    valid_evaluations = {
        m: predict_evaluations(
            MODEL_NAME_TO_MODEL_CLASS[m],
            MODEL_NAME_TO_PREFIX[m],
            paths,
            valid_trajectories["image_trajectories"][args.valid_index][0][None],
            valid_trajectories["missions"][args.valid_index][None],
        )[:, 0]
        for m, paths in tqdm(model_paths.items(), desc="Validation evaluations")
    }

    train_env_rendering = render_integer_encoded_grid(
        train_trajectories["image_trajectories"][args.train_index][0],
        train_trajectories["direction_trajectories"][args.train_index][0],
        64,
    )
    train_env_instruction = (
        " ".join([WORDS[w] for w in train_trajectories["missions"][args.train_index]]),
    )
    valid_env_rendering = render_integer_encoded_grid(
        valid_trajectories["image_trajectories"][args.valid_index][0],
        valid_trajectories["direction_trajectories"][args.valid_index][0],
        64,
    )
    valid_env_instruction = (
        " ".join([WORDS[w] for w in valid_trajectories["missions"][args.valid_index]]),
    )

    with open(args.output_json, "w") as f:
        f.write(
            json.dumps(
                to_lists(
                    {
                        "train": {
                            "render": train_env_rendering,
                            "instruction": train_env_instruction,
                            "mean": {
                                m: array.mean(axis=0)
                                for m, array in train_evaluations.items()
                            },
                            "std": {
                                m: array.std(axis=0)
                                for m, array in train_evaluations.items()
                            },
                        },
                        "valid": {
                            "render": valid_env_rendering,
                            "instruction": valid_env_instruction,
                            "mean": {
                                m: array.mean(axis=0)
                                for m, array in valid_evaluations.items()
                            },
                            "std": {
                                m: array.std(axis=0)
                                for m, array in valid_evaluations.items()
                            },
                        },
                    }
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
