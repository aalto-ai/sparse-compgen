import argparse
import os
import json
from unittest.mock import DEFAULT

import numpy as np
import matplotlib.pyplot as plt
from rliable import plot_utils as rl_plot_utils


EXPERIMENT_PRETTY_NAMES = {
    "vin_sample_nogoal_16_logsumexp_end_to_end_sparse:mvprop": "End-to-End Factored/MVProp",
    "vin_sample_nogoal_16_logsumexp_ignorance:mvprop": "Factored/MVProp Planner (ours)",
    "vin_sample_nogoal_16_logsumexp_ignorance_transformer:mvprop": "Transformer/MVProp Planner",
    "vin_sample_nogoal_16_logsumexp_k_0:mvprop": "Factored/CNN Policy",
    "imitation:pure_transformer": "Encoder-Decoder Transformer",
    "imitation:fused_inputs_next_step_encoder": "Encoder-only Transformer",
    "imitation:film_lstm_policy": "GRU-Encoder ResNet/FiLM Decoder",
}
DEFAULT_LIMITS = [50, 100, 250, 500, 1000, 2500, 5000, 9980]


OOD_ANNOTATIONS = (
    ("vin_sample_nogoal_16_logsumexp_ignorance:mvprop", -3, 0.03),
    ("vin_sample_nogoal_16_logsumexp_ignorance:mvprop", -4, 0.03),
    ("vin_sample_nogoal_16_logsumexp_ignorance:mvprop", -5, 0.03),
    ("vin_sample_nogoal_16_logsumexp_ignorance:mvprop", 0, 0.03),
    ("vin_sample_nogoal_16_logsumexp_ignorance:mvprop", 1, 0.03),
    ("vin_sample_nogoal_16_logsumexp_ignorance:mvprop", 2, 0.03),
    ("imitation:film_lstm_policy", -1, 0.03),
    ("imitation:film_lstm_policy", -2, 0.03),
    ("imitation:film_lstm_policy", -3, 0.03),
    ("vin_sample_nogoal_16_logsumexp_k_0:mvprop", -1, 0.03),
    ("vin_sample_nogoal_16_logsumexp_k_0:mvprop", -2, 0.03),
    ("vin_sample_nogoal_16_logsumexp_k_0:mvprop", -3, 0.03),
    ("imitation:pure_transformer", -1, -0.05),
    ("imitation:pure_transformer", -2, -0.05),
    ("imitation:pure_transformer", -3, -0.05),
)

ID_ANNOTATIONS = (
    ("vin_sample_nogoal_16_logsumexp_ignorance:mvprop", 0, 0.03),
    ("vin_sample_nogoal_16_logsumexp_ignorance:mvprop", 1, 0.03),
    ("vin_sample_nogoal_16_logsumexp_ignorance:mvprop", 2, 0.03),
    ("imitation:pure_transformer", -1, -0.05),
    ("imitation:pure_transformer", -2, -0.05),
    ("imitation:pure_transformer", -3, -0.05),
)


def add_annotations(ax, annotations, limits, means):
    for tag, offset, yoffset in annotations:
        real = EXPERIMENT_PRETTY_NAMES.get(tag, tag)
        if real in means:
            ax.text(
                limits[offset],
                means[real][offset] + yoffset,
                f"{means[real][offset]:.2}"[1:],
            )


def translate_dict_keys(dictionary, translation):
    return {translation.get(k, k): v for k, v in dictionary.items()}


def plot_to_file(
    aggregate_iqms, aggregate_cis, limits, annotations, filename, title, ylabel=None
):
    ax = rl_plot_utils.plot_sample_efficiency_curve(
        limits,
        translate_dict_keys(aggregate_iqms, EXPERIMENT_PRETTY_NAMES),
        translate_dict_keys(aggregate_cis, EXPERIMENT_PRETTY_NAMES),
        algorithms=None,
        xlabel="Samples per goal",
        ylabel=ylabel or "Success rate",
    )

    for line, marker in zip(ax.lines, ("o", "P", "s", "^", "p", "*", "x")):
        line.set_marker(marker)

    ax.set_xscale("log")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.1))
    ax.set_ylim((-0.05, 1.05))
    ax.spines["left"].set_position(("outward", 0))
    ax.spines["bottom"].set_position(("outward", 0))

    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(12)

    ax.set_title(title)
    add_annotations(ax, annotations, limits, aggregate_iqms)

    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    ax.get_figure().savefig(filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plot_data", type=str)
    parser.add_argument("output_id_pdf", type=str)
    parser.add_argument("output_ood_pdf", type=str)
    parser.add_argument("--ylabel", type=str)
    parser.add_argument("--limits", nargs="*", type=int)
    args = parser.parse_args()
    plt.rcParams["text.usetex"] = True

    with open(args.plot_data) as f:
        plot_data = json.load(f)

    plot_to_file(
        plot_data["ood"]["means"],
        plot_data["ood"]["cis"],
        args.limits or DEFAULT_LIMITS,
        OOD_ANNOTATIONS,
        args.output_ood_pdf,
        "$\\mathcal{D}_{\\textrm{v\\_OOD}}$\n",
    )

    plot_to_file(
        plot_data["id"]["means"],
        plot_data["id"]["cis"],
        args.limits or DEFAULT_LIMITS,
        ID_ANNOTATIONS,
        args.output_id_pdf,
        "$\\mathcal{D}_{\\textrm{v\\_ID}}$\n",
    )


if __name__ == "__main__":
    main()
