import argparse
import itertools
import json

import numpy as np
import matplotlib.pyplot as plt

from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX

WORDS = sorted(
    "go to the a red blue purple green grey yellow box key ball door [act]".split()
)


def plt_normalized_heatmap_with_numbers(ax, matrix, cmap=None, threshold=0.01, vmax=1):
    table_data = np.round(matrix, decimals=2)

    ax.imshow(table_data, cmap=cmap or "viridis", vmin=0, vmax=vmax, aspect="auto")

    for i in range(table_data.shape[0]):
        for j in range(table_data.shape[1]):
            if np.abs(table_data[i, j]) > threshold:
                text = ax.text(
                    j,
                    i,
                    str(table_data[i, j])[:4],
                    ha="center",
                    va="center",
                    color="w" if (table_data[i, j] < 0.8) else "blue",
                )


def plot_embedding_correlation_matrix_to_ax(ax, matrix, cmap=None, threshold=0.01):
    plt_normalized_heatmap_with_numbers(ax, matrix, cmap=cmap, threshold=threshold)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(
        list(OBJECT_TO_IDX.keys()) + list(COLOR_TO_IDX.keys()), rotation=40, fontsize=16
    )
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(WORDS, fontsize=16)


def plot_projected_embedding_correlation_matrix_to_ax(
    ax, matrix, cmap=None, threshold=None
):
    # Need to figure out to label these
    plt_normalized_heatmap_with_numbers(
        ax, matrix.T[5 * 6 : 8 * 6].T, cmap=cmap, threshold=threshold
    )
    ax.set_yticks(np.arange(len(WORDS)))
    ax.set_yticklabels(WORDS, fontsize=16)
    ax.set_xticklabels(
        [
            " ".join(
                [
                    list(COLOR_TO_IDX.keys())[j - len(OBJECT_TO_IDX)],
                    list(OBJECT_TO_IDX.keys())[i],
                ]
            )
            for i, j in list(
                itertools.product(
                    range(0, len(OBJECT_TO_IDX.keys())),
                    range(
                        len(OBJECT_TO_IDX.keys()),
                        len(COLOR_TO_IDX.keys()) + len(OBJECT_TO_IDX.keys()),
                    ),
                )
            )[5 * 6 : 8 * 6]
        ],
        rotation=40,
        ha="right",
        fontsize=16,
    )
    ax.set_xticks(range(3 * 6))


def plot_projected_embedding_correlation_matrix(
    matrix, save_filename=None, cmap=None, threshold=None
):
    # Need to figure out to label these
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plot_projected_embedding_correlation_matrix_to_ax(
        ax, matrix, cmap=cmap, threshold=threshold
    )

    if save_filename:
        fig.savefig(save_filename)


def plot_correlations_3x1(
    simple_mean_correlations,
    independent_noreg_mean_correlations,
    independent_correlations,
    threshold=0.03,
    savefig=None,
    **rc_params
):
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))

    plot_projected_embedding_correlation_matrix_to_ax(
        ax[0], simple_mean_correlations, cmap="viridis", threshold=threshold
    )
    ax[0].get_yaxis().set_visible(True)
    ax[0].set_ylabel("word")
    ax[0].set_xlabel("Sparse Attention")

    plot_embedding_correlation_matrix_to_ax(
        ax[1], independent_noreg_mean_correlations, cmap="viridis", threshold=threshold
    )
    ax[1].get_yaxis().set_visible(False)
    ax[1].set_ylabel("word")
    ax[1].set_xlabel("Factored Attention")

    plot_embedding_correlation_matrix_to_ax(
        ax[2], independent_correlations, cmap="viridis", threshold=threshold
    )
    ax[2].get_yaxis().set_visible(False)
    ax[2].set_xlabel("Sparse Factored Attention")

    for a in ax.ravel():
        a.set_anchor("N")
        for item in [a.title, a.xaxis.label, a.yaxis.label]:
            item.set_fontsize(28)

    fig.subplots_adjust(wspace=0.025, hspace=0.1)
    fig.align_xlabels()

    if savefig:
        fig.savefig(savefig, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json")
    parser.add_argument("output_pdf")
    args = parser.parse_args()

    with open(args.input_json) as f:
        plot_data = json.load(f)

    plot_correlations_3x1(
        np.array(plot_data["means"]["simple"]),
        np.array(plot_data["means"]["independent_noreg"]),
        np.array(plot_data["means"]["independent"]),
        savefig=args.output_pdf,
    )


if __name__ == "__main__":
    main()
