import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


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


def plot_qualitative(
    rendered_training_input,
    rendered_training_label,
    rendered_validation_input,
    rendered_validation_label,
    film_training,
    film_validation,
    film_validation_std,
    transformer_training,
    transformer_validation,
    transformer_validation_std,
    independent_noreg_training,
    independent_noreg_validation,
    independent_noreg_validation_std,
    simple_attention_training,
    simple_attention_validation,
    simple_attention_validation_std,
    independent_training,
    independent_validation,
    independent_validation_std,
    threshold=0.1,
):
    fig, ax = plt.subplots(3, 6, figsize=(22.5, 11.5))
    ax[0, 0].imshow(rendered_training_input)
    ax[0, 0].set_yticklabels([])
    ax[0, 0].set_xticklabels([])
    ax[0, 0].set_ylabel(
        "$\\mathcal{{D}}_{{\\textrm{{v\\_ID}}}}$\n{}".format(rendered_training_label),
        multialignment="center",
    )
    ax[0, 1].imshow(film_training)
    plt_normalized_heatmap_with_numbers(ax[0, 1], film_training, threshold=threshold)
    ax[0, 1].axis("off")
    ax[0, 1].set_xticklabels([])
    ax[0, 1].set_yticklabels([])
    plt_normalized_heatmap_with_numbers(
        ax[0, 2], transformer_training, threshold=threshold
    )
    ax[0, 2].axis("off")
    ax[0, 2].set_xticklabels([])
    ax[0, 2].set_yticklabels([])
    plt_normalized_heatmap_with_numbers(
        ax[0, 3], independent_noreg_training, threshold=threshold
    )
    ax[0, 3].axis("off")
    ax[0, 3].set_xticklabels([])
    ax[0, 3].set_yticklabels([])
    plt_normalized_heatmap_with_numbers(
        ax[0, 4], simple_attention_training, threshold=threshold
    )
    ax[0, 4].axis("off")
    ax[0, 4].set_xticklabels([])
    ax[0, 4].set_yticklabels([])
    plt_normalized_heatmap_with_numbers(
        ax[0, 5], independent_training, threshold=threshold
    )
    ax[0, 5].axis("off")
    ax[0, 5].set_xticklabels([])
    ax[0, 5].set_yticklabels([])

    ax[1, 0].imshow(rendered_validation_input)
    ax[1, 0].set_yticklabels([])
    ax[1, 0].set_xticklabels([])
    ax[1, 0].set_ylabel(
        "$\\mathcal{{D}}_{{\\textrm{{v\\_OOD}}}}$\n{}".format(
            rendered_validation_label
        ),
        multialignment="center",
    )
    plt_normalized_heatmap_with_numbers(ax[1, 1], film_validation, threshold=threshold)
    ax[1, 1].set_yticklabels([])
    ax[1, 1].set_xticklabels([])
    ax[1, 1].axis("off")
    plt_normalized_heatmap_with_numbers(
        ax[1, 2], transformer_validation, threshold=threshold
    )
    ax[1, 2].set_yticklabels([])
    ax[1, 2].set_xticklabels([])
    ax[1, 2].axis("off")
    plt_normalized_heatmap_with_numbers(
        ax[1, 3], independent_noreg_validation, threshold=threshold
    )
    ax[1, 3].set_yticklabels([])
    ax[1, 3].set_xticklabels([])
    ax[1, 3].axis("off")
    plt_normalized_heatmap_with_numbers(
        ax[1, 4], simple_attention_validation, threshold=threshold
    )
    ax[1, 4].set_yticklabels([])
    ax[1, 4].set_xticklabels([])
    ax[1, 4].axis("off")
    plt_normalized_heatmap_with_numbers(
        ax[1, 5], independent_validation, threshold=threshold
    )
    ax[1, 5].set_yticklabels([])
    ax[1, 5].set_xticklabels([])
    ax[1, 5].axis("off")

    ax[2, 0].imshow(rendered_validation_input)
    ax[2, 0].set_yticklabels([])
    ax[2, 0].set_xticklabels([])
    ax[2, 0].set_ylabel(
        "$\\mathcal{{D}}_{{\\textrm{{v\\_OOD}}}}$ std.\n{}".format(
            rendered_validation_label
        ),
        multialignment="center",
    )
    plt_normalized_heatmap_with_numbers(
        ax[2, 1], film_validation_std, threshold=threshold, cmap="inferno"
    )
    ax[2, 1].set_yticklabels([])
    ax[2, 1].set_xticklabels([])
    ax[2, 1].set_xlabel("FiLM")
    plt_normalized_heatmap_with_numbers(
        ax[2, 2], transformer_validation_std, threshold=threshold, cmap="inferno"
    )
    ax[2, 2].set_yticklabels([])
    ax[2, 2].set_xticklabels([])
    ax[2, 2].set_xlabel("Transformer")
    plt_normalized_heatmap_with_numbers(
        ax[2, 3], independent_noreg_validation_std, threshold=threshold, cmap="inferno"
    )
    ax[2, 3].set_yticklabels([])
    ax[2, 3].set_xticklabels([])
    ax[2, 3].set_xlabel("Factored Attention")
    plt_normalized_heatmap_with_numbers(
        ax[2, 4], simple_attention_validation_std, threshold=threshold, cmap="inferno"
    )
    ax[2, 4].set_yticklabels([])
    ax[2, 4].set_xticklabels([])
    ax[2, 4].set_xlabel("Sparse Attention")
    plt_normalized_heatmap_with_numbers(
        ax[2, 5], independent_validation_std, threshold=threshold, cmap="inferno"
    )
    ax[2, 5].set_yticklabels([])
    ax[2, 5].set_xticklabels([])
    ax[2, 5].set_xlabel("Sparse Factored Attention")

    for a in ax.ravel():
        a.set_anchor("N")
        for item in [a.title, a.xaxis.label, a.yaxis.label]:
            item.set_fontsize(18)

    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    return fig


def evaluate_and_plot_qualitative(
    training_plot_data, validation_plot_data, savefig=None, threshold=0.1
):
    fig = plot_qualitative(
        np.array(training_plot_data["render"], dtype=int),
        training_plot_data["instruction"],
        np.array(validation_plot_data["render"], dtype=int),
        validation_plot_data["instruction"],
        np.array(training_plot_data["mean"]["film"]),
        np.array(validation_plot_data["mean"]["film"]),
        np.array(validation_plot_data["std"]["film"]),
        np.array(training_plot_data["mean"]["transformer"]),
        np.array(validation_plot_data["mean"]["transformer"]),
        np.array(validation_plot_data["std"]["transformer"]),
        np.array(training_plot_data["mean"]["independent_noreg"]),
        np.array(validation_plot_data["mean"]["independent_noreg"]),
        np.array(validation_plot_data["std"]["independent_noreg"]),
        np.array(training_plot_data["mean"]["simple"]),
        np.array(validation_plot_data["mean"]["simple"]),
        np.array(validation_plot_data["std"]["simple"]),
        np.array(training_plot_data["mean"]["independent"]),
        np.array(validation_plot_data["mean"]["independent"]),
        np.array(validation_plot_data["std"]["independent"]),
        threshold=threshold,
    )

    if savefig:
        fig.savefig(savefig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json")
    parser.add_argument("output_pdf")
    args = parser.parse_args()

    plt.rcParams["text.usetex"] = True

    with open(args.input_json, "r") as f:
        plot_data = json.load(f)

    evaluate_and_plot_qualitative(
        plot_data["train"], plot_data["valid"], savefig=args.output_pdf
    )


if __name__ == "__main__":
    main()
