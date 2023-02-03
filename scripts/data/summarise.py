import argparse
import os

import matplotlib
import pykeen.datasets
import pykeen.datasets.analysis


def relation_counts():

    # Get dataset
    dataset = pykeen.datasets.get_dataset(dataset="biokg")

    # Summarise
    dataset.summarize()

    # Plot relation counts
    pykeen.datasets.analysis.get_relation_count_df(dataset).sort_values(
        "count", ascending=False
    ).plot.bar(x="relation_label", y="count", legend=None)
    matplotlib.pyplot.xticks(
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    matplotlib.pyplot.xlabel("Relation")
    matplotlib.pyplot.ylabel("Log Count")
    matplotlib.pyplot.yscale("log")
    matplotlib.pyplot.savefig(
        os.path.join(os.getcwd(), "plots", "relation_counts.pdf"),
        format="pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Protein Knowledge Graph Embedding Project: summarising data"
    )
    parser.add_argument(
        "--type", type=str, default="relation_counts", choices=["relation_counts"]
    )
    args = parser.parse_args()

    if args.type == "relation_counts":
        relation_counts()
