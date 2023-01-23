import os
import sys
import argparse

sys.path.append(os.getcwd())

from src.datasets.BioKG import BioKGDataset
from src.configs import Configs
from src.utils import common_utils


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Protein Knowledge Graph Embedding Project: plotting data statistics"
    )
    parser.add_argument("--config_filepath", type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = argument_parser()
    configs = Configs(**common_utils.load_yaml(args.config_filepath))
    dataset = BioKGDataset(configs.dataset_configs)

    # Make plots dir if needed
    plots_dir = os.path.join(os.getcwd(), "plots")
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)

    # Plot statistics
    dataset.plot_statistics(os.path.join(plots_dir, "edge_distribution.pdf"))


if __name__ == "__main__":
    main()
