import argparse
import os
import sys

import pykeen.constants
import pykeen.pipeline
import pykeen.utils

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv(".env")

from src import utils
from src.configs import Configs


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Protein Knowledge Graph Embedding Project"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(os.getcwd(), "outputs"),
    )
    parser.add_argument("--log_to_wandb", action="store_true")
    args = parser.parse_args()
    return args


def main():

    # Get args and configs
    args = argument_parser()
    configs = Configs(**utils.load_yaml(args.config))

    # Run
    result = pykeen.pipeline.pipeline(
        random_seed=configs.training_config.random_seed,
        # Dataset
        dataset=configs.dataset_config.name,
        # Model
        model=configs.model_config.name,
        # Training
        training_loop=configs.training_config.training_loop,
        epochs=configs.training_config.epochs,
        # Evaluation
        result_tracker="wandb",
        result_tracker_kwargs=dict(
            project="kge_ppa",
            entity="protein-kge",
            mode="online" if args.log_to_wandb else "disabled",
        ),
        stopper="early",
        stopper_kwargs=dict(frequency=1, patience=5, relative_delta=0.002),
    )

    # Save results
    result.save_to_directory(args.output_path)


if __name__ == "__main__":
    main()
