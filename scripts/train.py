import argparse
import os
import sys

import pykeen.constants
import pykeen.pipeline
import pykeen.utils
import wandb

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
        model_kwargs=dict(embedding_dim=configs.model_config.embedding_dim),
        # Training
        training_loop=configs.training_config.training_loop,
        epochs=configs.training_config.epochs,
        optimizer_kwargs=dict(lr=configs.training_config.learning_rate),
        negative_sampler_kwargs=dict(
            num_negs_per_pos=configs.training_config.num_negs_per_pos
        ),
        # Evaluation
        evaluator_kwargs=dict(filtered=True),  # filter false negatives
        use_testing_data=False,
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
    result.save_to_directory(os.path.join(args.output_path), wandb.run.name)


if __name__ == "__main__":
    main()
