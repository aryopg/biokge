import argparse
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv("env/.env")

import wandb

from src.configs import Configs
from src.datasets.BioKG import BioKGDataset
from src.evaluator import Evaluator
from src.trainer import Trainer
from src.utils import common_utils
from src.utils.logger import Logger


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Protein Knowledge Graph Embedding Project"
    )
    parser.add_argument("--config_filepath", type=str, required=True)
    parser.add_argument("--log_to_wandb", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = argument_parser()
    configs = Configs(**common_utils.load_yaml(args.config_filepath))

    common_utils.setup_random_seed(configs.training_configs.random_seed)
    outputs_dir, checkpoint_path = common_utils.setup_experiment_folder(
        os.path.join(os.getcwd(), configs.training_configs.outputs_dir)
    )
    device = common_utils.setup_device(configs.training_configs.device)

    wandb.init(
        project="kge_ppa",
        entity="protein-kge",
        mode="online" if args.log_to_wandb else "disabled",
    )
    wandb.config.update(configs.dict())

    if configs.dataset_configs.dataset_name == "dsi-bdi-biokg":
        dataset = BioKGDataset(configs.dataset_configs)
        data_stats = {
            "num_entities": dataset.num_entities,
            "num_relations": dataset.num_relations,
        }
        wandb.config.update(data_stats)

        loggers = {
            "Hits@10_TOTAL": Logger(outputs_dir, "Hits@10_TOTAL"),
            "Hits@50_TOTAL": Logger(outputs_dir, "Hits@50_TOTAL"),
            "Hits@100_TOTAL": Logger(outputs_dir, "Hits@100_TOTAL"),
        }
        trainer = Trainer(
            data_stats["num_entities"],
            data_stats["num_relations"],
            configs,
            outputs_dir,
            checkpoint_path,
            loggers,
            device,
        )

        trainer.train(dataset, Evaluator("dsi-bdi-biokg"))
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
