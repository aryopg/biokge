import argparse
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv("env/.env")

from ogb.linkproppred import LinkPropPredDataset

import wandb
from src.configs import Configs
from src.trainer import Trainer
from src.utils import common_utils
from src.utils.logger import Logger


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Protein Knowledge Graph Embedding Project"
    )
    parser.add_argument("--config_filepath", type=str, default="configs/complex.yaml")
    args = parser.parse_args()
    return args


def main():
    args = argument_parser()
    configs = Configs(**common_utils.load_yaml(args.config_filepath))

    common_utils.setup_random_seed(configs.training_configs.random_seed)
    outputs_dir, checkpoint_path = common_utils.setup_experiment_folder(
        configs.training_configs.outputs_dir
    )
    device = common_utils.setup_device(configs.training_configs.device)

    wandb.init(project="kge_ppa", entity="aryopg")
    wandb.config.update(configs.dict())

    if configs.dataset_configs.dataset_name == "ogbl-ppa":
        dataset = LinkPropPredDataset(name="ogbl-ppa")
        split_edge = dataset.get_edge_split()
        data = dataset[0]
        data_stats = {
            "num_entities": data["num_nodes"],
            "num_relations": 1,
        }
        wandb.config.update(data_stats)

        loggers = {
            "Hits@10": Logger(outputs_dir, "Hits@10"),
            "Hits@50": Logger(outputs_dir, "Hits@50"),
            "Hits@100": Logger(outputs_dir, "Hits@100"),
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
        trainer.train(split_edge)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()