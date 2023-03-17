import argparse
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv("env/.env")

import numpy as np
import wandb

from src.configs import Configs
from src.datasets.benchmark import BenchmarkLibKGEDataset
from src.trainer import Trainer
from src.utils import common_utils

BENCHMARKING_TASKS = [
    "ddi_efficacy",
    "ddi_minerals",
    "dep_fda_exp",
    "dpi_fda",
]


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Protein Knowledge Graph Embedding Project: benchmark relation classification"
    )
    parser.add_argument("--config_filepath", type=str, required=True)
    parser.add_argument(
        "--negative_samples_ratio", type=float, required=True, default=1.0
    )
    parser.add_argument("--log_to_wandb", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = argument_parser()
    configs = Configs(**common_utils.load_yaml(args.config_filepath))
    common_utils.setup_random_seed(configs.testing_configs.random_seed)
    device = common_utils.setup_device(configs.testing_configs.device)

    task = configs.dataset_configs.tasks[0]
    model_training_regime = (
        "_frozen" if configs.testing_configs.freeze_embedding else ""
    )
    run_name = f"{configs.model_configs.model_type}{model_training_regime}-{task}-negratio_{args.negative_samples_ratio}"
    wandb.init(
        project="kge_ppa",
        entity="protein-kge",
        name=run_name,
        mode="online" if args.log_to_wandb else "disabled",
    )
    wandb.config.update(configs.dict())

    # Load Benchmark dataset
    data_stats = {}
    dataset = BenchmarkLibKGEDataset(
        configs.dataset_configs,
        task,
        from_scratch=configs.model_configs.pretrained_path is None,
        multilabel=True,
        negative_samples_ratio=args.negative_samples_ratio,
    )
    data_stats[f"{task}_num_entities"] = dataset.num_entities
    data_stats[f"{task}_num_relations"] = dataset.num_relations
    data_stats[f"{task}_num_train_inputs"] = len(dataset.train_inputs)
    data_stats[f"{task}_num_valid_inputs"] = len(dataset.valid_inputs)
    data_stats[f"{task}_num_test_inputs"] = len(dataset.test_inputs)
    data_stats[
        f"{task}_num_positive_train_samples"
    ] = dataset.num_positive_train_samples
    data_stats[
        f"{task}_num_positive_valid_samples"
    ] = dataset.num_positive_valid_samples
    data_stats[f"{task}_num_positive_test_samples"] = dataset.num_positive_test_samples
    data_stats[
        f"{task}_num_negative_train_samples"
    ] = dataset.num_negative_train_samples
    data_stats[
        f"{task}_num_negative_valid_samples"
    ] = dataset.num_negative_valid_samples
    data_stats[f"{task}_num_negative_test_samples"] = dataset.num_negative_test_samples

    print(f"num_entities: {dataset.num_entities}")
    print(f"num_relations: {dataset.num_relations}")
    print(f"dataset.num_positive_train_samples: {dataset.num_positive_train_samples}")
    print(f"dataset.num_positive_valid_samples: {dataset.num_positive_valid_samples}")
    print(f"dataset.num_positive_test_samples: {dataset.num_positive_test_samples}")
    print(f"dataset.num_negative_train_samples: {dataset.num_negative_train_samples}")
    print(f"dataset.num_negative_valid_samples: {dataset.num_negative_valid_samples}")
    print(f"dataset.num_negative_test_samples: {dataset.num_negative_test_samples}")
    print(f"configs.model_configs.hidden_size: {configs.model_configs.hidden_size}")

    wandb.config.update(data_stats)

    trainer = Trainer(
        configs,
        dataset.num_entities,
        dataset.num_relations,
        configs.model_configs.hidden_size,
        configs.model_configs.pretrained_path,
        configs.testing_configs.freeze_embedding,
        device,
    )

    # FIXME: Very bad way, but works for now
    X_train = np.array(dataset.train_inputs, dtype=np.int32)
    X_valid = np.array(dataset.valid_inputs, dtype=np.int32)
    X_test = np.array(dataset.test_inputs, dtype=np.int32)

    print("Starting training...")
    trainer.train(
        X_train,
        X_valid,
        X_test,
        np.array(dataset.train_labels, dtype=np.float32),
        np.array(dataset.valid_labels, dtype=np.float32),
        np.array(dataset.test_labels, dtype=np.float32),
    )
    print("Training completed!")


if __name__ == "__main__":
    main()
