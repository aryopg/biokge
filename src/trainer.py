from typing import Dict, List

import torch
from ogb.linkproppred import Evaluator
from torch import nn
from torch.utils.data import DataLoader

import wandb

from .configs import Configs
from .models.complex import ComplEx
from .models.regularizers import F2, N3, Regularizer
from .utils.logger import Logger

REGULARIZER_MAP = {
    "n3": N3,
    "f2": F2,
}


class Trainer:
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        configs: Configs,
        outputs_dir: str,
        checkpoint_path: str,
        loggers: Dict[str, Logger] = None,
        device: torch.device = None,
    ):
        """
        A python class that governs the training and testing process.

        Args:
            num_entities (int): Number of entities in the KG to initialize the model
            num_relations (int): Number of relations in the KG to initialize the model
            configs (Configs): Configurations of the training
            outputs_dir (str): Directory that stores the training weights and logs
            checkpoint_path (str): Path to the checkpoint directory
            loggers (Dict[str, Logger], optional): Logger object to record the
                training. Defaults to None.
            device (torch.device, optional): The dedicated deviice used for
                training. Defaults to None.
        """
        self.configs = configs
        self.device = device

        self.model = self.setup_model(
            num_entities, num_relations, configs.model_configs
        )
        self.optimizer = self.setup_optimizer(configs.model_configs.optimizer)
        self.loss_fn = self.setup_loss_function(configs.model_configs.loss_fn)
        self.regularizers = self.setup_regularizers(configs.model_configs.regularizers)
        self.grad_accumulation_step = configs.model_configs.grad_accumulation_step

        self.outputs_dir = outputs_dir
        self.checkpoint_path = checkpoint_path
        self.loggers = loggers

    def setup_model(
        self, num_entities: int, num_relations: int, model_configs: dict
    ) -> nn.Module:
        """
        Setup model based on the model type, number of entities and relations
        mentioned in the config file

        Args:
            num_entities (int): Number of entities in the KG
            num_relations (int): Number of relations in the KG
            model_configs (dict): Model configurations

        Returns:
            nn.Module: The KGE model to be trained
        """
        if model_configs.model_type == "complex":
            return ComplEx(
                num_entities,
                num_relations,
                model_configs.hidden_size,
                model_configs.init_range,
                model_configs.init_size,
            ).to(self.device)

    def setup_optimizer(self, optimizer: str) -> torch.optim.Optimizer:
        """
        Setup optimizer based on the optimizer name

        Args:
            optimizer (str): optimizer name

        Returns:
            torch.optim.Optimizer: Optimizer class
        """
        if optimizer == "adam":
            return torch.optim.Adam(
                list(self.model.parameters()),
                lr=self.configs.model_configs.learning_rate,
            )
        elif optimizer == "adagrad":
            return torch.optim.Adagrad(
                list(self.model.parameters()),
                lr=self.configs.model_configs.learning_rate,
            )

    def setup_regularizers(self, regularizers_list: List[dict]) -> List[Regularizer]:
        """
        Setup regularizers based on the regularizer names

        Args:
            regularizers_list (List[dict]): List of regularizer configs

        Returns:
            List[Regularizer]: List of regularizers that will be used
        """
        regularizers = []
        for regularizer in regularizers_list:
            regularizers += [REGULARIZER_MAP[regularizer.type](regularizer.coeff)]
        return regularizers

    def setup_loss_function(self, loss_fn: str) -> nn.Module:
        """
        Setup loss function based on the loss function name

        Args:
            loss_fn (str): loss function name

        Returns:
            nn.Module: Loss function
        """
        if loss_fn == "crossentropy":
            return nn.CrossEntropyLoss()

    def training_epoch(self, dataset) -> float:
        """
        Method that represents one training epoch (multiple training steps).
        Currently designed only for OGB LinkPropPredDataset

        Args:
            dataset (tbd): Dataset object containing the triplets

        Returns:
            float: the averaged loss of the epoch
        """

        # Set model mode to train
        self.model.train()

        # Retrieve all training triples
        ## Convert to a torch tensor
        if dataset.name == "ogbl-ppa":
            pos_train_edge = torch.from_numpy(dataset["train"]["edge"])
        elif dataset.name == "dsi-bdi-biokg":
            pos_train_edge = torch.from_numpy(dataset.train)

        ## Load to DataLoader for sampling
        train_data_loader = DataLoader(
            range(pos_train_edge.size(0)),
            self.configs.model_configs.batch_size,
            shuffle=True,
        )

        total_loss = 0
        total_examples = 0
        for iteration, perm in enumerate(train_data_loader):
            # Sample triplets
            edge = pos_train_edge[perm]

            # Preprocess triples based on the dataset specification
            edge = self.preprocessing_triples(edge).to(self.device)

            # Get right hand and left hand predictions (symmetry)
            predictions, factors = self.model(edge, score_rhs=True, score_lhs=True)
            # Right hand side loss
            rhs_loss_fit = self.loss_fn(predictions[0], edge[2].squeeze())
            # Left hand side loss
            lhs_loss_fit = self.loss_fn(predictions[1], edge[0].squeeze())

            # Model loss is a summation of Right hand and Left hand losses
            loss_fit = rhs_loss_fit + lhs_loss_fit

            # Compute regularizers losses
            loss_regs = 0
            for regularizer in self.regularizers:
                loss_reg = regularizer.penalty(factors)
                loss_regs += loss_reg

            # Sum of model and regularizers losses
            loss = loss_fit + loss_regs

            # Average by gradient accumulation step if any
            loss = loss / self.grad_accumulation_step

            # Compute loss
            loss.backward()

            # Run backprop if iteration falls on the gradient accumulation step
            if ((iteration + 1) % self.grad_accumulation_step == 0) or (
                (iteration + 1) == len(train_data_loader)
            ):
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Record epoch loss
            num_examples = predictions[0].size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples

        return total_loss / total_examples

    def train(self, dataset, evaluator=None):
        for epoch in range(1, 1 + self.configs.training_configs.epochs):
            train_loss = self.training_epoch(dataset)

            if epoch % self.configs.training_configs.eval_steps == 0:
                results = self.test(dataset, evaluator)

                wandb_logs = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                }
                for key, result in results.items():
                    if key in self.loggers:
                        self.loggers[key].add_result(result)
                    train_hits, valid_hits, test_hits = result
                    wandb_logs.update(
                        {
                            f"train_{key.lower()}": train_hits,
                            f"valid_{key.lower()}": valid_hits,
                            f"test_{key.lower()}": test_hits,
                        }
                    )

                    print(key)
                    print(
                        f"Epoch: {epoch:02d}, "
                        f"Loss: {train_loss:.4f}, "
                        f"Train: {100 * train_hits:.2f}%, "
                        f"Valid: {100 * valid_hits:.2f}%, "
                        f"Test: {100 * test_hits:.2f}%"
                    )
                wandb.log(wandb_logs)
            wandb.watch(self.model)
            self.save_checkpoint(epoch, wandb_logs)

        for key in self.loggers.keys():
            self.loggers[key].save_statistics()

    @torch.no_grad()
    def test(self, dataset, evaluator: Evaluator) -> dict:
        """
        Test model performance.
        This code is mostly adopted from the OGB Link Prediction codes.

        Args:
            dataset (tbd): Dataset of triples
            evaluator (Evaluator): OGB evaluator class

        Returns:
            dict: Model metrics
        """

        def test_iteration(edge) -> torch.Tensor:
            """Run inference on data batches"""
            preds = []
            for perm in DataLoader(
                range(edge.size(0)), self.configs.model_configs.batch_size
            ):
                perm_edge = edge[perm]
                perm_edge = self.preprocessing_triples(perm_edge).to(self.device)
                preds += [self.model.score(perm_edge).squeeze().cpu()]

            return torch.cat(preds, dim=0)

        def hits_evaluation(y_pred_pos, y_pred_neg, K: int) -> float:
            """Use OGB Evaluator to get the Hits metrics"""
            evaluator.K = K
            return evaluator.eval(
                {
                    "y_pred_pos": y_pred_pos,
                    "y_pred_neg": y_pred_neg,
                }
            )[f"hits@{K}"]

        # Set model mode to evaluation
        self.model.eval()

        # Loop over predicates and collect metrics
        results = {"Hits@10": (0, 0, 0), "Hits@50": (0, 0, 0), "Hits@100": (0, 0, 0)}
        for relation_name, relation in dataset.relation_voc.items():

            # Get all the triples with negative samples for validation and test data
            pos_train_edge = torch.from_numpy(dataset.train_separated[relation])
            pos_valid_edge = torch.from_numpy(dataset.valid[relation])
            neg_valid_edge = torch.from_numpy(dataset.valid_negs[relation])
            pos_test_edge = torch.from_numpy(dataset.test[relation])
            neg_test_edge = torch.from_numpy(dataset.test_negs[relation])

            # Run test iterations using the loaded triples
            pos_train_pred = test_iteration(pos_train_edge)
            pos_valid_pred = test_iteration(pos_valid_edge)
            neg_valid_pred = test_iteration(neg_valid_edge)
            pos_test_pred = test_iteration(pos_test_edge)
            neg_test_pred = test_iteration(neg_test_edge)

            # Calculate the results by comparing the inference of
            # positive and negative samples
            for K in [10, 50, 100]:
                train_hits = hits_evaluation(pos_train_pred, neg_valid_pred, K)
                valid_hits = hits_evaluation(pos_valid_pred, neg_valid_pred, K)
                test_hits = hits_evaluation(pos_test_pred, neg_test_pred, K)

                results[f"Hits@{K}_{relation_name}"] = (
                    train_hits,
                    valid_hits,
                    test_hits,
                )

                # Keep track of unseparated metrics as well: weighted average
                results[f"Hits@{K}"] = [
                    avg + ((weight * metric) / normaliser)
                    for avg, metric, weight, normaliser in zip(
                        results[f"Hits@{K}"],
                        results[f"Hits@{K}_{relation_name}"],
                        (len(pos_train_pred), len(pos_valid_pred), len(pos_test_pred)),
                        (
                            dataset.num_train_entries,
                            dataset.num_valid_entries,
                            dataset.num_test_entries,
                        ),
                    )
                ]

        return results

    def preprocessing_triples(self, edge) -> torch.Tensor:
        """
        Preprocess triples based on the dataset.
        OGBL-PPA does not provide relations within the triples,
        but it only contains one type of relation.

        Args:
            edge (tbd): subject, predicate and object.
                OGBL-PPA misses the predicate.

        Returns:
            torch.Tensor: Tensors of the triple (subject, predicate, object)
        """
        if self.configs.dataset_configs.dataset_name == "ogbl-ppa":
            subj = edge[:, 0].unsqueeze(0)
            obj = edge[:, 1].unsqueeze(0)
            # Assign zeros to represent the uniform predicates of OGBL-PPA
            return torch.cat([subj, torch.zeros_like(subj), obj], axis=0)
        elif self.configs.dataset_configs.dataset_name == "dsi-bdi-biokg":
            subj = edge[:, 0].unsqueeze(0)
            relation = edge[:, 1].unsqueeze(0)
            obj = edge[:, 2].unsqueeze(0)
            return torch.cat([subj, relation, obj], axis=0)

    # TODO: save checkpoint of regularizers too
    def save_checkpoint(self, epoch: int, metrics: dict):
        """
        Save checkpoints of all training components

        Args:
            epoch (int): Current epoch
            metrics (dict): Current metrics achieved by the model
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        checkpoint.update(metrics)
        torch.save(checkpoint, f"{self.checkpoint_path}_{epoch}.pt")
