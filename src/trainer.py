from typing import Dict, List

import numpy
import torch
import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import wandb

from .configs import Configs
from .evaluator import Evaluator
from .models.complex import ComplEx
from .models.regularizers import F2, N3, Regularizer
from .models.rotate import RotatE
from .utils.logger import Logger

REGULARIZER_MAP = {
    "n3": N3,
    "f2": F2,
}

MODELS_MAP = {"complex": ComplEx, "rotate": RotatE}


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
        silent: bool = False,
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
        self.silent = silent

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

        return MODELS_MAP[model_configs.model_type](
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
        if loss_fn == "1vsAll":
            return nn.CrossEntropyLoss()
        elif loss_fn == "negsampling":
            return nn.BCEWithLogitsLoss()

    def training_epoch(self, dataset, epoch) -> float:
        """
        Method that represents one training epoch (multiple training steps).

        Args:
            dataset (tbd): Dataset object containing the triplets

        Returns:
            float: the averaged loss of the epoch
        """

        # Set model mode to train
        self.model.train()

        # Retrieve all training triples and convert to a torch tensor
        pos_train_edge = torch.from_numpy(dataset.train)

        ## Create DataLoader for sampling
        train_data_loader = DataLoader(
            range(pos_train_edge.size(1)),
            self.configs.model_configs.batch_size,
            shuffle=True,
        )

        total_loss = 0
        total_examples = 0
        for iteration, perm in tqdm.tqdm(
            enumerate(train_data_loader),
            desc=f"EPOCH {epoch}, batch ",
            unit="",
            total=len(train_data_loader),
            disable=self.silent,
        ):

            # Sample triples
            edge = pos_train_edge[:, perm].to(self.device)

            # Get right hand and left hand predictions (symmetry)
            predictions, factors = self.model(
                edge,
                score_rhs=self.configs.model_configs.score_rhs,
                score_lhs=self.configs.model_configs.score_lhs,
                score_rel=self.configs.model_configs.score_rel,
            )

            if self.configs.model_configs.loss_fn == "negsampling":
                # Generate negative samples
                negs = dataset.generate_negs_tensor(
                    edge, self.configs.model_configs.neg_sampling_rate
                )
                # Score generated samples
                neg_predictions, factors = self.model(
                    negs,
                    score_rhs=False,
                    score_lhs=False,
                    score_rel=False,
                )

                positive_sample_loss = self.loss_fn(
                    predictions, torch.ones_like(predictions)
                )
                negative_sample_loss = self.loss_fn(
                    neg_predictions, torch.zeros_like(neg_predictions)
                )

                loss_fit = (positive_sample_loss + negative_sample_loss) / 2

                # Sum loss
            elif self.configs.model_configs.loss_fn == "1vsAll":
                # Model loss is a summation of Right hand, Relation, Left hand losses
                loss_fit = 0

                # Right hand side loss
                if self.configs.model_configs.score_rhs:
                    rhs_loss_fit = self.loss_fn(predictions[0], edge[2].squeeze())
                    loss_fit += rhs_loss_fit
                # Relationship loss
                if self.configs.model_configs.score_rel:
                    rel_loss_fit = self.loss_fn(predictions[1], edge[1].squeeze())
                    loss_fit += rel_loss_fit
                # Left hand side loss
                if self.configs.model_configs.score_lhs:
                    if self.configs.model_configs.score_rel:
                        lhs_loss_fit = self.loss_fn(predictions[2], edge[0].squeeze())
                    else:
                        lhs_loss_fit = self.loss_fn(predictions[1], edge[0].squeeze())
                    loss_fit += lhs_loss_fit

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
            if self.configs.model_configs.loss_fn == "negsampling":
                num_examples = predictions.size(0)
            elif self.configs.model_configs.loss_fn == "1vsAll":
                num_examples = predictions[0].size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples

        return total_loss / total_examples

    def train(self, dataset, evaluator):
        for epoch in range(1, 1 + self.configs.training_configs.epochs):
            train_loss = self.training_epoch(dataset, epoch)

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
        This code is mostly adopted from the OGB Link Prediction code: https://github.com/snap-stanford/ogb/tree/master/ogb/linkproppred

        Args:
            dataset (Dataset): dataset instance of triples
            evaluator (Evaluator): evaluator instance

        Returns:
            metrics (dict)
        """

        def test_iteration(edge) -> torch.Tensor:
            """Run inference on data batches"""
            return torch.cat(
                [
                    self.model.score(edge[:, perm].to(self.device)).squeeze().cpu()
                    for perm in DataLoader(
                        range(edge.size(1)), self.configs.model_configs.batch_size
                    )
                ],
                dim=0,
            )

        # Set model mode to evaluation
        self.model.eval()

        # Loop over Ks and collect metrics
        results = {}
        for K in [10, 50, 100]:

            ### TOTAL
            pos_train_edge = torch.from_numpy(dataset.train)
            pos_valid_edge = torch.from_numpy(
                numpy.hstack(list(dataset.valid.values()))
            )
            neg_valid_edge = torch.from_numpy(
                numpy.hstack(list(dataset.valid_negs.values()))
            )
            pos_test_edge = torch.from_numpy(numpy.hstack(list(dataset.test.values())))
            neg_test_edge = torch.from_numpy(
                numpy.hstack(list(dataset.test_negs.values()))
            )

            # Run test iterations using the loaded triples
            pos_train_pred = test_iteration(pos_train_edge)
            pos_valid_pred = test_iteration(pos_valid_edge)
            neg_valid_pred = test_iteration(neg_valid_edge)
            pos_test_pred = test_iteration(pos_test_edge)
            neg_test_pred = test_iteration(neg_test_edge)

            # Evaluate
            train_hits = evaluator.eval(pos_train_pred, neg_valid_pred, K)
            valid_hits = evaluator.eval(pos_valid_pred, neg_valid_pred, K)
            test_hits = evaluator.eval(pos_test_pred, neg_test_pred, K)

            # Collect
            results[f"Hits@{K}"] = (
                train_hits,
                valid_hits,
                test_hits,
            )

            ### SEPARATED
            for relation_name, relation in dataset.relation_voc.items():

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

                # Evaluate
                train_hits = evaluator.eval(pos_train_pred, neg_valid_pred, K)
                valid_hits = evaluator.eval(pos_valid_pred, neg_valid_pred, K)
                test_hits = evaluator.eval(pos_test_pred, neg_test_pred, K)

                # Collect
                results[f"Hits@{K}_{relation_name}"] = (
                    train_hits,
                    valid_hits,
                    test_hits,
                )

        return results

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
