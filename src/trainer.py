from typing import Dict

import torch
import torch.nn.functional as F
from ogb.linkproppred import Evaluator
from torch import nn
from torch.utils.data import DataLoader

import wandb

from .configs import Configs
from .models.complex import ComplEx
from .models.regularizers import F2, N3
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
        self.configs = configs
        self.device = device

        self.model = self.setup_model(
            num_entities, num_relations, configs.model_configs
        )
        self.optimizer = self.setup_optimizer(configs.model_configs.optimizer)
        self.loss_fn = self.setup_loss_function(configs.model_configs.loss_fn)
        self.regularizers = self.setup_regularizers(configs.model_configs.regularizers)

        self.evaluator = Evaluator(name="ogbl-ppa")

        self.outputs_dir = outputs_dir
        self.checkpoint_path = checkpoint_path
        self.loggers = loggers

    def setup_model(self, num_entities: int, num_relations: int, model_configs: dict):
        if model_configs.model_type == "complex":
            return ComplEx(
                num_entities,
                num_relations,
                model_configs.hidden_size,
                model_configs.init_range,
                model_configs.init_size,
            ).to(self.device)

    def setup_optimizer(self, optimizer):
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

    def setup_regularizers(self, regularizer):
        regularizers = []
        for regularizer in regularizers:
            if regularizer.coeff > 0:
                regularizers += [REGULARIZER_MAP[regularizer.type](regularizer.coeff)]
        return regularizers

    def setup_loss_function(self, loss_fn):
        if loss_fn == "crossentropy":
            return nn.CrossEntropyLoss()

    def training_step(self, dataset):
        self.model.train()

        pos_train_edge = torch.from_numpy(dataset["train"]["edge"])

        total_loss = 0
        total_examples = 0
        for iteration, perm in enumerate(
            DataLoader(
                range(pos_train_edge.size(0)),
                self.configs.model_configs.batch_size,
                shuffle=True,
            )
        ):
            edge = pos_train_edge[perm]
            edge = self.preprocessing_triples(edge).to(self.device)

            predictions, factors = self.model(edge, score_rhs=True, score_lhs=True)
            # Right hand side loss
            rhs_loss_fit = self.loss_fn(predictions[0], edge[2].squeeze())
            # Left hand side loss
            lhs_loss_fit = self.loss_fn(predictions[1], edge[0].squeeze())
            loss_fit = rhs_loss_fit + lhs_loss_fit
            loss_regs = 0
            for regularizer in self.regularizers:
                loss_reg, loss_reg_raw, avg_lmbda = regularizer.penalty(factors)
                loss_regs += loss_reg

            loss = loss_fit + loss_regs

            num_examples = predictions[0].size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples

            loss = loss / self.configs.model_configs.grad_accumulation_step
            loss.backward()
            if (iteration + 1) % self.configs.model_configs.grad_accumulation_step == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        return total_loss / total_examples

    def train(self, dataset):
        for epoch in range(1, 1 + self.configs.training_configs.epochs):
            train_loss = self.training_step(dataset)

            if epoch % self.configs.training_configs.eval_steps == 0:
                results = self.test(dataset)

                wandb_logs = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                }
                for key, result in results.items():
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
    def test(self, dataset):
        self.model.eval()

        pos_train_edge = torch.from_numpy(dataset["train"]["edge"])
        pos_valid_edge = torch.from_numpy(dataset["valid"]["edge"])
        neg_valid_edge = torch.from_numpy(dataset["valid"]["edge_neg"])
        pos_test_edge = torch.from_numpy(dataset["test"]["edge"])
        neg_test_edge = torch.from_numpy(dataset["test"]["edge_neg"])

        pos_train_preds = []
        for perm in DataLoader(
            range(pos_train_edge.size(0)), self.configs.model_configs.batch_size
        ):
            edge = pos_train_edge[perm]
            edge = self.preprocessing_triples(edge).to(self.device)
            pos_train_preds += [self.model.score(edge).squeeze().cpu()]

        pos_train_pred = torch.cat(pos_train_preds, dim=0)

        pos_valid_preds = []
        for perm in DataLoader(
            range(pos_valid_edge.size(0)), self.configs.model_configs.batch_size
        ):
            edge = pos_valid_edge[perm]
            edge = self.preprocessing_triples(edge).to(self.device)
            pos_valid_preds += [self.model.score(edge).squeeze().cpu()]

        pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

        neg_valid_preds = []
        for perm in DataLoader(
            range(neg_valid_edge.size(0)), self.configs.model_configs.batch_size
        ):
            edge = neg_valid_edge[perm]
            edge = self.preprocessing_triples(edge).to(self.device)
            neg_valid_preds += [self.model.score(edge).squeeze().cpu()]

        neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

        pos_test_preds = []
        for perm in DataLoader(
            range(pos_test_edge.size(0)), self.configs.model_configs.batch_size
        ):
            edge = pos_test_edge[perm]
            edge = self.preprocessing_triples(edge).to(self.device)
            pos_test_preds += [self.model.score(edge).squeeze().cpu()]

        pos_test_pred = torch.cat(pos_test_preds, dim=0)

        neg_test_preds = []
        for perm in DataLoader(
            range(neg_test_edge.size(0)), self.configs.model_configs.batch_size
        ):
            edge = neg_test_edge[perm]
            edge = self.preprocessing_triples(edge).to(self.device)
            neg_test_preds += [self.model.score(edge).squeeze().cpu()]

        neg_test_pred = torch.cat(neg_test_preds, dim=0)

        results = {}
        for K in [10, 50, 100]:
            self.evaluator.K = K
            train_hits = self.evaluator.eval(
                {
                    "y_pred_pos": pos_train_pred,
                    "y_pred_neg": neg_valid_pred,
                }
            )[f"hits@{K}"]
            valid_hits = self.evaluator.eval(
                {
                    "y_pred_pos": pos_valid_pred,
                    "y_pred_neg": neg_valid_pred,
                }
            )[f"hits@{K}"]
            test_hits = self.evaluator.eval(
                {
                    "y_pred_pos": pos_test_pred,
                    "y_pred_neg": neg_test_pred,
                }
            )[f"hits@{K}"]

            results[f"Hits@{K}"] = (train_hits, valid_hits, test_hits)

        return results

    def preprocessing_triples(self, edge):
        if self.configs.dataset_configs.dataset_name == "ogbl-ppa":
            subj = edge[:, 0].unsqueeze(0)
            obj = edge[:, 1].unsqueeze(0)
            return torch.cat([subj, torch.zeros_like(subj), obj], axis=0)

    def save_checkpoint(self, epoch: int, metrics: dict):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        checkpoint.update(metrics)
        torch.save(checkpoint, f"{self.checkpoint_path}_{epoch}.pt")
