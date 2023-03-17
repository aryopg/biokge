import torch
import tqdm
import wandb
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .evaluators.classification_evaluator import calculate_metrics
from .models.classifier import Classifier


class Trainer:
    def __init__(
        self,
        configs,
        entity_size,
        num_relations,
        rank,
        pretrained_model_path,
        freeze_embedding,
        device,
    ):
        self.configs = configs
        self.entity_size = entity_size
        self.num_relations = num_relations
        self.grad_clip = configs.model_configs.grad_clip

        self.model = Classifier(entity_size, num_relations, rank).to(device)
        self.device = device
        if pretrained_model_path is not None:
            self.load_embedding_weights(pretrained_model_path, freeze_embedding)
        else:
            self.init_embedding_weights()

        if self.num_relations > 1:
            self.loss_fn = torch.nn.NLLLoss()
        else:
            self.loss_fn = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(
            [param for param in self.model.parameters() if param.requires_grad == True],
            lr=self.configs.model_configs.learning_rate,
        )

    def load_embedding_weights(self, pretrained_model_path, freeze_embedding=True):
        model_state_dict = torch.load(pretrained_model_path, map_location=self.device)
        self.model.entity_embeddings.weight.data = model_state_dict["model"][0][
            "_entity_embedder._embeddings.weight"
        ]
        if freeze_embedding:
            self.model.entity_embeddings.weight.requires_grad = False

    def init_embedding_weights(self, init_range=1, init_size=1e-3):
        self.model.entity_embeddings.weight.data.uniform_(-init_range, init_range)
        self.model.entity_embeddings.weight.data *= init_size

    def training_epoch(self, inputs, labels, epoch):
        # Set model mode to train
        self.model.train()

        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels)
        if self.num_relations > 1:
            labels = labels.argmax(dim=1)

        ## Create DataLoader for sampling
        data_loader = DataLoader(
            range(inputs.size(0)),
            self.configs.model_configs.batch_size,
            shuffle=True,
        )

        total_loss = 0
        total_examples = 0
        for iteration, perm in tqdm.tqdm(
            enumerate(data_loader),
            desc=f"EPOCH {epoch}, batch ",
            unit="",
            total=len(data_loader),
        ):
            self.optimizer.zero_grad()

            batch_inputs = inputs[perm].to(self.device)
            batch_labels = labels[perm].to(self.device)

            predictions = self.model(batch_inputs)
            # print(predictions)
            if self.num_relations > 1:
                loss = self.loss_fn(F.log_softmax(predictions), batch_labels)
            else:
                loss = self.loss_fn(torch.sigmoid(predictions).squeeze(), batch_labels)

            loss.backward()

            # Check gradient norm if loss turns into infinity
            # if torch.isinf(loss):
            #     total_norm = 0
            #     for p in self.model.parameters():
            #         param_norm = p.grad.data.norm(2)
            #         total_norm += param_norm.item() ** 2
            #     total_norm = total_norm ** (1.0 / 2)
            #     print(total_norm)
            #     raise Exception("loss is infinity. Stopping training...")

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            num_examples = predictions.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples
        return total_loss / total_examples

    @torch.no_grad()
    def test(self, inputs, labels, split) -> dict:
        # Set model mode to evaluation
        self.model.eval()

        inputs = torch.from_numpy(inputs).to(self.device)

        if self.num_relations > 1:
            preds = F.softmax(self.model(inputs), dim=1)
        else:
            preds = F.sigmoid(self.model(inputs))

        results = calculate_metrics(
            labels, preds.cpu().numpy(), self.num_relations, split
        )

        return results

    def train(
        self,
        train_inputs,
        valid_inputs,
        test_inputs,
        train_labels,
        valid_labels,
        test_labels,
    ):
        for epoch in range(1, 1 + self.configs.testing_configs.epochs):
            train_loss = self.training_epoch(train_inputs, train_labels, epoch)

            train_results = self.test(train_inputs, train_labels, "train")
            valid_results = self.test(valid_inputs, valid_labels, "valid")
            test_results = self.test(test_inputs, test_labels, "test")

            wandb_logs = {
                "epoch": epoch,
                "train_loss": train_loss,
            }
            wandb_logs.update(train_results)
            wandb_logs.update(valid_results)
            wandb_logs.update(test_results)
            for metrics_name, metrics_value in wandb_logs.items():
                if metrics_name.endswith(
                    (
                        "loss",
                        "mean_average_precision",
                        "averaged_auc_precision_recall",
                        "averaged_auc_roc",
                    )
                ):
                    print(f" {metrics_name}: {metrics_value}")

            wandb.log(wandb_logs)
