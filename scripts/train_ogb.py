import argparse
import os
import sys
sys.path.append(os.getcwd())

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ogb.linkproppred import LinkPropPredDataset, Evaluator

import wandb

from models.complex import ComplEx
from models.regularizers import N3

from utils.logger import Logger


def preprocessing_triples(edge, device):
    subj = edge[:, 0].unsqueeze(0)
    obj = edge[:, 1].unsqueeze(0)
    return torch.cat(
        [subj, torch.zeros_like(subj, device=device), obj],
        axis=0
    )



def train(model, split_edge, optimizer, batch_size, reg_lambda, device):
    model.train()

    pos_train_edge = torch.from_numpy(split_edge["train"]["edge"]).to(device)

    total_loss = total_examples = 0
    loss_fn = nn.CrossEntropyLoss()
    regularizer = N3(reg_lambda)
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        edge = pos_train_edge[perm]
        edge = preprocessing_triples(edge, device)

        predictions, factors = model(edge, score_rhs=True, score_lhs=True)
        # Right hand side loss
        rhs_loss_fit = loss_fn(predictions[0], edge[2].squeeze())
        # Left hand side loss
        lhs_loss_fit = loss_fn(predictions[1], edge[0].squeeze())
        loss_fit = rhs_loss_fit + lhs_loss_fit
        loss_reg, loss_reg_raw, avg_lmbda = regularizer.penalty(factors)

        loss = loss_fit + loss_reg
        loss.backward()
        optimizer.step()

        num_examples = predictions[0].size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, split_edge, evaluator, batch_size, device):
    model.eval()

    pos_train_edge = torch.from_numpy(split_edge["train"]["edge"]).to(device)
    pos_valid_edge = torch.from_numpy(split_edge["valid"]["edge"]).to(device)
    neg_valid_edge = torch.from_numpy(split_edge["valid"]["edge_neg"]).to(device)
    pos_test_edge = torch.from_numpy(split_edge["test"]["edge"]).to(device)
    neg_test_edge = torch.from_numpy(split_edge["test"]["edge_neg"]).to(device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm]
        edge = preprocessing_triples(edge, device)
        pos_train_preds += [model.score(edge).squeeze().cpu()]
        
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm]
        edge = preprocessing_triples(edge, device)
        pos_valid_preds += [model.score(edge).squeeze().cpu()]
        
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm]
        edge = preprocessing_triples(edge, device)
        neg_valid_preds += [model.score(edge).squeeze().cpu()]
        
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm]
        edge = preprocessing_triples(edge, device)
        pos_test_preds += [model.score(edge).squeeze().cpu()]
        
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm]
        edge = preprocessing_triples(edge, device)
        neg_test_preds += [model.score(edge).squeeze().cpu()]
        
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            "y_pred_pos": pos_train_pred,
            "y_pred_neg": neg_valid_pred,
        })[f"hits@{K}"]
        valid_hits = evaluator.eval({
            "y_pred_pos": pos_valid_pred,
            "y_pred_neg": neg_valid_pred,
        })[f"hits@{K}"]
        test_hits = evaluator.eval({
            "y_pred_pos": pos_test_pred,
            "y_pred_neg": neg_test_pred,
        })[f"hits@{K}"]

        results[f"Hits@{K}"] = (train_hits, valid_hits, test_hits)

    return results


def main():
    parser = argparse.ArgumentParser(description="OGBL-PPA (MF)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--reg_lambda", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    print(args)
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    wandb.init(project="kge_ppa", entity="aryopg")
    wandb.config.update({
        "lr": args.lr,
        "reg_lambda": args.reg_lambda,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_channels": args.hidden_channels,
        "dropout": args.dropout,
    })

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)

    dataset = LinkPropPredDataset(name="ogbl-ppa")
    split_edge = dataset.get_edge_split()
    data = dataset[0]

    evaluator = Evaluator(name="ogbl-ppa")
    loggers = {
        "Hits@10": Logger(args.runs, args.output_dir, args),
        "Hits@50": Logger(args.runs, args.output_dir, args),
        "Hits@100": Logger(args.runs, args.output_dir, args),
    }

    for run in range(args.runs):
        model = ComplEx(data["num_nodes"], args.hidden_channels).to(device)
        optimizer = torch.optim.Adagrad(list(model.parameters()), lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, split_edge, optimizer, args.batch_size, args.reg_lambda, device)

            if epoch % args.eval_steps == 0:
                results = test(model, split_edge, evaluator,
                                args.batch_size, device)

                wandb_logs = {
                    "epoch": epoch,
                    "train_loss": loss,
                }
                for key, result in results.items():
                    loggers[key].add_result(run, result)
                    train_hits, valid_hits, test_hits = result
                    wandb_logs.update({
                        f"train_{key.lower()}": train_hits,
                        f"valid_{key.lower()}": valid_hits,
                        f"test_{key.lower()}": test_hits,
                    })

                    print(key)
                    print(f"Run: {run + 1:02d}, "
                            f"Epoch: {epoch:02d}, "
                            f"Loss: {loss:.4f}, "
                            f"Train: {100 * train_hits:.2f}%, "
                            f"Valid: {100 * valid_hits:.2f}%, "
                            f"Test: {100 * test_hits:.2f}%")
                wandb.log(wandb_logs)
            wandb.watch(model)

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()


if __name__ == "__main__":
    main()