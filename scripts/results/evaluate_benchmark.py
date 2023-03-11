import argparse
import itertools
import random

import numpy
import sklearn
import torch

import kge
import kge.model


def evaluate(model_path):

    # Load model
    model = kge.model.KgeModel.create_from(kge.util.io.load_checkpoint(model_path))

    # Load data
    dataset = model.dataset
    train_triples = dataset.load_triples("train")
    valid_triples = dataset.load_triples("valid")
    test_triples = dataset.load_triples("test")

    if dataset._num_relations == 1:
        random.seed(42)
        subjects = set(
            list(
                itertools.chain.from_iterable(
                    [
                        triples[:, 0].tolist()
                        for triples in [train_triples, valid_triples, test_triples]
                    ]
                )
            )
        )
        objects = set(
            list(
                itertools.chain.from_iterable(
                    [
                        triples[:, 2].tolist()
                        for triples in [train_triples, valid_triples, test_triples]
                    ]
                )
            )
        )

        def get_negatives(triples, others_triples, others_negatives, ratio=1):
            def get_adjacency_matrix(triples):
                matrix = numpy.zeros(
                    (dataset._num_entities, dataset._num_entities), dtype=bool
                )
                matrix[triples[:, 0], triples[:, 2]] = 1
                return matrix

            adjacency_matrix = get_adjacency_matrix(triples)
            adjacency_others = [
                get_adjacency_matrix(other_triples) for other_triples in others_triples
            ]

            count = 0
            negatives = set()
            while count != int(len(triples) * ratio):
                subject = random.sample(subjects, 1)[0]
                object = random.sample(objects, 1)[0]
                if (
                    (adjacency_matrix[subject, object] == 0)
                    and all(
                        [
                            adjacency_other[subject, object] == 0
                            for adjacency_other in adjacency_others
                        ]
                    )
                    and (subject, object) not in negatives
                    and all(
                        [
                            (subject, object) not in other_negatives
                            for other_negatives in others_negatives
                        ]
                    )
                ):
                    negatives.add((subject, object))
                    count += 1
            return negatives

        valid_negatives = get_negatives(
            valid_triples, [train_triples, test_triples], []
        )
        test_negatives = get_negatives(
            test_triples, [train_triples, valid_triples], [valid_negatives]
        )

        # Swap relation to correct binary code
        valid_triples[:, 1] = torch.full(
            (
                len(
                    valid_triples,
                ),
            ),
            1,
        )
        test_triples[:, 1] = torch.full(
            (
                len(
                    test_triples,
                ),
            ),
            1,
        )
        # Add negatives
        valid_triples = torch.concatenate(
            [
                valid_triples,
                torch.stack(
                    [
                        torch.tensor([subject, 0, object])
                        for subject, object in valid_negatives
                    ]
                ),
            ]
        )
        test_triples = torch.concatenate(
            [
                test_triples,
                torch.stack(
                    [
                        torch.tensor([subject, 0, object])
                        for subject, object in test_negatives
                    ]
                ),
            ]
        )

        # Save negatives
        numpy.savetxt(
            "valid_with_negatives.del", valid_triples, fmt="%i", delimiter="\t"
        )
        numpy.savetxt("test_with_negatives.del", test_triples, fmt="%i", delimiter="\t")

    # Get trues
    def get_trues(triples):
        res = {
            (triple[0].item(), triple[2].item()): torch.zeros(
                max(dataset._num_relations, 2)
            )
            for triple in triples
        }
        for triple in triples:
            res[triple[0].item(), triple[2].item()][triple[1].item()] = 1
        return res

    valid_trues = get_trues(valid_triples)
    test_trues = get_trues(test_triples)

    # Get predictions
    def get_preds(triples):
        return {
            (triple[0].item(), triple[2].item()): torch.concatenate(
                [
                    model.score_spo(
                        triple[None, 0],
                        torch.tensor([relation], requires_grad=False).long(),
                        triple[None, 2],
                        "o",
                    ).detach()
                    for relation in range(dataset._num_relations)
                ]
            )
            for triple in triples
        }

    valid_preds = get_preds(valid_triples)
    test_preds = get_preds(test_triples)

    # Transform
    assert valid_trues.keys() == valid_preds.keys()
    assert test_trues.keys() == test_preds.keys()
    valid_trues = numpy.stack(list(valid_trues.values()))
    test_trues = numpy.stack(list(test_trues.values()))

    if dataset._num_relations == 1:
        valid_preds = torch.stack(list(valid_preds.values()))  # .sigmoid()
        valid_preds = torch.hstack([1 - valid_preds, valid_preds])
        test_preds = torch.stack(list(test_preds.values()))  # .sigmoid()
        test_preds = torch.hstack([1 - test_preds, test_preds])
    elif dataset._num_relations == 2:
        valid_preds = torch.stack(list(valid_preds.values())).numpy()
        test_preds = torch.stack(list(test_preds.values())).numpy()
    else:
        valid_preds = torch.stack(list(valid_preds.values())).softmax(dim=1).numpy()
        test_preds = torch.stack(list(test_preds.values())).softmax(dim=1).numpy()

    ## Calculate scores

    # Hits@1
    valid_hits_at_1 = valid_preds.argmax(axis=1) == valid_trues.argmax(axis=1)
    valid_hits_at_1 = sum(valid_hits_at_1) / len(valid_hits_at_1)
    test_hits_at_1 = test_preds.argmax(axis=1) == test_trues.argmax(axis=1)
    test_hits_at_1 = sum(test_hits_at_1) / len(test_hits_at_1)
    print(f"Validation hits@1: {valid_hits_at_1}")
    print(f"Test hits@1: {test_hits_at_1}")

    # AUROC
    valid_roc_auc = sklearn.metrics.roc_auc_score(
        valid_trues.argmax(axis=1),
        valid_preds[:, 1] if dataset._num_relations < 3 else valid_preds,
        multi_class="ovr",
    )
    test_roc_auc = sklearn.metrics.roc_auc_score(
        test_trues.argmax(axis=1),
        test_preds[:, 1] if dataset._num_relations < 3 else test_preds,
        multi_class="ovr",
    )
    print(f"Validation AUROC: {valid_roc_auc}")
    print(f"Test AUROC: {test_roc_auc}")

    # AUPRC
    valid_prc = [
        sklearn.metrics.precision_recall_curve(
            valid_trues[:, relation], valid_preds[:, relation]
        )
        for relation in range(dataset._num_relations)
    ]
    test_prc = [
        sklearn.metrics.precision_recall_curve(
            test_trues[:, relation], test_preds[:, relation]
        )
        for relation in range(dataset._num_relations)
    ]
    valid_prc_auc = [
        sklearn.metrics.auc(relation_prc[1], relation_prc[0])
        for relation_prc in valid_prc
    ]
    test_prc_auc = [
        sklearn.metrics.auc(relation_prc[1], relation_prc[0])
        for relation_prc in test_prc
    ]
    valid_prc_auc = sum(valid_prc_auc) / len(valid_prc_auc)
    test_prc_auc = sum(test_prc_auc) / len(test_prc_auc)
    print(f"Validation AUPRC: {valid_prc_auc}")
    print(f"Test AUPRC: {test_prc_auc}")

    # MAP
    valid_map = [
        sklearn.metrics.average_precision_score(
            valid_trues[:, relation], valid_preds[:, relation]
        )
        for relation in range(dataset._num_relations)
    ]
    test_map = [
        sklearn.metrics.average_precision_score(
            test_trues[:, relation], test_preds[:, relation]
        )
        for relation in range(dataset._num_relations)
    ]
    valid_map = sum(valid_map) / len(valid_map)
    test_map = sum(test_map) / len(test_map)
    print(f"Validation MAP: {valid_map}")
    print(f"Test MAP: {test_map}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Protein Knowledge Graph Embedding Project: benchmark evaluation"
    )
    parser.add_argument(
        "--model",
        type=str,
    )
    args = parser.parse_args()

    evaluate(args.model)
