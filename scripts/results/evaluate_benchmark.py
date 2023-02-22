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
        return {
            (triple[0].item(), triple[2].item()): triple[1].item() for triple in triples
        }

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
        valid_preds = torch.stack(list(valid_preds.values())).sigmoid().numpy()[:, 0]
        test_preds = torch.stack(list(test_preds.values())).sigmoid().numpy()[:, 0]
    else:
        valid_preds = torch.stack(list(valid_preds.values())).softmax(dim=1)
        test_preds = torch.stack(list(test_preds.values())).softmax(dim=1)

    # Calculate scores
    valid_roc_auc = sklearn.metrics.roc_auc_score(
        valid_trues,
        valid_preds,
        multi_class="ovr",
    )
    test_roc_auc = sklearn.metrics.roc_auc_score(
        test_trues,
        test_preds,
        multi_class="ovr",
    )
    print(valid_roc_auc)
    print(test_roc_auc)


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
