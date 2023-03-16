import argparse
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

    # Set seed
    random.seed(42)

    # Create negative samples
    def get_negatives(triples, others_triples, others_negatives, ratio=1):
        def get_adjacency_matrix(triples):
            matrix = numpy.full(
                (dataset._num_entities, dataset._num_entities),
                fill_value=-1,
                dtype=numpy.byte,
            )
            matrix[triples[:, 0], triples[:, 2]] = triples[:, 1]
            return matrix

        adjacency_matrix = get_adjacency_matrix(triples)
        adjacency_others = [
            get_adjacency_matrix(other_triples) for other_triples in others_triples
        ]

        count = 0
        negatives = set()
        while count != int(len(triples) * ratio):
            subject = random.randrange(dataset._num_entities)
            object = random.randrange(dataset._num_entities)
            relation = random.randrange(dataset._num_relations)
            if (
                (adjacency_matrix[subject, object] != relation)
                and all(
                    [
                        adjacency_other[subject, object] != relation
                        for adjacency_other in adjacency_others
                    ]
                )
                and (subject, relation, object) not in negatives
                and all(
                    [
                        (subject, relation, object) not in other_negatives
                        for other_negatives in others_negatives
                    ]
                )
            ):
                negatives.add((subject, relation, object))
                count += 1
        return negatives

    train_negatives = get_negatives(train_triples, [valid_triples, test_triples], [])
    valid_negatives = get_negatives(
        valid_triples, [train_triples, test_triples], [train_negatives]
    )
    test_negatives = get_negatives(
        test_triples, [train_triples, valid_triples], [train_negatives, valid_negatives]
    )

    # Stack
    train_negatives = torch.stack(
        [
            torch.tensor([subject, relation, object])
            for subject, relation, object in train_negatives
        ]
    )
    valid_negatives = torch.stack(
        [
            torch.tensor([subject, relation, object])
            for subject, relation, object in valid_negatives
        ]
    )
    test_negatives = torch.stack(
        [
            torch.tensor([subject, relation, object])
            for subject, relation, object in test_negatives
        ]
    )

    # Save negatives
    numpy.savetxt(
        "train_with_negatives.del",
        torch.concatenate([train_triples, train_negatives]),
        fmt="%i",
        delimiter="\t",
    )
    numpy.savetxt(
        "valid_with_negatives.del",
        torch.concatenate([valid_triples, valid_negatives]),
        fmt="%i",
        delimiter="\t",
    )
    numpy.savetxt(
        "test_with_negatives.del",
        torch.concatenate([test_triples, test_negatives]),
        fmt="%i",
        delimiter="\t",
    )

    # Calculate metrics per relation
    auc_roc = []
    auc_prc = []
    map = []
    for relation in range(dataset._num_relations):

        # Get positive triples
        relation_valid_pos_triples = valid_triples[valid_triples[:, 1] == relation, :]
        relation_test_pos_triples = test_triples[test_triples[:, 1] == relation, :]

        # Get negative triples
        relation_valid_neg_triples = valid_negatives[
            random.sample(
                range(0, len(valid_negatives)), len(relation_valid_pos_triples)
            ),
            :,
        ]
        relation_test_neg_triples = test_negatives[
            random.sample(
                range(0, len(test_negatives)), len(relation_test_pos_triples)
            ),
            :,
        ]

        # Combine
        relation_valid_triples = torch.vstack(
            [relation_valid_pos_triples, relation_valid_neg_triples]
        )
        relation_test_triples = torch.vstack(
            [relation_test_pos_triples, relation_test_neg_triples]
        )
        relation_valid_trues = torch.vstack(
            [
                torch.full((len(relation_valid_pos_triples),), 1),
                torch.full((len(relation_valid_neg_triples),), 0),
            ]
        )
        relation_test_trues = torch.vstack(
            [
                torch.full((len(relation_test_pos_triples),), 1),
                torch.full((len(relation_test_neg_triples),), 0),
            ]
        )

        # Get scores
        relation_valid_scores = model.score_spo(
            relation_valid_triples[:, 0],
            relation_valid_triples[:, 1],
            relation_valid_triples[:, 2],
            "o",
        ).detach()
        relation_test_scores = model.score_spo(
            relation_test_triples[:, 0],
            relation_test_triples[:, 1],
            relation_test_triples[:, 2],
            "o",
        ).detach()

        ## Calculate metrics

        # AUROC
        auc_roc.append(
            (
                sklearn.metrics.roc_auc_score(
                    relation_valid_trues, relation_valid_scores
                ),
                sklearn.metrics.roc_auc_score(
                    relation_test_trues, relation_test_scores
                ),
            )
        )

        # AUPRC
        relation_valid_prc = sklearn.metrics.precision_recall_curve(
            relation_valid_trues, relation_valid_scores
        )
        relation_test_prc = sklearn.metrics.precision_recall_curve(
            relation_test_trues, relation_test_scores
        )
        auc_prc.append(
            (
                sklearn.metrics.auc(relation_valid_prc[1], relation_valid_prc[0]),
                sklearn.metrics.auc(relation_test_prc[1], relation_test_prc[0]),
            )
        )

        # MAP
        map.append(
            (
                sklearn.metrics.average_precision_score(
                    relation_valid_trues, relation_valid_scores
                ),
                sklearn.metrics.average_precision_score(
                    relation_test_trues, relation_test_scores
                ),
            )
        )

    # Print relation average
    print(
        f"Validation AUROC = {sum([valid_auc_roc for valid_auc_roc, _ in auc_roc]) / dataset._num_relations}"
    )
    print(
        f"Testing AUROC    = {sum([test_auc_roc for _, test_auc_roc in auc_roc]) / dataset._num_relations}"
    )
    print(
        f"Validation AUPRC = {sum([valid_auc_prc for valid_auc_prc, _ in auc_prc]) / dataset._num_relations}"
    )
    print(
        f"Testing AUPRC    = {sum([test_auc_prc for _, test_auc_prc in auc_prc]) / dataset._num_relations}"
    )
    print(
        f"Validation MAP   = {sum([valid_map for valid_map, _ in map]) / dataset._num_relations}"
    )
    print(
        f"Testing MAP      = {sum([test_map for _, test_map in map]) / dataset._num_relations}"
    )


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
