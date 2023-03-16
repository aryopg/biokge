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

    def get_metrics_per_relation(relation):
        valid_metrics = {}
        test_metrics = {}

        def get_adjacency_matrix(triples):
            matrix = numpy.zeros(
                (dataset._num_entities, dataset._num_entities),
                dtype=bool,
            )
            matrix[
                triples[triples[:, 1] == relation, 0],
                triples[triples[:, 1] == relation, 2],
            ] = 1
            return matrix

        # Create negative samples
        def get_negatives(triples, others_triples, others_negatives, ratio=1):
            adjacency_matrix = get_adjacency_matrix(triples)
            adjacency_others = [
                get_adjacency_matrix(other_triples) for other_triples in others_triples
            ]

            count = 0
            negatives = set()
            while count != int(len(triples[:, 1] == relation) * ratio):
                subject = random.randrange(dataset._num_entities)
                object = random.randrange(dataset._num_entities)
                other_relation = random.randrange(dataset._num_relations)
                if (
                    not adjacency_matrix[subject, object]
                    and all(
                        [
                            not adjacency_other[subject, object]
                            for adjacency_other in adjacency_others
                        ]
                    )
                    and (subject, other_relation, object) not in negatives
                    and all(
                        [
                            (subject, other_relation, object) not in other_negatives
                            for other_negatives in others_negatives
                        ]
                    )
                ):
                    negatives.add((subject, other_relation, object))
                    count += 1
            return negatives

        valid_negatives = get_negatives(
            valid_triples, [train_triples, test_triples], []
        )
        test_negatives = get_negatives(
            test_triples,
            [train_triples, valid_triples],
            [valid_negatives],
        )

        # Stack
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

        # Save
        numpy.savetxt(
            "valid_negatives.del",
            valid_negatives,
            fmt="%i",
            delimiter="\t",
        )
        numpy.savetxt(
            "test_negatives.del",
            test_negatives,
            fmt="%i",
            delimiter="\t",
        )

        # Combine
        relation_valid_triples = torch.vstack([valid_triples, valid_negatives])
        relation_test_triples = torch.vstack([test_triples, test_negatives])
        relation_valid_trues = torch.concatenate(
            [
                torch.full((len(valid_triples),), 1),
                torch.full((len(valid_negatives),), 0),
            ]
        )
        relation_test_trues = torch.concatenate(
            [
                torch.full((len(test_triples),), 1),
                torch.full((len(test_negatives),), 0),
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
        valid_metrics["auc_roc"] = sklearn.metrics.roc_auc_score(
            relation_valid_trues, relation_valid_scores
        )
        test_metrics["auc_roc"] = sklearn.metrics.roc_auc_score(
            relation_test_trues, relation_test_scores
        )

        # AUPRC
        relation_valid_prc = sklearn.metrics.precision_recall_curve(
            relation_valid_trues, relation_valid_scores
        )
        relation_test_prc = sklearn.metrics.precision_recall_curve(
            relation_test_trues, relation_test_scores
        )
        valid_metrics["auc_prc"] = sklearn.metrics.auc(
            relation_valid_prc[1], relation_valid_prc[0]
        )
        test_metrics["auc_prc"] = sklearn.metrics.auc(
            relation_test_prc[1], relation_test_prc[0]
        )

        # MAP
        valid_metrics["map"] = sklearn.metrics.average_precision_score(
            relation_valid_trues, relation_valid_scores
        )
        test_metrics["map"] = sklearn.metrics.average_precision_score(
            relation_test_trues, relation_test_scores
        )

        return valid_metrics, test_metrics

    # Get metrics per relation
    metrics = [
        get_metrics_per_relation(relation) for relation in range(dataset._num_relations)
    ]

    # Print relation average
    print(
        f"Validation AUROC = {sum([valid_metrics['auc_roc'] for valid_metrics, _  in metrics]) / dataset._num_relations}"
    )
    print(
        f"Testing AUROC    = {sum([test_metrics['auc_roc'] for _, test_metrics in metrics]) / dataset._num_relations}"
    )
    print(
        f"Validation AUPRC = {sum([valid_metrics['auc_prc'] for valid_metrics, _ in metrics]) / dataset._num_relations}"
    )
    print(
        f"Testing AUPRC    = {sum([test_metrics['auc_prc'] for _, test_metrics in metrics]) / dataset._num_relations}"
    )
    print(
        f"Validation MAP   = {sum([valid_metrics['map'] for valid_metrics, _ in metrics]) / dataset._num_relations}"
    )
    print(
        f"Testing MAP      = {sum([test_metrics['map'] for _, test_metrics in metrics]) / dataset._num_relations}"
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
