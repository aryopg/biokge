import argparse

import torch

import kge
import kge.model


def evaluate(model_path):

    model = kge.model.KgeModel.create_from(kge.util.io.load_checkpoint(model_path))
    valid_triples = model.dataset.load_triples("valid")
    test_triples = model.dataset.load_triples("test")
    print(test_triples[0])

    print(model.score_spo(valid_triples[0], valid_triples[1], valid_triples[2], "o"))

    return 0


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
