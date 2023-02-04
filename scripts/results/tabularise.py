import argparse
import io
import os

import pandas

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Protein Knowledge Graph Embedding Project: tabularising results"
    )
    parser.add_argument(
        "--type", type=str, default="simple", choices=["simple", "hyperparams"]
    )
    parser.add_argument("--experiment_folder", type=str, required=True)
    parser.add_argument(
        "--keys_file", type=str, default="scripts/results/extract_fields.conf"
    )
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    col_selection = dict(
        simple=[
            "model",
            "train_type",
            "train_loss",
            "fil_hits@10",
            "fil_mrr",
        ],
        hyperparams=[
            "model",
            "train_type",
            "epochs",
            "emb_dim",
            "train_loss",
            "num_negs_o",
            "num_negs_s",
            "fil_hits@10",
            "fil_mrr",
        ],
    )

    # Collect best entries for every experiment
    tables = []
    for experiment in os.listdir(args.experiment_folder):

        # Read table
        try:
            total = pandas.read_csv(
                io.StringIO(
                    os.popen(
                        f"kge dump trace --keysfile {os.path.join(os.getcwd(), args.keys_file)} {os.path.join(args.experiment_folder, experiment)}"
                    ).read()
                )
            )
        except:
            print("Caught exception, continuing...")
            continue

        # Format
        formatted = total.iloc[[pandas.to_numeric(total["fil_mrr"]).idxmax()]][
            col_selection[args.type]
        ]

        # Store
        tables.append(formatted)

    # Concatenated table
    concatenated = pandas.concat(tables)
    if args.output_file:
        concatenated.to_csv(os.path.join(os.getcwd(), args.output_file), index=False)
    else:
        print(concatenated)
