import argparse
import io
import os

import pandas

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Protein Knowledge Graph Embedding Project: tabularising results"
    )
    parser.add_argument("--type", type=str, default="simple", choices=["simple"])
    parser.add_argument("--experiment_folder", type=str, required=True)
    parser.add_argument(
        "--keys_file", type=str, default="scripts/results/extract_fields.conf"
    )
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    col_selection = dict(
        simple=[
            "train_type",
            "model",
            "train_loss",
            "fil_hits@10",
            "fil_mrr",
        ]
    )

    # Collect best entries for every experiment
    tables = []
    for experiment in os.listdir(args.experiment_folder):

        # Read table
        total = pandas.read_csv(
            io.StringIO(
                os.popen(
                    f"kge dump trace --search --keysfile {os.path.join(os.getcwd(), args.keys_file)} {os.path.join(args.experiment_folder, experiment)}"
                ).read()
            )
        )

        # Format
        formatted = total.iloc[[pandas.to_numeric(total["fil_mrr"]).idxmax()]][
            col_selection[args.type]
        ]

        # Store
        tables.append(formatted)

    # Save concatenated table
    pandas.concat(tables).to_csv(
        os.path.join(os.getcwd(), args.output_file), index=False
    )
