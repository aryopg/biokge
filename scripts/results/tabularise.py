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
            "dataset",
            "model",
            "reciprocal",
            "train_type",
            "train_loss",
            "epochs",
            "lr",
            "emb_regularize_p",
            "emb_dim",
            "split",
            "raw_hits@10",
            "fil_hits@10",
            "fwt_hits@10",
            "raw_mrr",
            "fil_mrr",
            "fwt_mrr",
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
        formatted = total[col_selection[args.type]]

        # Sort on fwt_mrr
        formatted = formatted.sort_values(by="fwt_mrr")

        # Store
        tables.append(formatted)

    # Concatenated table
    concatenated = pandas.concat(tables)
    if args.output_file:
        concatenated.to_csv(os.path.join(os.getcwd(), args.output_file), index=False)
    else:
        print(concatenated)
