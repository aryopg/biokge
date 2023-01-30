import os
import shutil

import pandas
import pykeen.datasets
import yaml


def main(libkge_path):

    # Get dataset via pykeen
    dataset = pykeen.datasets.get_dataset(dataset="biokg")

    # Collect metadata for libkge
    metadata = dict(name="biokg")

    # For every split
    for data_split in ["training", "validation", "testing"]:
        data_split_short = data_split.replace("ing", "").replace("ation", "")

        # Save to libkge folder
        dataset.factory_dict[data_split].to_path_binary(
            os.path.join(libkge_path, data_split)
        )

        # Open triples
        triples = pandas.read_csv(
            os.path.join(libkge_path, data_split, "numeric_triples.tsv.gz"),
            compression="gzip",
            sep="\t",
        )

        # Save in libkge format
        split_filename = os.path.join(libkge_path, data_split_short + ".del")
        triples.to_csv(
            split_filename,
            sep="\t",
            header=False,
            index=False,
        )
        metadata[f"files.{data_split_short}.filename"] = os.path.basename(
            split_filename
        )
        metadata[f"files.{data_split_short}.size"] = len(triples)
        metadata[f"files.{data_split_short}.split_type"] = data_split_short
        metadata[f"files.{data_split_short}.type"] = "triples"

        if data_split == "training":

            # Open entity-id voc
            entity_to_id = pandas.read_csv(
                os.path.join(libkge_path, data_split, "entity_to_id.tsv.gz"),
                compression="gzip",
                sep="\t",
            )

            # Save in libkge format
            entity_to_id_filename = os.path.join(libkge_path, "entity_ids.del")
            entity_to_id.to_csv(
                entity_to_id_filename,
                sep="\t",
                header=False,
                index=False,
            )
            metadata["files.entity_ids.filename"] = os.path.basename(
                entity_to_id_filename
            )
            metadata["files.entity_ids.type"] = "map"
            metadata["num_entities"] = len(entity_to_id)

            # Open relation-id voc
            relation_to_id = pandas.read_csv(
                os.path.join(libkge_path, data_split, "relation_to_id.tsv.gz"),
                compression="gzip",
                sep="\t",
            )

            # Save in libkge format
            relation_to_id_filename = os.path.join(libkge_path, "relation_ids.del")
            relation_to_id.to_csv(
                relation_to_id_filename,
                sep="\t",
                header=False,
                index=False,
            )
            metadata["files.relation_ids.filename"] = os.path.basename(
                relation_to_id_filename
            )
            metadata["files.relation_ids.type"] = "map"
            metadata["num_relations"] = len(relation_to_id)

        # Cleanup
        shutil.rmtree(os.path.join(libkge_path, data_split))

    # Write metadata
    with open(os.path.join(libkge_path, "dataset.yaml"), "w+") as filename:
        filename.write(yaml.dump(dict(dataset=metadata)))


if __name__ == "__main__":

    # Assumes kge is in cwd
    libkge_path = os.path.join(os.getcwd(), "kge/data/biokg")

    # Assumes kge/data/biokg does not exist
    if os.path.exists(libkge_path):
        print(f"{libkge_path} already exists, delete first!")
        quit()

    main(libkge_path)
