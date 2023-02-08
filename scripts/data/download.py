import argparse
import csv
import io
import itertools
import os
import urllib
import zipfile

import pandas
import pykeen.datasets
import yaml


def save_id_entity_vocs(dataset, dataset_folder, metadata):
    entity_to_id = dataset.factory_dict["training"].entity_labeling.label_to_id
    id_to_entity = {value: key for key, value in entity_to_id.items()}
    # Save
    id_to_entity_file = os.path.join(dataset_folder, "entity_ids.del")
    with open(id_to_entity_file, "w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerows(id_to_entity.items())
    metadata["files.entity_ids.filename"] = os.path.basename(id_to_entity_file)
    metadata["files.entity_ids.type"] = "map"
    metadata["num_entities"] = len(id_to_entity)

    return entity_to_id, id_to_entity


def save_id_relation_vocs(dataset, dataset_folder, metadata):
    relation_to_id = dataset.factory_dict["training"].relation_labeling.label_to_id
    id_to_relation = {value: key for key, value in relation_to_id.items()}
    # Save
    id_to_relation_file = os.path.join(dataset_folder, "relation_ids.del")
    with open(id_to_relation_file, "w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerows(id_to_relation.items())
    metadata["files.relation_ids.filename"] = os.path.basename(id_to_relation_file)
    metadata["files.relation_ids.type"] = "map"
    metadata["num_relations"] = len(id_to_relation)

    return relation_to_id, id_to_relation


def get_splits(dataset):
    return {
        split_name.replace("ing", "").replace("ation", ""): pandas.DataFrame(
            dataset.factory_dict[split_name].mapped_triples.numpy(),
            columns=["head", "relation", "tail"],
        )
        for split_name in ["training", "validation", "testing"]
    }


def pykeen_split():
    name = "biokg"

    # Get dataset via pykeen
    dataset = pykeen.datasets.get_dataset(dataset="biokg")

    # Collect metadata for libkge
    metadata = dict(name=name)

    # Make dataset folder
    dataset_folder = os.path.join(os.getcwd(), name)
    if os.path.exists(dataset_folder):
        print(f"{dataset_folder} exists, please delete/rename first!")
        quit()
    os.mkdir(dataset_folder)

    # Id<->entity vocabularies
    entity_to_id, id_to_entity = save_id_entity_vocs(dataset, dataset_folder, metadata)

    # Id<->relation vocabularies
    relation_to_id, id_to_relation = save_id_relation_vocs(
        dataset, dataset_folder, metadata
    )

    # Get and save splits
    for split_name, split in get_splits(dataset).items():
        split_file = os.path.join(dataset_folder, split_name + ".del")
        split.to_csv(split_file, sep="\t", header=False, index=False)
        metadata[f"files.{split_name}.filename"] = os.path.basename(split_file)
        metadata[f"files.{split_name}.size"] = len(split)
        metadata[f"files.{split_name}.split_type"] = split_name
        metadata[f"files.{split_name}.type"] = "triples"

    # Write metadata
    with open(os.path.join(dataset_folder, "dataset.yaml"), "w+") as filename:
        filename.write(yaml.dump(dict(dataset=metadata)))


def pykeen_inverses_split():
    name = "biokg_inverses"

    # Get dataset via pykeen
    dataset = pykeen.datasets.get_dataset(dataset="biokg")

    # Collect metadata for libkge
    metadata = dict(name=name)

    # Make dataset folder
    dataset_folder = os.path.join(os.getcwd(), name)
    if os.path.exists(dataset_folder):
        print(f"{dataset_folder} exists, please delete/rename first!")
        quit()
    os.mkdir(dataset_folder)

    # Id<->entity vocabularies
    save_id_entity_vocs(dataset, dataset_folder, metadata)

    # Id<->relation vocabularies
    relation_to_id, _ = save_id_relation_vocs(dataset, dataset_folder, metadata)

    # Get splits
    splits = get_splits(dataset)

    # Add inverses
    splits = {
        split_name: pandas.concat(
            [
                split,
                split.loc[
                    split["relation"].isin(
                        [
                            relation_to_id[relation]
                            for relation in [
                                "DDI",
                                "PPI",
                                "DISEASE_PATHWAY_ASSOCIATION",
                                "DPI",
                                "DRUG_DISEASE_ASSOCIATION",
                                "DRUG_PATHWAY_ASSOCIATION",
                                "PROTEIN_DISEASE_ASSOCIATION",
                                "PROTEIN_PATHWAY_ASSOCIATION",
                            ]
                        ]  # only for these relations
                    )
                ].rename(columns={"head": "tail", "tail": "head"}),
            ]
        )
        for split_name, split in splits.items()
    }

    # Check overlap
    for split_a, split_b in itertools.combinations(splits.values(), r=2):
        assert len(pandas.merge(split_a, split_b, on=["head", "relation", "tail"])) == 0

    # Save
    for split_name, split in splits.items():
        split_file = os.path.join(dataset_folder, split_name + ".del")
        split.to_csv(split_file, sep="\t", header=False, index=False)
        metadata[f"files.{split_name}.filename"] = os.path.basename(split_file)
        metadata[f"files.{split_name}.size"] = len(split)
        metadata[f"files.{split_name}.split_type"] = split_name
        metadata[f"files.{split_name}.type"] = "triples"

    # Write metadata
    with open(os.path.join(dataset_folder, "dataset.yaml"), "w+") as filename:
        filename.write(yaml.dump(dict(dataset=metadata)))


def no_split():
    name = "biokg_no_split"

    # Make dataset folder
    dataset_folder = os.path.join(os.getcwd(), name)
    if os.path.exists(dataset_folder):
        print(f"{dataset_folder} exists, please delete/rename first!")
        quit()
    os.mkdir(dataset_folder)

    # Get dataset via pykeen
    dataset = pykeen.datasets.get_dataset(dataset="biokg")

    # Collect metadata for libkge
    metadata = dict(name=name)

    # Id<->entity vocabularies
    entity_to_id, _ = save_id_entity_vocs(dataset, dataset_folder, metadata)

    # Id<->relation vocabularies
    relation_to_id, _ = save_id_relation_vocs(dataset, dataset_folder, metadata)

    # Get Biokg as a whole
    train_data = pandas.concat(get_splits(dataset).values())

    # Add inverses
    train_data = pandas.concat(
        [
            train_data,
            train_data.loc[
                train_data["relation"].isin(
                    [
                        relation_to_id[relation] for relation in ["DDI", "PPI"]
                    ]  # only for these relations
                )
            ].rename(columns={"head": "tail", "tail": "head"}),
        ]
    )

    # Save training data in libkge format
    train_file = os.path.join(dataset_folder, "train.del")
    train_data.to_csv(train_file, sep="\t", index=False, header=False)
    metadata[f"files.train.filename"] = os.path.basename(train_file)
    metadata[f"files.train.size"] = train_data.shape[0]
    metadata[f"files.train.split_type"] = "train"
    metadata[f"files.train.type"] = "triples"

    # Get benchmarks via biokg
    valid_data = "ddi_efficacy.tsv"
    test_data = "ddi_minerals.tsv"
    with zipfile.ZipFile(
        io.BytesIO(
            urllib.request.urlopen(
                "https://github.com/dsi-bdi/biokg/releases/download/v1.0.0/benchmarks.zip"
            ).read()
        )
    ) as zip_file:
        valid_data = pandas.read_csv(
            io.TextIOWrapper(zip_file.open(valid_data), newline=""),
            delimiter="\t",
            names=["subject", "predicate", "object"],
        )
        test_data = pandas.read_csv(
            io.TextIOWrapper(zip_file.open(test_data), newline=""),
            delimiter="\t",
            names=["subject", "predicate", "object"],
        )

    # Overwrite relation column
    valid_data["predicate"] = relation_to_id["DDI"]
    test_data["predicate"] = relation_to_id["DDI"]

    # Map entity columns
    valid_data.replace({"subject": entity_to_id, "object": entity_to_id}, inplace=True)
    test_data.replace({"subject": entity_to_id, "object": entity_to_id}, inplace=True)

    # Save valid data in libkge format
    valid_file = os.path.join(dataset_folder, "valid.del")
    valid_data.to_csv(valid_file, sep="\t", index=False, header=False)
    metadata[f"files.valid.filename"] = os.path.basename(valid_file)
    metadata[f"files.valid.size"] = len(valid_data)
    metadata[f"files.valid.split_type"] = "valid"
    metadata[f"files.valid.type"] = "triples"

    # Save test data in libkge format
    test_file = os.path.join(dataset_folder, "test.del")
    test_data.to_csv(test_file, sep="\t", index=False, header=False)
    metadata[f"files.test.filename"] = os.path.basename(test_file)
    metadata[f"files.test.size"] = len(test_data)
    metadata[f"files.test.split_type"] = "test"
    metadata[f"files.test.type"] = "triples"

    # Write metadata
    with open(os.path.join(dataset_folder, "dataset.yaml"), "w+") as filename:
        filename.write(yaml.dump(dict(dataset=metadata)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Protein Knowledge Graph Embedding Project: downloading data"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="simple",
        choices=["pykeen", "inverses", "no_split"],
    )
    args = parser.parse_args()

    # Download and split
    if args.type == "pykeen":
        pykeen_split()
    elif args.type == "inverses":
        pykeen_inverses_split()
    elif args.type == "no_split":
        no_split()
