import argparse
import csv
import io
import os
import urllib
import zipfile

import numpy
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


def benchmark(name):

    # Make dataset folder
    dataset_folder = os.path.join(os.getcwd(), name)
    if os.path.exists(dataset_folder):
        print(f"{dataset_folder} exists, please delete/rename first!")
        quit()
    os.mkdir(dataset_folder)

    # Get original biokg dataset via pykeen
    dataset = pykeen.datasets.get_dataset(dataset="biokg")

    # Collect metadata for libkge
    metadata = dict(name=name)

    # Get benchmark via biokg
    with zipfile.ZipFile(
        io.BytesIO(
            urllib.request.urlopen(
                "https://github.com/dsi-bdi/biokg/releases/download/v1.0.0/benchmarks.zip"
            ).read()
        )
    ) as zip_file:
        data = pandas.read_csv(
            io.TextIOWrapper(zip_file.open(name + ".tsv"), newline=""),
            delimiter="\t",
            names=["subject", "predicate", "object"],
        )

    # Id<->entity vocabularies
    entity_to_id = dataset.factory_dict["training"].entity_labeling.label_to_id
    num_entities = len(entity_to_id)

    def map_or_new(entity):
        if entity in entity_to_id:
            return entity_to_id[entity]
        else:
            new = max(entity_to_id.values()) + 1
            entity_to_id[entity] = new
            return new

    # Map subjects
    data["subject"] = data["subject"].map(map_or_new)

    # Map objects
    data["object"] = data["object"].map(map_or_new)

    print(f"Found {len(entity_to_id)-num_entities} new entities.")

    # Save
    id_to_entity = {value: key for key, value in entity_to_id.items()}
    id_to_entity_file = os.path.join(dataset_folder, "entity_ids.del")
    with open(id_to_entity_file, "w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerows(id_to_entity.items())
    metadata["files.entity_ids.filename"] = os.path.basename(id_to_entity_file)
    metadata["files.entity_ids.type"] = "map"
    metadata["num_entities"] = len(id_to_entity)

    # Create new relation<->vocabulary
    predicates = data["predicate"].unique()
    relation_to_id = {
        relation: id for relation, id in zip(predicates, list(range(len(predicates))))
    }
    id_to_relation = {id: relation for relation, id in relation_to_id.items()}
    # Save
    id_to_relation_file = os.path.join(dataset_folder, "relation_ids.del")
    with open(id_to_relation_file, "w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerows(id_to_relation.items())
    metadata["files.relation_ids.filename"] = os.path.basename(id_to_relation_file)
    metadata["files.relation_ids.type"] = "map"
    metadata["num_relations"] = len(id_to_relation)

    # Map relation column
    data["predicate"] = data["predicate"].map(relation_to_id).astype(int)

    # Shuffle and split
    train, valid, test = numpy.split(
        data.sample(frac=1), [int(0.8 * len(data)), int(0.9 * len(data))]
    )

    # Save training data in libkge format
    train_file = os.path.join(dataset_folder, "train.del")
    train.to_csv(train_file, sep="\t", index=False, header=False)
    metadata[f"files.train.filename"] = os.path.basename(train_file)
    metadata[f"files.train.size"] = train.shape[0]
    metadata[f"files.train.split_type"] = "train"
    metadata[f"files.train.type"] = "triples"

    # Save valid data in libkge format
    valid_file = os.path.join(dataset_folder, "valid.del")
    valid.to_csv(valid_file, sep="\t", index=False, header=False)
    metadata[f"files.valid.filename"] = os.path.basename(valid_file)
    metadata[f"files.valid.size"] = len(valid)
    metadata[f"files.valid.split_type"] = "valid"
    metadata[f"files.valid.type"] = "triples"

    # Save test data in libkge format
    test_file = os.path.join(dataset_folder, "test.del")
    test.to_csv(test_file, sep="\t", index=False, header=False)
    metadata[f"files.test.filename"] = os.path.basename(test_file)
    metadata[f"files.test.size"] = len(test)
    metadata[f"files.test.split_type"] = "test"
    metadata[f"files.test.type"] = "triples"

    # Write metadata
    with open(os.path.join(dataset_folder, "dataset.yaml"), "w+") as filename:
        filename.write(yaml.dump(dict(dataset=metadata)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Protein Knowledge Graph Embedding Project: downloading data"
    )
    parser.add_argument("--pykeen", action="store_true", default=False)
    parser.add_argument(
        "--benchmark",
        type=str,
        default=False,
        options=["ddi_efficacy", "ddi_minerals", "dpi_fda", "dep_fda_exp"],
    )
    args = parser.parse_args()

    # Download and split
    if args.pykeen:
        pykeen_split()
    if args.benchmark:
        benchmark(args.benchmark)
