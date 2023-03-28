import os

import numpy
import pandas


class BenchmarkLibKGEDataset:
    def __init__(
        self, config, task, from_scratch, multilabel: bool, negative_samples_ratio: int
    ):
        """
        Initialisation

        Args:
        - config (DatasetConfigs): dataset config
        """
        self.datasets_dir = config.datasets_dir
        self.train_frac = config.train_frac
        self.valid_frac = config.valid_frac

        self.from_scratch = from_scratch
        self.multilabel = multilabel
        self.negative_samples_ratio = negative_samples_ratio

        self._load_datasets(task)
        self._load_entity_vocab(task)

        # Process
        self._process_datasets()

    def _load_datasets(self, task):
        self.train_dataset = pandas.read_csv(
            os.path.join(self.datasets_dir, task, "train.del"),
            names=["subject", "relation", "object"],
            sep="\t",
        )
        self.valid_dataset = pandas.read_csv(
            os.path.join(self.datasets_dir, task, "valid.del"),
            names=["subject", "relation", "object"],
            sep="\t",
        )
        self.test_dataset = pandas.read_csv(
            os.path.join(self.datasets_dir, task, "test.del"),
            names=["subject", "relation", "object"],
            sep="\t",
        )

    def _load_entity_vocab(self, task):
        id_entity = pandas.read_csv(
            os.path.join(self.datasets_dir, task, "entity_ids.del"),
            names=["idx", "entity"],
            sep="\t",
        )
        self.entity_voc = {
            entity: idx for idx, entity in zip(id_entity.idx, id_entity.entity)
        }
        self.reverse_entity_voc = {
            idx: entity for entity, idx in self.entity_voc.items()
        }

    def _process_datasets(self):
        # Establish relation vocabulary
        relations = list(
            set(
                list(self.train_dataset["relation"].unique())
                + list(self.valid_dataset["relation"].unique())
                + list(self.test_dataset["relation"].unique())
            )
        )
        all_data = pandas.concat(
            [self.train_dataset, self.valid_dataset, self.test_dataset]
        )
        all_entities = pandas.concat([all_data["subject"], all_data["object"]]).unique()

        small_entity_voc = {
            entity: idx for entity, idx in zip(all_entities, range(len(all_entities)))
        }

        self.relation_voc = {
            relation: idx for relation, idx in zip(relations, range(len(relations)))
        }
        self.reverse_relation_voc = {
            idx: relation for relation, idx in zip(relations, range(len(relations)))
        }
        self.train_dataset["relation"] = self.train_dataset["relation"].apply(
            lambda relation: self.relation_voc[relation]
        )
        self.valid_dataset["relation"] = self.valid_dataset["relation"].apply(
            lambda relation: self.relation_voc[relation]
        )
        self.test_dataset["relation"] = self.test_dataset["relation"].apply(
            lambda relation: self.relation_voc[relation]
        )

        if self.from_scratch:
            self.train_dataset["subject"] = self.train_dataset["subject"].apply(
                lambda entity: small_entity_voc[entity]
            )
            self.valid_dataset["subject"] = self.valid_dataset["subject"].apply(
                lambda entity: small_entity_voc[entity]
            )
            self.test_dataset["subject"] = self.test_dataset["subject"].apply(
                lambda entity: small_entity_voc[entity]
            )

            self.train_dataset["object"] = self.train_dataset["object"].apply(
                lambda entity: small_entity_voc[entity]
            )
            self.valid_dataset["object"] = self.valid_dataset["object"].apply(
                lambda entity: small_entity_voc[entity]
            )
            self.test_dataset["object"] = self.test_dataset["object"].apply(
                lambda entity: small_entity_voc[entity]
            )

        self.num_relations = (
            len(self.relation_voc)
            if self.negative_samples_ratio == 0
            else len(self.relation_voc) + 1
        )

        self.num_entities = len(
            set(self.train_dataset["subject"].values)
            | set(self.train_dataset["object"].values)
            | set(self.valid_dataset["subject"].values)
            | set(self.valid_dataset["object"].values)
            | set(self.test_dataset["subject"].values)
            | set(self.test_dataset["object"].values)
        )

        so_pairs = []
        for subject, object in self.train_dataset[["subject", "object"]].values:
            so_pairs += [(subject, object)]
        for subject, object in self.valid_dataset[["subject", "object"]].values:
            so_pairs += [(subject, object)]
        for subject, object in self.test_dataset[["subject", "object"]].values:
            so_pairs += [(subject, object)]

        if self.multilabel:
            self.train_inputs = numpy.array(
                self.train_dataset[["subject", "object"]].values
            )
            self.valid_inputs = numpy.array(
                self.valid_dataset[["subject", "object"]].values
            )
            self.test_inputs = numpy.array(
                self.test_dataset[["subject", "object"]].values
            )

            if self.num_relations > 1:
                self.train_labels = numpy.array(self.train_dataset["relation"].values)
                self.valid_labels = numpy.array(self.valid_dataset["relation"].values)
                self.test_labels = numpy.array(self.test_dataset["relation"].values)

                self.train_inputs, self.train_labels = self.create_multilabel(
                    self.train_inputs, self.train_labels
                )
                self.valid_inputs, self.valid_labels = self.create_multilabel(
                    self.valid_inputs, self.valid_labels
                )
                self.test_inputs, self.test_labels = self.create_multilabel(
                    self.test_inputs, self.test_labels
                )
            else:
                self.train_labels = numpy.array(
                    [1] * len(self.train_dataset["relation"])
                )
                self.valid_labels = numpy.array(
                    [1] * len(self.valid_dataset["relation"].values)
                )
                self.test_labels = numpy.array(
                    [1] * len(self.test_dataset["relation"].values)
                )

            self.num_positive_train_samples = len(self.train_labels)
            self.num_positive_valid_samples = len(self.valid_labels)
            self.num_positive_test_samples = len(self.test_labels)

            self.num_negative_train_samples = int(
                self.num_positive_train_samples * self.negative_samples_ratio
            )
            self.num_negative_valid_samples = int(
                self.num_positive_valid_samples * self.negative_samples_ratio
            )
            self.num_negative_test_samples = int(
                self.num_positive_test_samples * self.negative_samples_ratio
            )

            if self.negative_samples_ratio > 0:
                (
                    self.train_negative_inputs,
                    self.train_negative_labels,
                ) = self.generate_negative_samples(
                    self.train_dataset,
                    so_pairs,
                    self.num_negative_train_samples,
                )
                (
                    self.valid_negative_inputs,
                    self.valid_negative_labels,
                ) = self.generate_negative_samples(
                    self.valid_dataset,
                    so_pairs,
                    self.num_negative_valid_samples,
                )
                (
                    self.test_negative_inputs,
                    self.test_negative_labels,
                ) = self.generate_negative_samples(
                    self.test_dataset,
                    so_pairs,
                    self.num_negative_test_samples,
                )

                self.train_inputs = numpy.vstack(
                    (self.train_inputs, self.train_negative_inputs)
                )
                self.valid_inputs = numpy.vstack(
                    (self.valid_inputs, self.valid_negative_inputs)
                )
                self.test_inputs = numpy.vstack(
                    (self.test_inputs, self.test_negative_inputs)
                )
                if self.num_relations > 1:
                    self.train_labels = numpy.hstack(
                        (numpy.zeros((len(self.train_labels), 1)), self.train_labels)
                    )
                    self.valid_labels = numpy.hstack(
                        (numpy.zeros((len(self.valid_labels), 1)), self.valid_labels)
                    )
                    self.test_labels = numpy.hstack(
                        (numpy.zeros((len(self.test_labels), 1)), self.test_labels)
                    )
                    self.train_labels = numpy.vstack(
                        (self.train_labels, self.train_negative_labels)
                    )
                    self.valid_labels = numpy.vstack(
                        (self.valid_labels, self.valid_negative_labels)
                    )
                    self.test_labels = numpy.vstack(
                        (self.test_labels, self.test_negative_labels)
                    )
                    print(self.test_labels.shape)
                else:
                    self.train_labels = numpy.hstack(
                        (self.train_labels, self.train_negative_labels)
                    )
                    self.valid_labels = numpy.hstack(
                        (self.valid_labels, self.valid_negative_labels)
                    )
                    self.test_labels = numpy.hstack(
                        (self.test_labels, self.test_negative_labels)
                    )
        else:
            raise NotImplementedError

    def create_multilabel(self, inputs, labels):
        one_hot_data = {}
        for input, label in zip(inputs, labels):
            so_key = (input[0], input[1])

            if so_key not in one_hot_data:
                one_hot_data[so_key] = numpy.zeros(shape=(self.num_relations,))

            one_hot_data[so_key][label] = 1

        return numpy.array(list(one_hot_data.keys())), numpy.array(
            list(one_hot_data.values())
        )

    def generate_negative_samples(self, dataset, so_pairs, num_negatives):
        entities = set(dataset["subject"].values) | set(dataset["object"].values)

        adjacency_matrix = {}  # subjects as keys and lists of objects as values
        for so_pair in so_pairs:
            if so_pair[0] not in adjacency_matrix:
                adjacency_matrix[so_pair[0]] = []

            adjacency_matrix[so_pair[0]] += [so_pair[1]]

        negative_samples = []
        for subject, objects in adjacency_matrix.items():
            for object in list(entities - set(objects)):
                negative_samples += [[subject, object]]

        negative_samples = negative_samples[:num_negatives]
        if self.num_relations > 1:
            negative_labels = numpy.zeros(
                (len(negative_samples), self.num_relations + 1)
            )
            negative_labels[:, 0] = numpy.ones((len(negative_samples),))
        else:
            negative_labels = numpy.zeros((len(negative_samples),))

        return numpy.array(negative_samples), negative_labels
