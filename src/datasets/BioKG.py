import math
import os

import matplotlib.pyplot
import numpy
import pandas
import torch


class BioKGDataset:
    def __init__(self, config):
        """
        Initialisation

        Args:
        - config (DatasetConfigs): dataset config
        """
        self.name = config.dataset_name
        self.datasets_dir = config.datasets_dir
        self.train_frac = config.train_frac
        self.valid_frac = config.valid_frac
        self.neg_sampling_strat = config.neg_sampling_strat

        # Load unprocessed data into attributes
        for filename in os.listdir(os.path.join(os.getcwd(), self.datasets_dir)):
            if "biokg" in filename:
                data_key = (
                    filename.replace("biokg.", "").replace(".tsv", "").replace(".", "_")
                )
                setattr(
                    self,
                    data_key,
                    pandas.read_csv(
                        os.path.join(os.getcwd(), self.datasets_dir, filename),
                        names=["subject", "relation", "object"],
                        sep="\t",
                    ),
                )

        # Process
        self.process()

    def process(self):
        """
        Process BioKG data:
        - establish and apply vocabularies
        - split into training|valid|test -> exact frac of each relation type
        - generate negative samples
        """

        # Establish relation vocabulary
        relations = self.links["relation"].unique()
        self.relation_voc = {
            relation: idx for relation, idx in zip(relations, range(len(relations)))
        }
        self.reverse_relation_voc = {
            idx: relation for relation, idx in zip(relations, range(len(relations)))
        }
        self.links["relation"] = self.links["relation"].apply(
            lambda relation: self.relation_voc[relation]
        )
        self.num_relations = len(self.relation_voc)

        # Establish entity vocabulary
        entities = pandas.concat([self.links["subject"], self.links["object"]]).unique()
        self.entity_voc = {
            entity: idx for entity, idx in zip(entities, range(len(entities)))
        }
        self.reverse_entity_voc = {
            idx: entity for entity, idx in zip(entities, range(len(entities)))
        }
        self.links["subject"] = self.links["subject"].apply(
            lambda entity: self.entity_voc[entity]
        )
        self.links["object"] = self.links["object"].apply(
            lambda entity: self.entity_voc[entity]
        )
        self.num_entities = len(self.entity_voc)

        # Split and generate negs
        self.train_separated = {}
        self.train = {}
        self.valid = {}
        self.valid_negs = {}
        self.test = {}
        self.test_negs = {}

        for relation in self.links["relation"].unique():
            relation_entries = self.links.loc[self.links["relation"] == relation]

            # Shuffle
            relation_entries = relation_entries.sample(frac=1)

            # Split
            relation_train, relation_valid, relation_test = numpy.split(
                relation_entries,
                [
                    int(self.train_frac * len(relation_entries)),
                    int((self.train_frac + self.valid_frac) * len(relation_entries)),
                ],
            )
            self.num_train_entries = len(relation_train)
            self.num_valid_entries = len(relation_valid)
            self.num_test_entries = len(relation_test)

            # Generate negatives
            n_neg = math.ceil(
                100 / len(relation_valid)
            )  # calculate specific number of negatives to generate, ensuring at least 100
            relation_valid_negs = self.generate_negs(relation_valid, relation, n_neg)
            n_neg = math.ceil(
                100 / len(relation_test)
            )  # calculate specific number of negatives to generate, ensuring at least 100
            relation_test_negs = self.generate_negs(relation_test, relation, n_neg)

            # Store as numpy arrays
            self.train_separated[relation] = relation_train.to_numpy()
            self.valid[relation] = relation_valid.to_numpy()
            self.valid_negs[relation] = relation_valid_negs
            self.test[relation] = relation_test.to_numpy()
            self.test_negs[relation] = relation_test_negs

        # Also store combined train for actual training
        self.train = numpy.vstack(list(self.train_separated.values()))

    def generate_negs(self, triples, relation, n_neg):
        """
        Switch case for negative sampling strategies
        """
        if self.neg_sampling_strat == "uniform":
            return self.generate_negs_uniform(triples, relation, n_neg)
        elif self.neg_sampling_strat == "bernoulli":

            # Calculate Bernoulli probs once
            if not hasattr(self, "bernoulli_probs"):
                hpt = (
                    self.links.groupby(["relation", "object"])
                    .count()
                    .groupby("relation")
                    .mean()
                    .values.squeeze()
                )
                tph = (
                    self.links.groupby(["subject", "relation"])
                    .count()
                    .groupby("relation")
                    .mean()
                    .values.squeeze()
                )
                self.bernoulli_probs = {
                    relation: hpt / (hpt + tph)
                    for relation, hpt, tph in zip(self.relation_voc.values(), hpt, tph)
                }

            return self.generate_negs_bernoulli(triples, relation, n_neg)
        else:
            print("Unknown negative sampling strategy, using uniform.")
            return self.generate_negs_uniform(triples, relation, n_neg)

    def generate_negs_uniform(self, triples, relation, n_neg):
        """
        Uniformly samples negatives for a given set of triples
        -> code after torchkge: https://torchkge.readthedocs.io/en/latest/_modules/torchkge/sampling.html#UniformNegativeSampler

        Args:
        - entries (pandas, Nx3): triples to generate negative samples for
        - relation (int): relation encoding
        - n_neg (int): number of negatives to generate per sample

        Returns:
        - negative samples (numpy, (N,3))
        """

        # Initialise negative subject and object arrays
        neg_subjects = triples["subject"].to_numpy().repeat(n_neg)
        neg_objects = triples["object"].to_numpy().repeat(n_neg)

        # Randomly corrupt subject and/or object
        mask = torch.bernoulli(torch.ones(size=(len(triples) * n_neg,)) / 2).double()
        n_h_cor = int(mask.sum().item())
        neg_subjects[mask == 1] = torch.randint(0, self.num_entities, (n_h_cor,))
        neg_objects[mask == 0] = torch.randint(
            0, self.num_entities, (len(triples) * n_neg - n_h_cor,)
        )

        # Return negative triples
        return numpy.vstack(
            [
                neg_subjects,
                numpy.full((len(neg_subjects),), relation),
                neg_objects,
            ]
        ).T

    def generate_negs_bernoulli(self, triples, relation, n_neg):
        """
        Samples negatives for a given set of triples using the Bernoulli strategy
        -> code after torchkge: https://torchkge.readthedocs.io/en/latest/_modules/torchkge/sampling.html#UniformNegativeSampler

        Args:
        - entries (pandas, Nx3): triples to generate negative samples for
        - relation (int): relation encoding
        - n_neg (int): number of negatives to generate per sample

        Returns:
        - negative samples (numpy, (N,3))
        """

        # Initialise negative subject and object arrays
        neg_subjects = triples["subject"].to_numpy().repeat(n_neg)
        neg_objects = triples["object"].to_numpy().repeat(n_neg)

        # Randomly corrupt subject and/or object
        mask = torch.bernoulli(
            torch.Tensor([self.bernoulli_probs[relation]]).repeat(len(triples) * n_neg)
        ).double()
        n_h_cor = int(mask.sum().item())
        neg_subjects[mask == 1] = torch.randint(0, self.num_entities, (n_h_cor,))
        neg_objects[mask == 0] = torch.randint(
            0, self.num_entities, (len(triples) * n_neg - n_h_cor,)
        )

        # Return negative triples
        return numpy.vstack(
            [
                neg_subjects,
                numpy.full((len(neg_subjects),), relation),
                neg_objects,
            ]
        ).T

    def generate_negs_tensor(self, edge, n_neg):
        """
        Switch case for negative sampling strategies
        Suitable for edge tensors during training
        """
        if self.neg_sampling_strat == "uniform":
            return self.generate_negs_uniform_tensor(edge, n_neg)
        elif self.neg_sampling_strat == "bernoulli":

            # Calculate Bernoulli probs once
            if not hasattr(self, "bernoulli_probs"):
                raise Exception("bernouli probs not calculated during preprocessing but used in training")

            return self.generate_negs_bernoulli_tensor(edge, n_neg)
        else:
            print("Unknown negative sampling strategy, using uniform.")
            return self.generate_negs_uniform_tensor(edge, n_neg)

    def generate_negs_uniform_tensor(self, edge, n_neg):
        """
        Uniformly samples negatives for a given set of triples
        Suitable for edge tensors during training

        Args:
        - edge (pandas, Nx3): triples to generate negative samples for
        - n_neg (int): number of negatives to generate per sample

        Returns:
        - negative samples (numpy, (N,3))
        """

        # Initialise negative subject and object arrays
        neg_subjects = edge[0].repeat(n_neg)
        neg_objects = edge[2].repeat(n_neg)
        neg_relationships = edge[1].repeat(n_neg)

        # Randomly corrupt subject and/or object
        mask = torch.bernoulli(torch.ones(size=(len(edge[2]) * n_neg,)).to(edge[1].device) / 2).double()
        n_h_cor = int(mask.sum().item())
        neg_subjects[mask == 1] = torch.randint(0, self.num_entities, (n_h_cor,), device=edge[1].device)

        neg_objects[mask == 0] = torch.randint(
            0, self.num_entities, (len(edge[0]) * n_neg - n_h_cor,), device=edge[1].device
        )

        # Return negative triples
        return torch.stack(
            [
                neg_subjects,
                neg_relationships,
                neg_objects,
            ]
        )

    def generate_negs_bernoulli_tensor(self, edge, n_neg):
        """
        Samples negatives for a given set of triples using the Bernoulli strategy
        Suitable for edge tensors during training

        Args:
        - edge (pandas, Nx3): triples to generate negative samples for
        - n_neg (int): number of negatives to generate per sample

        Returns:
        - negative samples (numpy, (N,3))
        """

        # Initialise negative subject and object arrays
        neg_subjects = edge[0].repeat(n_neg)
        neg_objects = edge[2].repeat(n_neg)
        neg_relationships = edge[1].repeat(n_neg)

        # Randomly corrupt subject and/or object
        mask = torch.bernoulli(neg_relationships.double().cpu().apply_(lambda val: self.bernoulli_probs.get(val)).to(edge[1].device)).int()

        n_h_cor = int(mask.sum().item())
        neg_subjects[mask == 1] = torch.randint(0, self.num_entities, (n_h_cor,), device=edge[1].device)

        neg_objects[mask == 0] = torch.randint(
            0, self.num_entities, (len(edge[0]) * n_neg - n_h_cor,), device=edge[1].device
        )

        # Return negative triples
        return torch.stack(
            [
                neg_subjects,
                neg_relationships,
                neg_objects,
            ]
        )

    def plot_statistics(self, save_location):
        """
        Plot pos/neg/false neg count per relation

        Args:
        - save_location (path): path used to store fig pdf
        """

        # Positives
        pos_counts = self.links["relation"].value_counts()
        pos_counts = {
            relation: count
            for relation, count in zip(pos_counts.index.values, pos_counts.values)
        }

        # Validation negatives
        valid_negs_counts = dict(
            sorted(
                {
                    relation: len(negs) for relation, negs in self.valid_negs.items()
                }.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        # False validation negatives
        valid_false_negs_counts = {
            relation: len(
                set(
                    [
                        tuple(x)
                        for x in self.links[
                            self.links["relation"] == relation
                        ].to_numpy()
                    ]
                )
                & set([tuple(x) for x in self.valid_negs[relation]])
            )
            for relation in valid_negs_counts.keys()
        }

        # Test negatives
        test_negs_counts = dict(
            sorted(
                {
                    relation: len(negs) for relation, negs in self.test_negs.items()
                }.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        # False test negatives
        test_false_negs_counts = {
            relation: len(
                set(
                    [
                        tuple(x)
                        for x in self.links[
                            self.links["relation"] == relation
                        ].to_numpy()
                    ]
                )
                & set([tuple(x) for x in self.test_negs[relation]])
            )
            for relation in test_negs_counts.keys()
        }

        ### Plot
        fig, ax = matplotlib.pyplot.subplots()
        x = numpy.arange(len(self.relation_voc.keys()))

        # Positives
        ax.bar(x, pos_counts.values(), width=0.25, color="r", label="P")

        # Validation negatives
        ax.bar(
            x + 0.25,
            valid_negs_counts.values(),
            width=0.25,
            color="seagreen",
            label="N valid",
        )
        ax.bar(
            x + 0.25,
            valid_false_negs_counts.values(),
            width=0.25,
            color="lime",
            label="FN valid",
        )

        # Test negatives
        ax.bar(
            x + 0.25 * 2,
            test_negs_counts.values(),
            width=0.25,
            color="lightsteelblue",
            label="N test",
        )
        ax.bar(
            x + 0.25 * 2,
            test_false_negs_counts.values(),
            width=0.25,
            color="blue",
            label="FN test",
        )

        ax.legend()
        matplotlib.pyplot.xlabel("Relation")
        matplotlib.pyplot.xticks(
            x + 0.25,
            [self.reverse_relation_voc[relation] for relation in pos_counts.keys()],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        matplotlib.pyplot.ylabel("Count (log)")
        matplotlib.pyplot.yscale("log")

        # Save
        fig.savefig(
            save_location,
            format="pdf",
            bbox_inches="tight",
        )

    def plot_neg_counts(self, save_location, type="valid"):
        """
        Plot negative count per relation

        Args:
        - save_location (path): path used to store fig pdf
        - type (str): valid or test, which negs to plot
        """
        negs_per_relation = getattr(self, type + "_negs")
        counts = [len(negs) for negs in negs_per_relation.values()]

        # Plot
        fig, ax = matplotlib.pyplot.subplots()
        ax.bar(
            [
                self.reverse_relation_voc[relation]
                for relation in negs_per_relation.keys()
            ],
            counts,
        )
        upper_bound = 60e3
        matplotlib.pyplot.xlabel("Relation")
        matplotlib.pyplot.xticks(rotation=45, ha="right", rotation_mode="anchor")
        matplotlib.pyplot.ylabel("Count")
        matplotlib.pyplot.ylim(0, upper_bound)
        matplotlib.pyplot.title("Negatives")

        # Add counts in bars
        for idx, relation in enumerate(counts.index):
            matplotlib.pyplot.text(
                idx, upper_bound / 2, counts[relation], rotation=90, ha="center"
            )

        # Save
        fig.savefig(
            save_location,
            format="pdf",
            bbox_inches="tight",
        )

    def __getitem__(self):
        return 0

    def __len__(self):
        return 0
