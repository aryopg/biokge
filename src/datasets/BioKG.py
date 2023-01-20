import os
import pandas
import math
import numpy
import matplotlib.pyplot
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
        self.links["relation"] = self.links["relation"].apply(
            lambda relation: self.relation_voc[relation]
        )
        self.num_relations = len(self.relation_voc)

        # Establish entity vocabulary
        entities = pandas.concat([self.links["subject"], self.links["object"]]).unique()
        self.entity_voc = {
            entity: idx for entity, idx in zip(entities, range(len(entities)))
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
            relation_valid_negs = self.generate_negs(relation_valid, relation)
            relation_test_negs = self.generate_negs(relation_test, relation)

            # Store as numpy arrays
            self.train_separated[relation] = relation_train.to_numpy()
            self.valid[relation] = relation_valid.to_numpy()
            self.valid_negs[relation] = relation_valid_negs
            self.test[relation] = relation_test.to_numpy()
            self.test_negs[relation] = relation_test_negs

        # Also store combined train for actual training
        self.train = numpy.vstack(list(self.train_separated.values()))

    def generate_negs(self, triples, relation):
        """
        Switch case for negative sampling strategies
        """
        if self.neg_sampling_strat == "uniform":
            return self.generate_negs_uniform(triples, relation)
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

            return self.generate_negs_bernoulli(triples, relation)
        else:
            print("Unknown negative sampling strategy, using uniform.")
            return self.generate_negs_uniform(triples, relation)

    def generate_negs_uniform(self, triples, relation):
        """
        Uniformly samples negatives for a given set of triples
        -> code after torchkge: https://torchkge.readthedocs.io/en/latest/_modules/torchkge/sampling.html#UniformNegativeSampler

        Args:
        - entries (pandas, Nx3): triples to generate negative samples for
        - relation (int): relation encoding

        Returns:
        - negative samples (numpy, (N,3))
        """
        n_neg = math.ceil(
            100 / len(triples)
        )  # calculate specific number of negatives to generate, ensuring at least 100

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

    def generate_negs_bernoulli(self, triples, relation):
        """
        Samples negatives for a given set of triples using the Bernoulli strategy
        -> code after torchkge: https://torchkge.readthedocs.io/en/latest/_modules/torchkge/sampling.html#UniformNegativeSampler

        Args:
        - entries (pandas, Nx3): triples to generate negative samples for
        - relation (int): relation encoding

        Returns:
        - negative samples (numpy, (N,3))
        """

        n_neg = math.ceil(
            100 / len(triples)
        )  # calculate specific number of negatives to generate, ensuring at least 100

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

    def plot_edge_distribution(self, save=False):
        """
        Plot relation type distribution

        - save (string): path to figure if saving, starts in cwd
        """
        counts = self.links["relation"].value_counts()

        # Plot
        fig, ax = matplotlib.pyplot.subplots()
        counts.plot(ax=ax, kind="bar")
        upper_bound = 60e3
        matplotlib.pyplot.xlabel("relation")
        matplotlib.pyplot.xticks(rotation=45, ha="right", rotation_mode="anchor")
        matplotlib.pyplot.ylabel("Count")
        matplotlib.pyplot.ylim(0, upper_bound)
        matplotlib.pyplot.title("relation distribution")

        # Add counts in bars
        for bar in range(len(counts)):
            matplotlib.pyplot.text(
                bar, upper_bound / 2, counts[bar], rotation=90, ha="center"
            )

        # Show
        matplotlib.pyplot.show()

        # Save
        if save:
            fig.savefig(
                os.path.join(os.getcwd(), save + ".pdf"),
                format="pdf",
                bbox_inches="tight",
            )

    def __getitem__(self):
        return 0

    def __len__(self):
        return 0
