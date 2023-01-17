import os
import pandas
import numpy
import matplotlib.pyplot

class BioKGDataset():

    def __init__(self, config):
        '''
            Initialisation

            - config (obj): dataset config object
        '''
        
        self.config = config

        # Load data and store in attributes
        for filename in os.listdir(os.path.join(os.getcwd(), self.config.datasets_dir)):
            if "biokg" in filename:
                data_key = filename.replace("biokg.", "").replace(".tsv","").replace(".","_")
                setattr(self, data_key, pandas.read_csv(os.path.join(os.getcwd(), self.config.datasets_dir, filename), names=["subject", "relation", "object"], sep='\t'))

        # Establish relation vocabulary
        relations = self.links["relation"].unique()
        self.relation_voc = {relation: idx for relation, idx in zip(relations, range(len(relations)))}
        self.links["relation"] = self.links["relation"].apply(lambda relation: self.relation_voc[relation])
        self.num_relations = len(self.relation_voc)

        # Establish entity vocabulary
        entities = pandas.concat([self.links["subject"], self.links["object"]]).unique()
        self.entity_voc = {entity: idx for entity, idx in zip(entities, range(len(entities)))}
        self.links["subject"] = self.links["subject"].apply(lambda entity: self.entity_voc[entity])
        self.links["object"] = self.links["object"].apply(lambda entity: self.entity_voc[entity])
        self.num_entities = len(self.entity_voc)

    def plot_edge_distribution(self, save=False):
        '''
            Plot relation type distribution
            
            - save (string): path to figure if saving, starts in cwd
        '''
        counts = self.links["relation"].value_counts()
        
        # Plot
        fig, ax = matplotlib.pyplot.subplots()
        counts.plot(ax=ax, kind='bar')
        upper_bound = 60e3
        matplotlib.pyplot.xlabel("relation")
        matplotlib.pyplot.xticks(rotation=45, ha="right",rotation_mode='anchor')
        matplotlib.pyplot.ylabel("Count")
        matplotlib.pyplot.ylim(0,upper_bound)
        matplotlib.pyplot.title("relation distribution")

        # Add counts in bars
        for bar in range(len(counts)):
            matplotlib.pyplot.text(bar, upper_bound / 2, counts[bar], rotation=90, ha="center")

        # Show 
        matplotlib.pyplot.show()

        # Save
        if save:
            fig.savefig(os.path.join(os.getcwd(), save + ".pdf"), format='pdf', bbox_inches='tight')

    def get_edge_split(self, training_frac=0.8, valid_frac=0.1):
        '''
            Split into training|valid|test by taking exact fractions of each relation type
            
            - training_frac (float, 0-1): fraction of relation type to put in train data
            - valid_frac (float, 0-1): fraction of relation type to put in valid data

            1 - training_frac - valid_frac is fraction of relation type to put in test data
        '''
        train, valid, test = [], [], []

        for relation in self.links["relation"].unique():
            relation_entries = self.links.loc[self.links["relation"] == relation]
            
            # Split
            relation_train, relation_valid, relation_test = numpy.split(relation_entries, [int(training_frac*len(relation_entries)), int((training_frac+valid_frac)*len(relation_entries))])

            # Store
            train.append(relation_train)
            valid.append(relation_valid)
            test.append(relation_test)

        # Combine and shuffle
        train = pandas.concat(train).to_numpy()
        valid = pandas.concat(valid).to_numpy()
        test = pandas.concat(test).to_numpy()
        
        # Return whole
        return {"train": train, "valid": valid, "test": test}

    def __getitem__(self):
        return 0

    def __len__(self):
        return 0