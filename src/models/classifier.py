import torch


class Classifier(torch.nn.Module):
    def __init__(self, entity_size: int, num_relations: int, rank: int):
        super().__init__()

        self.entity_size = entity_size
        self.num_relations = num_relations
        self.rank = rank

        print(f"entity_size: {entity_size}")
        self.entity_embeddings = torch.nn.Embedding(entity_size, rank)
        self.classifier = torch.nn.Linear(rank * 2, num_relations)

        # self.subject_linear = nn.Linear(rank, int(rank / 2))
        # self.object_linear = nn.Linear(rank, int(rank / 2))
        # self.classifier = nn.Linear(rank, num_relations)
        self.init_weight()

    def init_weight(self, init_range=1):
        self.entity_embeddings.weight.data.uniform_(-init_range, init_range)
        self.classifier.weight.data.uniform_(-init_range, init_range)
        self.entity_embeddings.weight.data *= 1e-3
        self.classifier.weight.data *= 1e-3

    def forward(self, input_pairs):
        subject = self.entity_embeddings(input_pairs[:, 0])
        object = self.entity_embeddings(input_pairs[:, 1])

        # subject = self.subject_linear(subject)
        # object = self.object_linear(object)

        # Do something with the subject and object
        representation = torch.cat((subject, object), dim=-1)
        preds = self.classifier(representation)

        return preds
