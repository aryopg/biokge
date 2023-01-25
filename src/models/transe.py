import numpy
import torch

from ..configs import ModelConfig


class TransE(torch.nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        config: ModelConfig,
    ):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_size = config.embedding_size
        self.entity_embeddings = torch.nn.Embedding(num_entities, config.embedding_size)
        self.relation_embeddings = torch.nn.Embedding(
            num_relations, config.embedding_size
        )
        self.init_weights()

    def init_weights(self) -> None:
        self.entity_embeddings.weight.data.uniform_(
            -6 / numpy.sqrt(self.embedding_size), 6 / numpy.sqrt(self.embedding_size)
        )
        self.relation_embeddings.weight.data.uniform_(
            -6 / numpy.sqrt(self.embedding_size), 6 / numpy.sqrt(self.embedding_size)
        )
        self.relation_embeddings.weight.data = torch.nn.functional.normalize(
            self.relation_embeddings.weight.data
        )  # Relation embeddings are normalised once at beginning

    def score(self, x: torch.Tensor):
        return self.forward(x)[0]

    def forward(self, x: torch.Tensor, **kwargs):

        # Entity embeddings are normalised every forward pass
        self.entity_embeddings.weight.data = torch.nn.functional.normalize(
            self.entity_embeddings.weight.data
        )

        # Get triple embeddings
        lhs = self.entity_embeddings(x[0])
        rel = self.relation_embeddings(x[1])
        rhs = self.entity_embeddings(x[2])

        # Return L2 norm
        return ((lhs + rel) - rhs).pow(2).sum(1).sqrt(), None
