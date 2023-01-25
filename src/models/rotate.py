import math
from typing import List

import torch
from torch import nn

from ..configs import ModelConfig


class RotatE(nn.Module):
    def __init__(
        self, entity_size: int, relation_size: int, config: ModelConfig, gamma: int = 0
    ):
        super(RotatE, self).__init__()
        self.entity_size = entity_size
        self.rank = config.embedding_size
        self.gamma = torch.tensor(gamma)

        self.entity_embeddings = nn.Embedding(entity_size, 2 * config.embedding_size)
        self.relation_embeddings = nn.Embedding(relation_size, config.embedding_size)

        self.init_weights(config.init_range, config.init_size)

    def init_weights(self, init_range=1, init_size=1e-3) -> None:
        self.entity_embeddings.weight.data.uniform_(-init_range, init_range)
        self.entity_embeddings.weight.data *= init_size
        self.relation_embeddings.weight.data.uniform_(-init_range, init_range)
        self.relation_embeddings.weight.data *= init_size

    @staticmethod
    def normalize_phases(rel_embedding):
        # normalize phases so that they lie in [-pi,pi]
        # first shift phases by pi
        out = rel_embedding + math.pi
        # compute the modulo (result then in [0,2*pi))
        out = out - out.div(2.0 * math.pi, rounding_mode="floor") * (2.0 * math.pi)
        # shift back
        out = out - math.pi
        return out

    def score(self, x: torch.Tensor):
        return self.forward(x)[0]

    def forward(self, x: torch.Tensor, **kwargs):
        lhs = self.entity_embeddings(x[0])
        rel = self.relation_embeddings(x[1])
        rhs = self.entity_embeddings(x[2])

        # Get real and imaginary embeddings
        lhs_real = lhs[:, : self.rank]
        lhs_img = lhs[:, self.rank :]
        rhs_real = rhs[:, : self.rank]
        rhs_img = rhs[:, self.rank :]

        phase_relation = self.normalize_phases(rel)

        rel_real = torch.cos(phase_relation)
        rel_img = torch.sin(phase_relation)

        score_real = lhs_real * rel_real - lhs_img * rel_img
        score_img = lhs_real * rel_img + lhs_img * rel_real
        score_real = score_real - rhs_real
        score_img = score_img - rhs_img

        score = torch.stack([score_real, score_img], dim=0)
        score = torch.norm(score, p=2, dim=0)

        score = self.gamma.item() - score.sum(dim=1)

        return score, None
