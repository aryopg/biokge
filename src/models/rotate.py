import math
from typing import List

import torch
from torch import nn


class RotatE(nn.Module):
    def __init__(
        self,
        entity_size: int,
        relation_size: int,
        rank: int,
        gamma: int = 0,
        init_range: float = 0.1,
        init_size: float = 1e-3,
    ):
        super(RotatE, self).__init__()
        self.entity_size = entity_size
        self.rank = rank
        self.gamma = torch.tensor(gamma)

        self.entity_embeddings = nn.Embedding(entity_size, 2 * rank)
        self.relation_embeddings = nn.Embedding(relation_size, rank)

        self.init_weights(init_range, init_size)

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
        return score

    def forward(
        self,
        x: torch.Tensor,
        score_rhs: bool = True,
        score_lhs: bool = False,
        score_rel: bool = False,  # for uniformity, but not implemented on RotatE
    ):
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

        if score_rhs:
            entity_embedding = self.entity_embeddings.weight
            entity_embedding_real = entity_embedding[:, : self.rank]
            entity_embedding_img = entity_embedding[:, self.rank :]

            rhs_score_real = lhs_real * rel_real - lhs_img * rel_img
            rhs_score_img = lhs_real * rel_img + lhs_img * rel_real

            rhs_score_real = rhs_score_real.unsqueeze(1) - entity_embedding_real
            rhs_score_img = rhs_score_img.unsqueeze(1) - entity_embedding_img

            rhs_scores = torch.stack([rhs_score_real, rhs_score_img], dim=0)
            rhs_scores = torch.norm(rhs_scores, p=2, dim=0)

            rhs_scores = self.gamma.item() - rhs_scores.sum(dim=2)
        if score_lhs:
            entity_embedding = self.entity_embeddings.weight
            entity_embedding_real = entity_embedding[:, : self.rank]
            entity_embedding_img = entity_embedding[:, self.rank :]

            rel_img = -rel_img
            lhs_score_real = rel_real * rhs_real - rel_img * rhs_img
            lhs_score_img = rel_real * rhs_img + rel_img * rhs_real

            lhs_score_real = lhs_score_real.unsqueeze(1) - entity_embedding_real
            lhs_score_img = lhs_score_img.unsqueeze(1) - entity_embedding_img

            lhs_scores = torch.stack([lhs_score_real, lhs_score_img], dim=0)
            lhs_scores = torch.norm(lhs_scores, p=2, dim=0)

            lhs_scores = self.gamma.item() - lhs_scores.sum(dim=2)

        # Retrieve the model factors wrt the input
        factors = (
            torch.sqrt(lhs_real**2 + lhs_img**2),  # left hand side factors
            torch.sqrt(rel_real**2 + rel_img**2),  # relation factors
            torch.sqrt(rhs_real**2 + rhs_img**2),  # right hand side factors
        )

        if score_rhs and score_lhs:
            return (rhs_scores, lhs_scores), factors
        elif score_rhs:
            return rhs_scores, factors
        elif score_lhs:
            return lhs_scores, factors
