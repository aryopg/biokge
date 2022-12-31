from typing import List

import torch
from torch import nn


class ComplEx(nn.Module):
    def __init__(
        self,
        entity_size: int,
        relation_size: int,
        rank: int,
        init_range: float = 0.1,
        init_size: float = 1e-3,
    ):
        super(ComplEx, self).__init__()
        self.entity_size = entity_size
        self.rank = rank

        self.entity_embeddings = nn.Embedding(entity_size, 2 * rank)
        self.relation_embeddings = nn.Embedding(relation_size, 2 * rank)

        self.init_weights(init_range, init_size)

    def init_weights(self, init_range=1, init_size=1e-3) -> None:
        self.entity_embeddings.weight.data.uniform_(-init_range, init_range)
        self.entity_embeddings.weight.data *= init_size
        self.relation_embeddings.weight.data.uniform_(-init_range, init_range)
        self.relation_embeddings.weight.data *= init_size

    def score(self, x: torch.Tensor):
        lhs = self.entity_embeddings(x[0])
        rel = self.relation_embeddings(x[1])
        rhs = self.entity_embeddings(x[2])

        lhs = lhs[:, : self.rank], lhs[:, self.rank :]
        rel = rel[:, : self.rank], rel[:, self.rank :]
        rhs = rhs[:, : self.rank], rhs[:, self.rank :]

        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0]
            + (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1,
            keepdim=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        score_rhs: bool = True,
        score_rel: bool = False,
        score_lhs: bool = False,
    ):
        lhs = self.entity_embeddings(x[0])
        rel = self.relation_embeddings(x[1])
        rhs = self.entity_embeddings(x[2])

        # Get real and imaginary embeddings
        lhs = lhs[:, : self.rank], lhs[:, self.rank :]
        rel = rel[:, : self.rank], rel[:, self.rank :]
        rhs = rhs[:, : self.rank], rhs[:, self.rank :]

        rhs_scores, rel_scores = None, None
        if score_rhs:
            score_entity = self.entity_embeddings.weight
            score_entity_real = score_entity[:, : self.rank]
            score_entity_img = score_entity[:, self.rank :]

            rhs_score_real = (
                lhs[0] * rel[0] - lhs[1] * rel[1]
            ) @ score_entity_real.transpose(0, 1)
            rhs_score_img = (
                lhs[0] * rel[1] + lhs[1] * rel[0]
            ) @ score_entity_img.transpose(0, 1)

            rhs_scores = rhs_score_real + rhs_score_img
        if score_rel:
            score_relation = self.relation_embeddings.weight
            score_relation_real = score_relation[:, : self.rank]
            score_relation_img = score_relation[:, self.rank :]

            rel_score_real = (
                lhs[0] * rhs[0] + lhs[1] * rhs[1]
            ) @ score_relation_real.transpose(0, 1)
            rel_score_img = (
                lhs[0] * rhs[1] - lhs[1] * rhs[0]
            ) @ score_relation_img.transpose(0, 1)

            rel_scores = rel_score_real + rel_score_img
        if score_lhs:
            score_entity = self.entity_embeddings.weight
            score_entity_real = score_entity[:, : self.rank]
            score_entity_img = score_entity[:, self.rank :]

            lhs_score_real = (
                rel[0] * rhs[0] + rel[1] * rhs[1]
            ) @ score_entity_real.transpose(0, 1)
            lhs_score_img = (
                rel[0] * rhs[1] - rel[1] * rhs[0]
            ) @ score_entity_img.transpose(0, 1)

            lhs_scores = lhs_score_real + lhs_score_img

        # Retrieve the model factors wrt the input
        factors = (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),  # left hand side factors
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),  # relation factors
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2),  # right hand side factors
        )

        if score_rhs and score_rel and score_lhs:
            return (rhs_scores, rel_scores, lhs_scores), factors
        elif score_rhs and score_rel:
            return (rhs_scores, rel_scores), factors
        elif score_lhs and score_rel:
            pass
        elif score_rhs and score_lhs:
            return (rhs_scores, lhs_scores), factors
        elif score_rhs:
            return rhs_scores, factors
        elif score_rel:
            return rel_scores, factors
        elif score_lhs:
            return lhs_scores, factors
        else:
            return None
