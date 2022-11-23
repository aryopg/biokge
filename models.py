from typing import Tuple, List, Dict
from collections import defaultdict
import torch
from torch import nn


def filtering(scores, these_queries, filters, n_rel, n_ent, 
              c_begin, chunk_size, query_type):
    # set filtered and true scores to -1e6 to be ignored
    # take care that scores are chunked
    for i, query in enumerate(these_queries):
        existing_s = (query[0].item(), query[1].item()) in filters # reciprocal training always has candidates = rhs
        existing_r = (query[2].item(), query[1].item() + n_rel) in filters # standard training separate rhs and lhs
        if query_type == 'rhs':
            if existing_s:
                filter_out = filters[(query[0].item(), query[1].item())]
                # filter_out += [queries[b_begin + i, 2].item()]
                filter_out += [query[2].item()]
        if query_type == 'lhs':
            if existing_r:
                filter_out = filters[(query[2].item(), query[1].item() + n_rel)]
                # filter_out += [queries[b_begin + i, 0].item()]    
                filter_out += [query[0].item()]                         
        if query_type == 'rel':
            pass
        if chunk_size < n_ent:
            filter_in_chunk = [
                    int(x - c_begin) for x in filter_out
                    if c_begin <= x < c_begin + chunk_size
            ]
            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
        else:
            scores[i, torch.LongTensor(filter_out)] = -1e6
    return scores


class ComplEx(nn.Module):
    def __init__(
            self, entity_size: int, rank: int,
            init_range: float = 0.1, init_size: float = 1e-3
    ):
        super(ComplEx, self).__init__()
        self.entity_size = entity_size
        self.rank = rank

        self.entity_embeddings = nn.Embedding(entity_size, 2 * rank)
        self.relation_embeddings = nn.Embedding(1, 2 * rank)  # Only one type of relation
        self.emb_check = self.entity_embeddings.weight
        
        self.init_weights(init_range, init_size)

    def init_weights(self, init_range=1, init_size=1e-3) -> None:
        self.entity_embeddings.weight.data.uniform_(-init_range, init_range)
        self.entity_embeddings.weight.data *= init_size
        self.relation_embeddings.weight.data.uniform_(-init_range, init_range)
        self.relation_embeddings.weight.data *= init_size

    def score(self, x):
        lhs = self.entity_embeddings(x[:, 0])
        rel = self.relation_embeddings(torch.IntTensor([0] * x.size(0)))
        rhs = self.entity_embeddings(x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x, score_rhs=True, score_rel=False, score_lhs=False):
        lhs = self.entity_embeddings(x[0])
        rel = self.relation_embeddings(x[1])
        rhs = self.entity_embeddings(x[2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        rhs_scores, rel_scores = None, None
        if score_rhs:
            to_score_entity = self.entity_embeddings.weight
            to_score_entity = to_score_entity[:, :self.rank], to_score_entity[:, self.rank:]
            rhs_scores = (
                (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score_entity[0].transpose(0, 1) +
                (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score_entity[1].transpose(0, 1)
            )
        if score_rel:
            to_score_rel = self.relation_embeddings.weight
            to_score_rel = to_score_rel[:, :self.rank], to_score_rel[:, self.rank:]
            rel_scores = (
                (lhs[0] * rhs[0] + lhs[1] * rhs[1]) @ to_score_rel[0].transpose(0, 1) +
                (lhs[0] * rhs[1] - lhs[1] * rhs[0]) @ to_score_rel[1].transpose(0, 1)
            )
        if score_lhs:
            to_score_lhs = self.entity_embeddings.weight
            to_score_lhs = to_score_lhs[:, :self.rank], to_score_lhs[:, self.rank:]
            lhs_scores = (
                (rel[0] * rhs[0] + rel[1] * rhs[1]) @ to_score_lhs[0].transpose(0, 1) + 
                (rel[0] * rhs[1] - rel[1] * rhs[0]) @ to_score_lhs[1].transpose(0, 1)
            )

        factors = self.get_factor(x)
        if score_rhs and score_rel and score_lhs:
            return (rhs_scores, rel_scores, lhs_scores), factors
        elif score_rhs and score_rel:
            return (rhs_scores, rel_scores), factors
        elif score_lhs and score_rel:
            pass
        elif score_rhs and score_lhs:
            pass
        elif score_rhs:
            return rhs_scores, factors
        elif score_rel:
            return rel_scores, factors
        elif score_lhs:
            return lhs_scores, factors
        else:
            return None

    def get_candidates(self, chunk_begin=None, chunk_size=None, target='rhs', indices=None):
        if target == 'rhs' or target == 'lhs': #TODO: extend to other models
            if indices == None:
                return self.entity_embeddings.weight.data[
                    chunk_begin:chunk_begin + chunk_size
                ].transpose(0, 1)
            else:
                bsz = indices.shape[0]
                num_cands = indices.shape[1]
                if target == 'rhs':
                    indices = indices[:, num_cands//2:]
                else:
                    indices = indices[:, 0:num_cands//2]
                return self.entity_embeddings.weight.data[indices.reshape(-1)].reshape(bsz, num_cands//2, -1)
        elif target == 'rel':
            return self.relation_embeddings.weight.data[
                chunk_begin:chunk_begin + chunk_size
            ].transpose(0, 1)
        
    def get_queries(self, queries, target='rhs'):
        lhs = self.entity_embeddings(queries[:, 0])
        rel = self.relation_embeddings(queries[:, 1])
        rhs = self.entity_embeddings(queries[:, 2])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        if target == 'rhs':
            return torch.cat([
                lhs[0] * rel[0] - lhs[1] * rel[1],
                lhs[0] * rel[1] + lhs[1] * rel[0]
            ], 1)
        elif target == 'lhs':
            return torch.cat([
                rhs[0] * rel[0] + rhs[1] * rel[1],
                rhs[1] * rel[0] - rhs[0] * rel[1]
            ], 1)
        elif target == 'rel':
            return torch.cat([
                lhs[0] * rhs[0] + lhs[1] * rhs[1],
                lhs[0] * rhs[1] - lhs[1] * rhs[0]
            ], 1)

    def get_factor(self, x):
        lhs = self.entity_embeddings(x[0])
        rel = self.relation_embeddings(x[1])
        rhs = self.entity_embeddings(x[2])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        return (torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2))