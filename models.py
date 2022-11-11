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


# class KBCModel(nn.Module):
#     def get_candidates(self, chunk_begin, chunk_size, target='rhs', indices=None):
#         """
#         Get scoring candidates for (q, ?)
#         """
#         pass

#     def get_queries(self, queries, target='rhs'):
#         """
#         Get queries in a comfortable format for evaluation on GPU
#         """
#         pass

#     def score(self, x: torch.Tensor):
#         pass
    
#     def forward_bpr(self, pos, neg):
#         pos_scores = self.score(pos)
#         neg_scores = self.score(neg)
#         delta = pos_scores - neg_scores
#         fac = self.get_factor(torch.cat((pos, neg), dim=0))
#         return delta, fac
    
#     def forward_mr(self, pos, neg):
#         pass

#     def checkpoint(self, model_cache_path, epoch_id):
#         if model_cache_path is not None:
#             print('Save the model at epoch {}'.format(epoch_id))
#             torch.save(self.state_dict(), model_cache_path + '{}.model'.format(epoch_id))
        
#     def get_ranking(self, 
#                     queries: torch.Tensor,
#                     filters: Dict[Tuple[int, int], List[int]],
#                     batch_size: int = 1000, chunk_size: int = -1,
#                     candidates='rhs'): 
#         """
#         Returns filtered ranking for each queries.
#         :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
#         :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
#         :param batch_size: maximum number of queries processed at once
#         :param chunk_size: maximum number of answering candidates processed at once
#         :return:
#         """
#         query_type = candidates
#         if chunk_size < 0: # not chunking, score against all candidates at once
#             chunk_size = self.sizes[2] # entity ranking
#         ranks = torch.ones(len(queries))
#         predicted = torch.zeros(len(queries))
#         with torch.no_grad():
#             c_begin = 0
#             while c_begin < self.sizes[2]:
#                 b_begin = 0
#                 cands = self.get_candidates(c_begin, chunk_size, target=query_type)
#                 while b_begin < len(queries):
#                     these_queries = queries[b_begin:b_begin + batch_size]
#                     q = self.get_queries(these_queries, target=query_type)
#                     scores = q @ cands # torch.mv MIPS
#                     targets = self.score(these_queries)
#                     if filters is not None:
#                         scores = filtering(scores, these_queries, filters, 
#                                            n_rel=self.sizes[1], n_ent=self.sizes[2], 
#                                            c_begin=c_begin, chunk_size=chunk_size,
#                                            query_type=query_type)
#                     ranks[b_begin:b_begin + batch_size] += torch.sum(
#                         (scores >= targets).float(), dim=1
#                     ).cpu()
#                     predicted[b_begin:b_begin + batch_size] = torch.max(scores, dim=1)[1].cpu()
#                     b_begin += batch_size
#                 c_begin += chunk_size
#         return ranks, predicted

#     def get_metric_ogb(self, 
#                        queries: torch.Tensor,
#                        batch_size: int = 1000, 
#                        query_type='rhs',
#                        evaluator=None): 
#         """No need to filter since the provided negatives are ready filtered
#         :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
#         :param batch_size: maximum number of queries processed at once
#         :return:
#         """
#         test_logs = defaultdict(list)
#         with torch.no_grad():
#             b_begin = 0
#             while b_begin < len(queries):
#                 these_queries = queries[b_begin:b_begin + batch_size]
#                 ##### hard code neg_indice TODO
#                 if these_queries.shape[1] > 5: # more than h,r,t,h_type,t_type
#                     tot_neg = 1000 if evaluator.name in ['ogbl-biokg', 'ogbl-wikikg2'] else 0
#                     neg_indices = these_queries[:, 3:3+tot_neg]
#                     chunk_begin, chunk_size = None, None
#                 else:
#                     neg_indices = None
#                     chunk_begin, chunk_size = 0, self.sizes[2] # all the entities
#                 q = self.get_queries(these_queries, target=query_type)
#                 cands = self.get_candidates(chunk_begin, chunk_size,
#                                             target=query_type,
#                                             indices=neg_indices)
#                 if cands.dim() >= 3:# each example has a different negative candidate embedding matrix
#                     scores = torch.bmm(cands, q.unsqueeze(-1)).squeeze(-1)
#                 else:
#                     scores = q @ cands # torch.mv MIPS, pos + neg scores
#                 targets = self.score(these_queries) # positive scores
#                 batch_results = evaluator.eval({'y_pred_pos': targets.squeeze(-1), 
#                                                 'y_pred_neg': scores})
#                 del targets, scores, q, cands
#                 for metric in batch_results:
#                     test_logs[metric].append(batch_results[metric])
#                 b_begin += batch_size
#         metrics = {}
#         for metric in test_logs:
#             metrics[metric] = torch.cat(test_logs[metric]).mean().item()
#         return metrics


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