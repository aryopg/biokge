import torch


class Evaluator:
    def __init__(self, dataset_name):

        self.dataset_name = dataset_name

    def eval(self, y_pred_pos, y_pred_neg, K):
        """
        Calculate Hits@K
        This code is mostly adopted from the OGB Link Prediction code: https://github.com/snap-stanford/ogb/tree/master/ogb/linkproppred

        Args:
            y_pred_pos (numpy): predictions for positives
            y_pred_neg (numpy): predictions for negatives
            K (int): K to calculate hits for

        Returns:
            Hits@K (float)
        """
        if len(y_pred_neg) < K:
            return {"hits@{}".format(K): 1.0}

        kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]
        return float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(
            y_pred_pos
        )
