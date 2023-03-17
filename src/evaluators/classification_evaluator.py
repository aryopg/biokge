import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)


def calculate_metrics(y_true, y_score, n_classes, split):
    # For each class
    precision = {}
    recall = {}
    threshold = {}
    average_precision = {}
    auc_precision_recall = {}
    fpr = {}
    tpr = {}
    auc_roc = {}
    if n_classes > 1:
        for i in range(n_classes):
            precision[i], recall[i], threshold[i] = precision_recall_curve(
                y_true[:, i], y_score[:, i]
            )
            average_precision[i] = average_precision_score(y_true[:, i], y_score[:, i])
            auc_precision_recall[i] = auc(recall[i], precision[i])
            fpr[i], tpr[i], threshold[i] = roc_curve(y_true[:, i], y_score[:, i])
            auc_roc[i] = auc(fpr[i], tpr[i])
        class_accuracy = accuracy_score(
            np.argmax(y_true, axis=1), np.argmax(y_score, axis=1)
        )
        class_precision = precision_score(
            np.argmax(y_true, axis=1), np.argmax(y_score, axis=1), average=None
        )
        class_recall = recall_score(
            np.argmax(y_true, axis=1), np.argmax(y_score, axis=1), average=None
        )
        class_f1 = f1_score(
            np.argmax(y_true, axis=1), np.argmax(y_score, axis=1), average=None
        )
        kappa = cohen_kappa_score(np.argmax(y_true, axis=1), np.argmax(y_score, axis=1))
    else:
        precision[0], recall[0], threshold[0] = precision_recall_curve(y_true, y_score)
        average_precision[0] = average_precision_score(y_true, y_score)
        auc_precision_recall[0] = auc(recall[0], precision[0])
        fpr[0], tpr[0], threshold[0] = roc_curve(y_true, y_score)
        auc_roc[0] = auc(fpr[0], tpr[0])

        class_accuracy = accuracy_score(y_true, (y_score >= 0.5))
        class_precision = precision_score(y_true, (y_score >= 0.5), average=None)
        class_recall = recall_score(y_true, (y_score >= 0.5), average=None)
        class_f1 = f1_score(y_true, (y_score >= 0.5), average=None)
        kappa = cohen_kappa_score(y_true, (y_score >= 0.5))

    mean_average_precision = sum(list(average_precision.values())) / len(
        average_precision
    )
    averaged_auc_precision_recall = sum(list(auc_precision_recall.values())) / len(
        auc_precision_recall
    )
    averaged_auc_roc = sum(list(auc_roc.values())) / len(auc_roc)
    return {
        f"{split}_precision": precision,
        f"{split}_recall": recall,
        f"{split}_threshold": threshold,
        f"{split}_average_precision": average_precision,
        f"{split}_mean_average_precision": mean_average_precision,
        f"{split}_auc_precision_recall": auc_precision_recall,
        f"{split}_averaged_auc_precision_recall": averaged_auc_precision_recall,
        f"{split}_fpr": fpr,
        f"{split}_tpr": tpr,
        f"{split}_auc_roc": auc_roc,
        f"{split}_averaged_auc_roc": averaged_auc_roc,
        f"{split}_class_accuracy": class_accuracy,
        f"{split}_class_precision": class_precision,
        f"{split}_class_recall": class_recall,
        f"{split}_class_f1": class_f1,
        f"{split}_kappa": kappa,
    }
