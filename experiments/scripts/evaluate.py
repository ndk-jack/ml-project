from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss


def compute_binary_metrics(y_true, y_score) -> dict:
    metrics = {}

    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    except Exception:
        metrics["pr_auc"] = None

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        metrics["roc_auc"] = None

    try:
        metrics["brier_score"] = float(brier_score_loss(y_true, y_score))
    except Exception:
        metrics["brier_score"] = None

    return metrics
