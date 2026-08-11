import numpy as np


def ranking_metrics(rankings, positives, queries, recall_k=5, ndcg_k=5):
    """Compute the two ranking measures reported in Grand Table 2.

    ``positives`` contains every minimum-GED candidate, as specified for AIDS
    in GRAND. NDCG uses binary relevance. Recall is the usual fraction of the
    complete relevant set recovered within the first K results.
    """
    if recall_k <= 0 or ndcg_k <= 0:
        raise ValueError("recall_k and ndcg_k must be positive")
    if len(rankings) != len(queries):
        raise ValueError("rankings and queries must have the same length")

    recalls, ndcgs = [], []
    for query, ranking in zip(queries, rankings):
        relevant = set(positives.get(query, ()))
        if not relevant:
            continue

        retrieved = set(ranking[:recall_k])
        recalls.append(len(retrieved.intersection(relevant)) / float(len(relevant)))

        gains = [1.0 if candidate in relevant else 0.0 for candidate in ranking[:ndcg_k]]
        dcg = sum(gain / np.log2(index + 2) for index, gain in enumerate(gains))
        ideal = sum(1.0 / np.log2(index + 2) for index in range(min(len(relevant), ndcg_k)))
        ndcgs.append(dcg / ideal if ideal else 0.0)

    if not recalls:
        raise ValueError("No queries with ground-truth positives")
    # Keep the same presentation order as Table 2: NDCG first, then Recall.
    return {
        "NDCG@{}".format(ndcg_k): float(np.mean(ndcgs)),
        "Recall@{}".format(recall_k): float(np.mean(recalls)),
    }
