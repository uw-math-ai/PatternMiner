#!/usr/bin/env python3
"""Compact exact upper-mu dataset and shallow placement-rule diagnostic."""

from collections import Counter

from kl_type_a import inv, leq, rs
from rank_defect import rank_defect
from cover_features import descents, up_cover_data
from exact_kl import P, interval


TRAINING_CORES = {
    "C012": ((1, 0, 6, 5, 4, 3, 2, 9, 8, 7),
             (5, 6, 7, 8, 9, 0, 1, 2, 3, 4)),
    "C014-terminal": ((3, 2, 1, 0, 8, 7, 6, 5, 4, 9),
                      (7, 8, 2, 3, 9, 4, 5, 0, 1, 6)),
    "E041-a": ((0, 5, 4, 3, 2, 9, 8, 1, 7, 6),
               (3, 8, 9, 4, 5, 6, 7, 0, 1, 2)),
    "E041-b": ((3, 2, 8, 1, 0, 7, 6, 5, 4, 9),
               (7, 8, 9, 2, 3, 4, 5, 0, 1, 6)),
    "sparse": ((0, 3, 2, 7, 6, 1, 5, 4, 9, 8),
               (3, 6, 7, 8, 9, 0, 1, 2, 4, 5)),
}

C016_X = (4, 3, 2, 1, 0, 9, 8, 7, 6, 5)
C016_W = (8, 9, 3, 4, 5, 6, 7, 0, 1, 2)
C016_Z = (5, 3, 2, 1, 0, 9, 8, 7, 6, 4)


def upper_features(z, facet, chosen_descent, cover_data):
    elements = interval(z, facet)
    bottom = inv(z)
    ranks = tuple(sum(inv(y) == bottom + r for y in elements)
                  for r in range(10))
    defects = [rank_defect(z, facet, k, j)
               for k in range(1, 10) for j in range(1, 10)]
    k = chosen_descent + 1
    local = [rank_defect(z, facet, k, j) for j in range(1, 10)]
    a, b, p, q = cover_data
    return {
        "r1": ranks[1], "r2": ranks[2], "r3": ranks[3], "r4": ranks[4],
        "size": len(elements),
        "defect_volume": sum(defects),
        "defect_zeros": sum(value == 0 for value in defects),
        "local_volume": sum(local),
        "local_zeros": sum(value == 0 for value in local),
        "position_span": b - a,
        "value_span": q - p,
        "simple_sided": int(b == a + 1 or q == p + 1),
        "facet_descents": len(descents(facet)),
        "bottom_descents": len(descents(z)),
    }


def labelled_candidates(name, pair):
    x, w = pair
    rows = []
    for i in descents(w):
        facet = rs(w, i)
        required = set(descents(facet))
        for a, b, p, q, z in up_cover_data(x):
            if (z[i] <= z[i + 1] or not leq(z, facet)
                    or not required <= set(descents(z))):
                continue
            pol = P(z, facet)
            mu = pol[4] if len(pol) > 4 else 0
            rows.append({"source": name, "descent": i + 1,
                         "label": int(mu > 0),
                         **upper_features(z, facet, i, (a, b, p, q))})
    return rows


def c016_holdout():
    rows = []
    # The fixed cover is eligible and compatible at both descents.
    for i in (1, 6):
        facet = rs(C016_W, i)
        pol = P(C016_Z, facet)
        mu = pol[4] if len(pol) > 4 else 0
        rows.append({"source": "C016", "descent": i + 1,
                     "label": int(mu > 0),
                     **upper_features(C016_Z, facet, i, (0, 9, 4, 5))})
    return rows


def best_stumps(rows):
    features = [key for key in rows[0]
                if key not in {"source", "descent", "label"}]
    results = []
    for feature in features:
        values = sorted({row[feature] for row in rows})
        cuts = [values[0] - 1] + [(a + b) / 2 for a, b in zip(values, values[1:])]
        for cut in cuts:
            for direction in (1, -1):
                predictions = [int(direction * row[feature] > direction * cut)
                               for row in rows]
                errors = sum(pred != row["label"]
                             for pred, row in zip(predictions, rows))
                results.append((errors, feature, cut, direction))
    return sorted(results)


def main():
    training = []
    for name, pair in TRAINING_CORES.items():
        training.extend(labelled_candidates(name, pair))
    holdout = c016_holdout()
    best = best_stumps(training)
    best_error = best[0][0]
    best_rules = [row for row in best if row[0] == best_error]
    holdout_results = []
    for errors, feature, cut, direction in best_rules:
        preds = [int(direction * row[feature] > direction * cut)
                 for row in holdout]
        holdout_results.append((feature, cut, direction, tuple(preds)))
    print({
        "training_rows": len(training),
        "training_label_histogram": dict(Counter(row["label"] for row in training)),
        "training_sources": dict(Counter(row["source"] for row in training)),
        "best_stump_errors": best_error,
        "best_stumps": best_rules,
        "C016_true_labels": tuple(row["label"] for row in holdout),
        "C016_best_stump_predictions": holdout_results,
    })


if __name__ == "__main__":
    main()
