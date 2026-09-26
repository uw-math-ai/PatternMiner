"""Rank-defect primitive extracted unchanged."""

def rank_defect(x, w, i, j):
    return sum(a >= j for a in w[:i]) - sum(a >= j for a in x[:i])
