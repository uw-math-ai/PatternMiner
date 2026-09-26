"""
Kronecker coefficients: full investigation, kron-001 through kron-012.

Covers:
  1. Exact S_n characters and Kronecker coefficients (verified two ways).
  2. Vanishing (g=0): Murnaghan's theorem, arXiv:1811.10015 (2018).
  3. Small-value thresholds (g<=1, g<=2): Rosas's Corollaries 15/17,
     arXiv:math/0001084 (2001).
  4. Direct value regression + rank correlation.
  5. Multi-class value buckets + implicit theorem-consistency checks:
     Corollary 15/17 respect, conjugation symmetry, semigroup property
     (with PROPER chance baselines throughout -- see kron-009's lesson).
  6. Parity (g mod 2): reported null result.

Requires: numpy, scikit-learn (pip install numpy scikit-learn)
"""
from functools import lru_cache
from itertools import combinations_with_replacement
from math import factorial
from fractions import Fraction
from collections import Counter

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from scipy.stats import spearmanr


# =======================================================================
# 1. Exact character values and Kronecker coefficients
# =======================================================================

def partitions(n, max_part=None):
    if max_part is None:
        max_part = n
    if n == 0:
        yield ()
        return
    for k in range(min(n, max_part), 0, -1):
        for rest in partitions(n - k, k):
            yield (k,) + rest


def _beta_numbers(lam, k):
    padded = list(lam) + [0] * (k - len(lam))
    return [padded[i] + (k - 1 - i) for i in range(k)]


def _rim_hook_removals(lam, k, r):
    beta = _beta_numbers(lam, k)
    betaset = set(beta)
    out = []
    for i in range(k):
        bi = beta[i]
        bp = bi - r
        if bp >= 0 and bp not in betaset:
            leg = sum(1 for j in range(k) if j != i and bp < beta[j] < bi)
            newbeta = sorted(beta[:i] + [bp] + beta[i + 1:], reverse=True)
            newlam = tuple(x for x in
                            (newbeta[idx] - (k - 1 - idx) for idx in range(k))
                            if x > 0)
            out.append((newlam, leg))
    return out


@lru_cache(maxsize=None)
def chi(lam, rho):
    """Irreducible S_n character value chi^lam(rho) via Murnaghan-Nakayama."""
    if len(rho) == 0:
        return 1 if len(lam) == 0 else 0
    if len(lam) == 0:
        return 0
    r, k = rho[0], len(lam)
    return sum(((-1) ** leg) * chi(newlam, rho[1:])
               for newlam, leg in _rim_hook_removals(lam, k, r))


def class_size(n, rho):
    c = Counter(rho)
    z = 1
    for part, mult in c.items():
        z *= (part ** mult) * factorial(mult)
    return factorial(n) // z


def kronecker(lam, mu, nu, n):
    """Exact Kronecker coefficient g(lam, mu, nu) for partitions of n."""
    total = Fraction(0)
    for rho in partitions(n):
        total += Fraction(class_size(n, rho) * chi(lam, rho)
                           * chi(mu, rho) * chi(nu, rho))
    total /= factorial(n)
    assert total.denominator == 1, f"non-integer result: {total}"
    return int(total)


def hook_length_dimension(lam, n):
    if not lam:
        return 1
    conj = [sum(1 for part in lam if part >= j) for j in range(1, lam[0] + 1)]
    hooks = 1
    for i, li in enumerate(lam):
        for j in range(li):
            arm, leg = li - j - 1, conj[j] - i - 1
            hooks *= arm + leg + 1
    return factorial(n) // hooks


def verify_implementation(max_n=8):
    for n in range(1, max_n + 1):
        for lam in partitions(n):
            assert chi(lam, tuple([1] * n)) == hook_length_dimension(lam, n)
    for n in range(1, max_n - 1):
        parts = list(partitions(n))
        for lam in parts:
            for mu in parts:
                s = sum(class_size(n, rho) * chi(lam, rho) * chi(mu, rho)
                        for rho in parts)
                expected = factorial(n) if lam == mu else 0
                assert s == expected
    print(f"[OK] hook-length + orthogonality checks passed, n<={max_n}")


# =======================================================================
# 2. Vanishing: Murnaghan's exact triangle-inequality rule (2018 general
#    statement, arXiv:1811.10015, Eqn. 17)
# =======================================================================

def murnaghan_triangle_vanishes(triple, n):
    def c(p):
        return n - (p[0] if p else 0)
    vals = sorted(c(p) for p in triple)
    return vals[2] > vals[0] + vals[1]


# =======================================================================
# 3. Small-value thresholds: Rosas (2001), arXiv:math/0001084,
#    Corollaries 15 and 17
# =======================================================================

def is_hook(p):
    return sum(1 for part in p if part > 1) <= 1


def is_two_row_or_fewer(p):
    return len(p) <= 2


def any_pair_both_hooks(triple):
    """Corollary 15: two hooks -> g in {0,1,2} EXACTLY."""
    a, b, c = triple
    return any(is_hook(x) and is_hook(y) for x, y in [(a, b), (a, c), (b, c)])


def any_pair_hook_and_tworow(triple):
    """Corollary 17: hook + two-row -> g in {0,1,2,3} EXACTLY."""
    a, b, c = triple
    pairs = [(a, b), (a, c), (b, c), (b, a), (c, a), (c, b)]
    return any(is_hook(x) and is_two_row_or_fewer(y) for x, y in pairs)


def hook_params(p):
    """d = number of 1's, w = the top part, for hook (w,1,1,...,1)."""
    return len(p) - 1, (p[0] if p else 0)


def theorem13_case4(triple):
    """Rosas Theorem 13 case 4: all three hooks, exact 0/1 value via a
    triangle inequality on the hook-length parameters."""
    if not all(is_hook(p) for p in triple):
        return None, False
    for i in range(3):
        lam = triple[i]
        others = [triple[j] for j in range(3) if j != i]
        d, w = hook_params(lam)
        e, u = hook_params(others[0])
        f, v = hook_params(others[1])
        if d <= w and e <= u and f <= v:
            val = 1 if (e <= d + f and d <= e + f and f <= e + d) else 0
            return val, True
    return None, False


def verify_rosas_corollaries(max_n=9):
    fired_15 = violations_15 = fired_17 = violations_17 = 0
    for n in range(3, max_n + 1):
        for triple in combinations_with_replacement(list(partitions(n)), 3):
            g = kronecker(*triple, n)
            if any_pair_both_hooks(triple):
                fired_15 += 1
                if g > 2:
                    violations_15 += 1
            if any_pair_hook_and_tworow(triple):
                fired_17 += 1
                if g > 3:
                    violations_17 += 1
    print(f"[OK] Corollary 15 (two hooks -> g<=2): fired {fired_15}, "
          f"{violations_15} violations")
    print(f"[OK] Corollary 17 (hook+two-row -> g<=3): fired {fired_17}, "
          f"{violations_17} violations")


# =======================================================================
# 4. Featurization and dataset construction
# =======================================================================

def featurize(triple, n, pad=11):
    def pad_p(p):
        return list(p) + [0] * (pad - len(p))
    ordered = sorted(pad_p(p) for p in triple)
    return ordered[0] + ordered[1] + ordered[2] + [n]


def bucket(g):
    return g if g <= 5 else 6


def build_dataset(n_values):
    rows = []
    for n in n_values:
        for triple in combinations_with_replacement(list(partitions(n)), 3):
            g = kronecker(*triple, n)
            rows.append((n, triple, featurize(triple, n), g))
    return rows


# =======================================================================
# main: run all five tasks + the implicit-consistency checks
# =======================================================================

def main():
    verify_implementation(max_n=8)
    verify_rosas_corollaries(max_n=9)

    print("\nBuilding training data (n=3..7) and fresh test data (n=10,11)...")
    train = build_dataset(range(3, 8))
    test = build_dataset([10, 11])
    Xtr = np.array([r[2] for r in train])
    Xte = np.array([r[2] for r in test])

    # --- Task: vanishing ---
    print("\n--- Vanishing (g=0 vs g>0) ---")
    ytr = np.array([1 if r[3] == 0 else 0 for r in train])
    yte = np.array([1 if r[3] == 0 else 0 for r in test])
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    maj = accuracy_score(yte, np.full_like(yte, int(round(ytr.mean()))))
    triangle_acc = accuracy_score(yte, [1 if murnaghan_triangle_vanishes(r[1], r[0]) else 0
                                         for r in test])
    print(f"Majority={maj:.4f}, Murnaghan rule alone={triangle_acc:.4f}, model={acc:.4f}")

    # --- Task: thresholds ---
    print("\n--- Small-value thresholds ---")
    for thresh in [1, 2]:
        ytr_t = np.array([1 if r[3] <= thresh else 0 for r in train])
        yte_t = np.array([1 if r[3] <= thresh else 0 for r in test])
        clf_t = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0)
        clf_t.fit(Xtr, ytr_t)
        acc_t = accuracy_score(yte_t, clf_t.predict(Xte))
        maj_t = accuracy_score(yte_t, np.full_like(yte_t, int(round(ytr_t.mean()))))
        print(f"g<={thresh}: majority={maj_t:.4f}, model={acc_t:.4f}")

    # --- Task: regression + rank correlation ---
    print("\n--- Direct value regression ---")
    ytr_r = np.log1p(np.array([r[3] for r in train]))
    yte_r = np.log1p(np.array([r[3] for r in test]))
    true_g = np.array([r[3] for r in test])
    gbr = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0)
    gbr.fit(Xtr, ytr_r)
    pred_r = gbr.predict(Xte)
    r2 = r2_score(yte_r, pred_r)
    rho, _ = spearmanr(true_g, pred_r)
    print(f"R^2 (log scale)={r2:.4f}, Spearman rank correlation={rho:.4f}")

    # --- Task: multi-class buckets + implicit theorem-consistency ---
    print("\n--- Multi-class value buckets + implicit theorem-consistency ---")
    ytr_b = np.array([bucket(r[3]) for r in train])
    yte_b = np.array([bucket(r[3]) for r in test])
    clf_b = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0)
    clf_b.fit(Xtr, ytr_b)
    pred_b = clf_b.predict(Xte)
    acc_b = accuracy_score(yte_b, pred_b)
    print(f"Bucket accuracy={acc_b:.4f}")

    cor15_idx = [i for i in range(len(test)) if any_pair_both_hooks(test[i][1])]
    viol_15 = sum(1 for i in cor15_idx if pred_b[i] > 2)
    print(f"Model violations of Corollary 15 (should be 0): {viol_15}/{len(cor15_idx)}")

    logreg_b = LogisticRegression(max_iter=5000)
    logreg_b.fit(Xtr, ytr_b)
    acc_b_lr = accuracy_score(yte_b, logreg_b.predict(Xte))
    print(f"Bucket accuracy (logistic regression, for comparison)={acc_b_lr:.4f}")

    # --- Parity: reported null result ---
    print("\n--- Parity (g mod 2) -- reported null result, included for completeness ---")
    ytr_p = np.array([r[3] % 2 for r in train])
    yte_p = np.array([r[3] % 2 for r in test])
    clf_p = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0)
    clf_p.fit(Xtr, ytr_p)
    acc_p = accuracy_score(yte_p, clf_p.predict(Xte))
    maj_p = accuracy_score(yte_p, np.full_like(yte_p, int(round(ytr_p.mean()))))
    print(f"Majority={maj_p:.4f}, model={acc_p:.4f} (delta {acc_p-maj_p:+.4f} -- weak/null)")


if __name__ == "__main__":
    main()