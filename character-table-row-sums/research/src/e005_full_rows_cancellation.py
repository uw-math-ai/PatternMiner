"""Bounded local-move audit of Roichman signed full-row-sum objects."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import resource
import time
from collections import deque
from functools import lru_cache
from pathlib import Path

from .budget import RunBudget

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "artifacts/e001/discovery_coefficients.csv"
FROZEN_HASH = "427647e76ce5b99bbc653ac5fd92f02bd9a62529e35190e66b67526f643de032"
MODES = ("fixed_tableau", "separate_adjacent", "joint_adjacent", "joint_interval")


@lru_cache(None)
def partitions(n, largest=None):
    if n == 0:
        return ((),)
    bound = min(n, largest if largest is not None else n)
    return tuple((a,) + rest for a in range(bound, 0, -1)
                 for rest in partitions(n-a, a))


@lru_cache(None)
def tableaux(shape):
    """Encode a tableau by the row containing each successive label."""
    if not shape:
        return ((),)
    result = []
    for row, length in enumerate(shape):
        if row+1 < len(shape) and length == shape[row+1]:
            continue
        prev = list(shape)
        prev[row] -= 1
        if not prev[-1]:
            prev.pop()
        result.extend(t + (row,) for t in tableaux(tuple(prev)))
    return tuple(result)


def hook_count(shape):
    product = 1
    for row, length in enumerate(shape):
        for col in range(length):
            product *= length-col + sum(other > col for other in shape[row+1:])
    return math.factorial(sum(shape)) // product


def descents(t):
    return frozenset(i+1 for i in range(len(t)-1) if t[i+1] > t[i])


@lru_cache(None)
def sign(descent_set, mu):
    start = 0
    internal_count = 0
    for part in mu:
        seen_gap = False
        for i in range(start+1, start+part):
            if i in descent_set:
                if seen_gap:
                    return 0
                internal_count += 1
            else:
                seen_gap = True
        start += part
    return -1 if internal_count % 2 else 1


@lru_cache(None)
def partition_moves(mu):
    result = set()
    for i, a in enumerate(mu):
        for b in range(1, a//2+1):
            result.add(tuple(sorted(mu[:i] + mu[i+1:] + (b,a-b), reverse=True)))
        for j in range(i+1, len(mu)):
            result.add(tuple(sorted(tuple(x for k,x in enumerate(mu) if k not in (i,j))
                                    + (a+mu[j],), reverse=True)))
    result.discard(mu)
    return result


def tableau_moves(t, valid, interval=False):
    result = set()
    for a in range(len(t)-1):
        for b in range(a+1, len(t) if interval else a+2):
            block = t[a:b+1]
            for rotated in (block[1:]+block[:1], block[-1:]+block[:-1]):
                candidate = t[:a]+rotated+t[b+1:]
                if candidate != t and candidate in valid:
                    result.add(candidate)
    return result


def maximum_matching(adj, right_count):
    """Hopcroft--Karp plus an alternating-reachability Hall certificate."""
    left_match = [-1] * len(adj)
    right_match = [-1] * right_count
    while True:
        distance = [0 if x == -1 else -1 for x in left_match]
        queue = deque(i for i,x in enumerate(left_match) if x == -1)
        found = False
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                nxt = right_match[v]
                if nxt == -1:
                    found = True
                elif distance[nxt] == -1:
                    distance[nxt] = distance[u]+1
                    queue.append(nxt)
        if not found:
            break

        def augment(u):
            for v in adj[u]:
                nxt = right_match[v]
                if nxt == -1 or (distance[nxt] == distance[u]+1 and augment(nxt)):
                    left_match[u] = v
                    right_match[v] = u
                    return True
            distance[u] = -1
            return False

        for u,x in enumerate(left_match):
            if x == -1:
                augment(u)

    assert len({v for v in left_match if v >= 0}) == sum(v >= 0 for v in left_match)
    assert all(v < 0 or v in adj[u] for u,v in enumerate(left_match))
    left_seen = {u for u,v in enumerate(left_match) if v < 0}
    queue = deque(left_seen)
    right_seen = set()
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v in right_seen:
                continue
            right_seen.add(v)
            nxt = right_match[v]
            assert nxt >= 0, "augmenting path left after claimed maximum"
            if nxt not in left_seen:
                left_seen.add(nxt)
                queue.append(nxt)
    if left_seen:
        assert {v for u in left_seen for v in adj[u]} == right_seen
        assert len(left_seen) > len(right_seen)
    return left_match, sorted(left_seen), sorted(right_seen)


def encode_object(obj):
    t, mu = obj
    rows = [[] for _ in range(max(t)+1)]
    for label, row in enumerate(t, 1):
        rows[row].append(label)
    return {"tableau_rows": rows, "cycle_partition": mu,
            "descents": sorted(descents(t)), "sign": sign(descents(t),mu)}


def freeze(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True)+"\n").encode()
    with path.open("xb") as handle:
        handle.write(raw)
    digest = hashlib.sha256(raw).hexdigest()
    with path.with_suffix(".sha256").open("x") as handle:
        handle.write(f"{digest}  {path.name}\n")
    return digest


def main():
    resource.setrlimit(resource.RLIMIT_AS, (512*1024**2, 512*1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (45, 45))
    started = time.monotonic()
    assert hashlib.sha256(FROZEN.read_bytes()).hexdigest() == FROZEN_HASH
    with FROZEN.open() as handle:
        frozen = {(int(r["n"]), tuple(json.loads(r["lambda"]))): int(r["coefficient"])
                  for r in csv.DictReader(handle) if r["family_id"] == "full"}
    outdir = ROOT / "artifacts/e005"
    outdir.mkdir(exist_ok=True)
    result = {"experiment": "E005", "max_n": 8, "frozen_control_sha256": FROZEN_HASH,
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "protocol_sha256": hashlib.sha256((ROOT / "experiments/E005-full-row-cancellation.md").read_bytes()).hexdigest(),
              "shapes": [], "first_failure": {mode: None for mode in MODES}}
    with RunBudget(ROOT, "e005-full-row-cancellation-v1") as budget:
        for n in range(1,9):
            for shape in partitions(n):
                budget.check()
                if time.monotonic()-started > 43:
                    raise TimeoutError("bounded E005 wall-time cap")
                tabs = tableaux(shape)
                assert len(tabs) == hook_count(shape)
                valid = set(tabs)
                positives, negatives = [], []
                for t in tabs:
                    for mu in partitions(n):
                        value = sign(descents(t),mu)
                        if value:
                            (positives if value == 1 else negatives).append((t,mu))
                assert len(positives)-len(negatives) == frozen[n,shape]
                positive_index = {obj:i for i,obj in enumerate(positives)}
                swaps = {t: tableau_moves(t, valid) for t in tabs}
                rotations = {t: tableau_moves(t, valid, True) for t in tabs}
                stats = {"n": n, "shape": shape, "tableaux": len(tabs),
                         "positive": len(positives), "negative": len(negatives),
                         "signed_sum": len(positives)-len(negatives), "modes": {}}
                for mode in MODES:
                    adj = []
                    for t,mu in negatives:
                        moves = partition_moves(mu)
                        candidates = {(t,nu) for nu in moves}
                        if mode == "separate_adjacent":
                            candidates.update((u,mu) for u in swaps[t])
                        elif mode in ("joint_adjacent", "joint_interval"):
                            ts = swaps[t] if mode == "joint_adjacent" else rotations[t]
                            candidates.update((u,nu) for u in ts for nu in moves | {mu})
                        adj.append(sorted(positive_index[obj] for obj in candidates if obj in positive_index))
                    match, left, right = maximum_matching(adj,len(positives))
                    matched = sum(v >= 0 for v in match)
                    stats["modes"][mode] = {"edges":sum(map(len,adj)),"matched":matched,
                         "deficiency":len(negatives)-matched,"isolated_negatives":sum(not x for x in adj),
                         "matching_sha256": hashlib.sha256(json.dumps(match).encode()).hexdigest()}
                    if left and result["first_failure"][mode] is None:
                        result["first_failure"][mode] = {"n":n,"shape":shape,
                            "hall_negative_count":len(left),"hall_positive_neighbor_count":len(right),
                            "negative_objects":[encode_object(negatives[i]) for i in left],
                            "positive_neighbors":[encode_object(positives[i]) for i in right]}
                result["shapes"].append(stats)
            print(json.dumps({"completed_n":n,"first_failure_n":{
                mode: w["n"] if w else None for mode,w in result["first_failure"].items()}}), flush=True)
        assert result["first_failure"]["fixed_tableau"]["n"] == 3
        assert result["first_failure"]["separate_adjacent"]["n"] == 3
        assert result["first_failure"]["joint_adjacent"]["n"] <= 5
        assert hashlib.sha256(FROZEN.read_bytes()).hexdigest() == FROZEN_HASH
        result["elapsed_seconds"] = time.monotonic()-started
        result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        result["control_shape_count"] = len(result["shapes"])
        result["status"] = "passed; finite construction diagnostics only"
        digest = freeze(outdir/"result.json", result)
        print(json.dumps({"result_sha256":digest,"seconds":result["elapsed_seconds"],
                          "shapes":result["control_shape_count"]}), flush=True)


if __name__ == "__main__":
    main()
