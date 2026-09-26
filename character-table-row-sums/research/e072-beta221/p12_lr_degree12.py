"""Explicit LR tableaux through degree twelve: a separate E044 dependency.

Adapted from the frozen p12_lr_degree9.py without modifying that file.
The cap is twelve; literal shape/witness lengths and positive coordinates
are checked before caching. Importing computes no tableaux. A witness is
(reading_cells, entries), in one-based coordinates, with rows read from
top to bottom and each row right to left. This extension requires its own
validation; earlier degree-nine tests do not certify it.
"""
from functools import lru_cache


MAX_DEGREE = 12


def partition_ok(shape):
    return (type(shape) is tuple and len(shape) <= MAX_DEGREE
            and all(type(x) is int and 0 < x <= MAX_DEGREE for x in shape)
            and sum(shape) <= MAX_DEGREE
            and all(shape[i] >= shape[i+1] for i in range(len(shape)-1)))


def parameters_ok(alpha, beta, nu):
    return (partition_ok(alpha) and partition_ok(beta) and partition_ok(nu)
            and sum(nu) == sum(alpha)+sum(beta) <= MAX_DEGREE
            and len(alpha) <= len(nu)
            and all(alpha[i] <= nu[i] for i in range(len(alpha))))


@lru_cache(None)
def reading_cells(alpha, nu):
    return tuple((row+1, col) for row, width in enumerate(nu)
                 for col in range(width, (alpha[row] if row < len(alpha) else 0), -1))


@lru_cache(None)
def _generate(alpha, beta, nu):
    cells = reading_cells(alpha, nu)
    counts, filling, word, result = [0]*len(beta), {}, [], []

    def visit(index):
        if index == len(cells):
            if tuple(counts) == beta:
                result.append((cells, tuple(word)))
            return
        row, col = cells[index]
        above, right = filling.get((row-1, col)), filling.get((row, col+1))
        for letter in range(1, len(beta)+1):
            if counts[letter-1] == beta[letter-1]:
                continue
            if above is not None and letter <= above:
                continue
            if right is not None and letter > right:
                continue
            counts[letter-1] += 1
            if all(counts[i] >= counts[i+1] for i in range(len(beta)-1)):
                filling[row, col] = letter
                word.append(letter)
                visit(index+1)
                word.pop()
                del filling[row, col]
            counts[letter-1] -= 1

    visit(0)
    return tuple(result)


def lr_tableaux(alpha, beta, nu):
    if not parameters_ok(alpha, beta, nu):
        return ()
    return _generate(alpha, beta, nu)


@lru_cache(None)
def _recognize(alpha, beta, nu, witness):
    cells, entries = witness
    if cells != reading_cells(alpha, nu) or len(entries) != sum(beta):
        return False
    if any(type(x) is not int or x < 1 or x > len(beta) for x in entries):
        return False
    filling = dict(zip(cells, entries))
    content = [0]*len(beta)
    for cell, letter in zip(cells, entries):
        row, col = cell
        right, below = filling.get((row, col+1)), filling.get((row+1, col))
        if right is not None and letter > right:
            return False
        if below is not None and letter >= below:
            return False
        content[letter-1] += 1
        if any(content[i] < content[i+1] for i in range(len(beta)-1)):
            return False
    return tuple(content) == beta


def is_lr_tableau(alpha, beta, nu, witness):
    if (not parameters_ok(alpha, beta, nu) or type(witness) is not tuple
            or len(witness) != 2 or type(witness[0]) is not tuple
            or type(witness[1]) is not tuple):
        return False
    cells, entries = witness
    if len(cells) > MAX_DEGREE or len(entries) > MAX_DEGREE:
        return False
    if any(type(cell) is not tuple or len(cell) != 2
           or any(type(x) is not int or not 1 <= x <= MAX_DEGREE for x in cell)
           for cell in cells):
        return False
    if any(type(x) is not int for x in entries):
        return False
    return _recognize(alpha, beta, nu, witness)


def cache_report():
    return {'generator': _generate.cache_info()._asdict(),
            'recognizer': _recognize.cache_info()._asdict(),
            'reading_cells': reading_cells.cache_info()._asdict()}


def clear_caches():
    """Release only this new module's caches between bounded stages."""
    _generate.cache_clear()
    _recognize.cache_clear()
    reading_cells.cache_clear()
