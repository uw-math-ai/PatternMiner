"""Direct degree-nine objects for the reviewed h3[s21] cancellation.

No imports, caches, coefficient table or work on import. Outputs are
partitions of nine. A..E labels store the one-based row of the sole 2
in a three-cell LR tableau of content (2,1); F/G/H are unique Pieri
origins. Negative tags retain the distinct signed origins. The fixed
49-pair table is an injection, not an output multiplicity lookup.
"""

MAX_DEGREE = 9
INNER_DEGREE = 6
LR21_INNERS = ((5, 1), (4, 2), (3, 2, 1), (3, 3))

# Each specification is (origin tag, inner partition, geometric type).
# A and B have the same shape but different Cauchy/P*e2[Q] origins.
POSITIVE_SPECS = (
    ('A', (5, 1), 'lr21'),
    ('B', (5, 1), 'lr21'),
    ('C', (4, 2), 'lr21'),
    ('D', (3, 2, 1), 'lr21'),
    ('E', (3, 3), 'lr21'),
    ('F', (4, 1, 1), 'vertical'),
    ('G', (3, 3), 'vertical'),
    ('H', (3, 3), 'horizontal'))

# Rp/Sp are the second, distinct 31*e2 origins of 42 and 321.
NEGATIVE_SPECS = (
    ('R', (4, 2), 'horizontal'),
    ('Rp', (4, 2), 'horizontal'),
    ('S', (3, 2, 1), 'horizontal'),
    ('Sp', (3, 2, 1), 'horizontal'),
    ('T', (4, 1, 1), 'horizontal'),
    ('U', (3, 1, 1, 1), 'horizontal'),
    ('J', (), 'fixed_leading_origin'))

# J carries the earlier e3[s3] cancellation's actual origin. For each
# (output, old tag, old inner shape), content (3,) and the all-one LR
# filling reconstruct its witness. Shape disambiguates the three V's.
J_ORIGINS = (
    ((7, 1, 1), 'U', (5, 1)),
    ((6, 3), 'V', (3, 3)),
    ((5, 3, 1), 'V', (3, 3)),
    ((3, 3, 3), 'V', (3, 3)))

# (output partition, negative tag, full positive label).
# An integer on A..E is a geometric row, never a copy index.
PAIR_ROWS = (
    ((7, 2), 'R', ('A', 2)),
    ((7, 2), 'Rp', ('B', 2)),
    ((7, 1, 1), 'T', ('A', 3)),
    ((7, 1, 1), 'J', ('B', 3)),
    ((6, 3), 'R', ('A', 2)),
    ((6, 3), 'Rp', ('B', 2)),
    ((6, 3), 'J', ('C', 2)),
    ((6, 2, 1), 'R', ('A', 2)),
    ((6, 2, 1), 'Rp', ('A', 3)),
    ((6, 2, 1), 'S', ('B', 2)),
    ((6, 2, 1), 'Sp', ('B', 3)),
    ((6, 2, 1), 'T', ('C', 3)),
    ((6, 1, 1, 1), 'T', ('A', 4)),
    ((6, 1, 1, 1), 'U', ('B', 4)),
    ((5, 4), 'R', ('C', 2)),
    ((5, 4), 'Rp', ('E', 2)),
    ((5, 3, 1), 'R', ('A', 3)),
    ((5, 3, 1), 'Rp', ('B', 3)),
    ((5, 3, 1), 'S', ('C', 2)),
    ((5, 3, 1), 'Sp', ('C', 3)),
    ((5, 3, 1), 'T', ('D', 2)),
    ((5, 3, 1), 'J', ('E', 3)),
    ((5, 2, 2), 'R', ('A', 3)),
    ((5, 2, 2), 'Rp', ('B', 3)),
    ((5, 2, 2), 'S', ('C', 3)),
    ((5, 2, 2), 'Sp', ('D', 3)),
    ((5, 2, 1, 1), 'S', ('A', 4)),
    ((5, 2, 1, 1), 'Sp', ('B', 4)),
    ((5, 2, 1, 1), 'T', ('C', 4)),
    ((5, 2, 1, 1), 'U', ('D', 4)),
    ((5, 1, 1, 1, 1), 'U', ('F',)),
    ((4, 4, 1), 'R', ('C', 3)),
    ((4, 4, 1), 'Rp', ('D', 2)),
    ((4, 4, 1), 'T', ('E', 2)),
    ((4, 3, 2), 'R', ('C', 3)),
    ((4, 3, 2), 'Rp', ('D', 2)),
    ((4, 3, 2), 'S', ('D', 3)),
    ((4, 3, 2), 'Sp', ('E', 3)),
    ((4, 3, 1, 1), 'S', ('C', 4)),
    ((4, 3, 1, 1), 'Sp', ('D', 2)),
    ((4, 3, 1, 1), 'T', ('D', 4)),
    ((4, 3, 1, 1), 'U', ('E', 4)),
    ((4, 2, 2, 1), 'S', ('C', 4)),
    ((4, 2, 2, 1), 'Sp', ('D', 3)),
    ((4, 2, 1, 1, 1), 'U', ('D', 5)),
    ((3, 3, 3), 'J', ('D', 3)),
    ((3, 3, 2, 1), 'S', ('D', 3)),
    ((3, 3, 2, 1), 'Sp', ('D', 4)),
    ((3, 3, 1, 1, 1), 'U', ('D', 5)))


def _partition_of(shape, total):
    """Bounded literal guard, before tuple comparisons or indexing."""
    if type(shape) is not tuple or not 1 <= len(shape) <= total:
        return False
    previous, size = total, 0
    for part in shape:
        if type(part) is not int or not 1 <= part <= previous:
            return False
        size += part
        if size > total:
            return False
        previous = part
    return size == total


def partition_ok(lam):
    """Recognize only exact built-in partitions of the output degree nine."""
    return _partition_of(lam, MAX_DEGREE)


def _contains(mu, lam):
    return len(mu) <= len(lam) and all(mu[i] <= lam[i] for i in range(len(mu)))


def _horizontal(mu, lam):
    """Interlacing on already guarded partitions; size is fixed by callers."""
    return _contains(mu, lam) and all(
        (mu[i] if i < len(mu) else 0) >= lam[i+1]
        for i in range(len(lam)-1))


def _vertical(mu, lam):
    """At most one new box in each row, on already guarded partitions."""
    return _contains(mu, lam) and all(
        lam[i]-(mu[i] if i < len(mu) else 0) <= 1
        for i in range(len(lam)))


def _lr21_cell(mu, lam, row):
    """Direct sole-2 cell criterion; arguments have literal bounded guards."""
    if not _contains(mu, lam) or not 1 <= row <= len(lam):
        return False
    index = row-1
    inner_width = mu[index] if index < len(mu) else 0
    below_width = lam[index+1] if index+1 < len(lam) else 0
    if lam[index] <= inner_width or lam[index] <= below_width:
        return False
    top = next(i for i in range(len(lam))
               if lam[i] > (mu[i] if i < len(mu) else 0))
    if index <= top:
        return False
    smaller = lam[:index]+(lam[index]-1,)+lam[index+1:]
    if smaller[-1] == 0:
        smaller = smaller[:-1]
    return _horizontal(mu, smaller)


def lr21_rows(mu, lam):
    """Ascending geometric row witnesses for the four specified inner shapes."""
    if not _partition_of(mu, INNER_DEGREE) or not partition_ok(lam):
        return ()
    if mu not in LR21_INNERS:
        return ()
    return tuple(row for row in range(1, len(lam)+1) if _lr21_cell(mu, lam, row))


def _positive_label_ok(obj):
    if type(obj) is not tuple or not 1 <= len(obj) <= 2:
        return False
    tag = obj[0]
    if type(tag) is not str or len(tag) != 1:
        return False
    if tag in ('A', 'B', 'C', 'D', 'E'):
        return (len(obj) == 2 and type(obj[1]) is int
                and 1 <= obj[1] <= MAX_DEGREE)
    return tag in ('F', 'G', 'H') and len(obj) == 1


def _negative_label_ok(obj):
    return (type(obj) is tuple and len(obj) == 1
            and type(obj[0]) is str and 1 <= len(obj[0]) <= 2
            and obj[0] in ('R', 'Rp', 'S', 'Sp', 'T', 'U', 'J'))


def is_positive(lam, obj):
    """Direct geometric/Pieri recognition, never generator membership."""
    if not partition_ok(lam) or not _positive_label_ok(obj):
        return False
    for tag, mu, kind in POSITIVE_SPECS:
        if tag == obj[0]:
            if kind == 'lr21':
                return _lr21_cell(mu, lam, obj[1])
            return _vertical(mu, lam) if kind == 'vertical' else _horizontal(mu, lam)
    return False


def is_negative(lam, obj):
    """Direct source-strip test or one of the four fixed named J origins."""
    if not partition_ok(lam) or not _negative_label_ok(obj):
        return False
    for tag, mu, kind in NEGATIVE_SPECS:
        if tag == obj[0]:
            if kind == 'fixed_leading_origin':
                return any(lam == shape for shape, _, _ in J_ORIGINS)
            return _horizontal(mu, lam)
    return False


def positive_objects(lam):
    """All positive labels, ordered by origin then geometric row."""
    if not partition_ok(lam):
        return ()
    objects = []
    for tag, mu, kind in POSITIVE_SPECS:
        if kind == 'lr21':
            objects.extend((tag, row) for row in lr21_rows(mu, lam))
        elif is_positive(lam, (tag,)):
            objects.append((tag,))
    return tuple(objects)


def negative_objects(lam):
    """All negative origins, using direct Pieri/fixed-origin tests."""
    if not partition_ok(lam):
        return ()
    return tuple((tag,) for tag, _, _ in NEGATIVE_SPECS if is_negative(lam, (tag,)))


def cancel(lam, neg):
    """Return the specified positive image, or None for an invalid source."""
    if not is_negative(lam, neg):
        return None
    for shape, negative_tag, positive in PAIR_ROWS:
        if lam == shape and neg[0] == negative_tag:
            return positive if is_positive(lam, positive) else None
    return None


def inverse(lam, pos):
    """Direct literal inverse; valid positive nonimages return None."""
    if not is_positive(lam, pos):
        return None
    for shape, negative_tag, positive in PAIR_ROWS:
        if lam == shape and pos == positive:
            negative = (negative_tag,)
            return negative if is_negative(lam, negative) else None
    return None


def is_survivor(lam, obj):
    """Recognize the positive complement without any generated-set query."""
    return is_positive(lam, obj) and inverse(lam, obj) is None
