"""Discard the supplied index while retaining all input permutations."""
import richardson_model as rm

SUPPLIED_COLUMNS = [1, 2, 9, 10, 11, 12, 13, 14, 17]
SUPPLIED_RELATIONS = list(range(6, 12))+list(range(42, 72))


class SuppliedIndependentSelector(rm.RichardsonSelector):
    def feature_map(self, x):
        features = super().feature_map(x).clone()
        features[:, :, SUPPLIED_COLUMNS] = 0
        return features

    @staticmethod
    def comparisons(x):
        relations = rm.AllPairOrderSelector.comparisons(x).clone()
        relations[:, :, :, SUPPLIED_RELATIONS] = 0
        return relations


def use(model):
    assert getattr(model, 'rank_only', False) or model.extra_kind == 'relative_roots'
    model.__class__ = SuppliedIndependentSelector
    model.supplied_index_independent = True
    return model
