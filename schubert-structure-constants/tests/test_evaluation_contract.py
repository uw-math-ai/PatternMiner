import pytest
from schubml.train.evaluation_contract import validate_contract


def test_product_checkpoint_cannot_claim_size_or_family_holdout(tmp_path):
    config = {"args": {"train_split": "product_disjoint:train", "data_dir": str(tmp_path)}}
    assert validate_contract(config, tmp_path, "product_disjoint:test")
    for tag in ("ambient_ood:test", "family_boolean:test", "coeff_mag:test"):
        with pytest.raises(ValueError, match="Train a separate model"):
            validate_contract(config, tmp_path, tag)
    with pytest.raises(ValueError, match="differs"):
        validate_contract(config, tmp_path / "replacement", "product_disjoint:test")


def test_separately_trained_size_checkpoint_is_accepted(tmp_path):
    config = {"args": {"train_split": "ambient_ood:train", "data_dir": str(tmp_path)}}
    assert validate_contract(config, tmp_path, "ambient_ood:test")
