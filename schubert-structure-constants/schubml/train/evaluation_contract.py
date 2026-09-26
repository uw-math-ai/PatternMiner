"""Validate the training exclusions required by an evaluation split."""
from pathlib import Path


def validate_contract(config, data_dir, test_split):
    args = config.get("args", {})
    train_split = args.get("train_split")
    expected = test_split.rsplit(":", 1)[0] + ":train"
    if train_split != expected:
        raise ValueError(
            f"Cannot report {test_split}: checkpoint trained on {train_split!r}; "
            f"this evaluation requires {expected!r}. Train a separate model "
            "with the required exclusion. Removing overlapping test rows alone "
            "does not establish a family or size holdout."
        )
    recorded = args.get("data_dir")
    if recorded is None or Path(recorded).resolve() != Path(data_dir).resolve():
        raise ValueError("Evaluation data directory differs from the recorded training data. "
                         "An explicit dataset migration audit is required.")
    return {"training_split": train_split, "test_split": test_split,
            "data_dir": str(Path(data_dir).resolve()),
            "scope": "Configured training exclusions; not a content-hash audit."}
