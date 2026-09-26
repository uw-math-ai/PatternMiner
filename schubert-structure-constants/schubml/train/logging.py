"""Reproducibility + model-card helpers (PROJECT.md 18.3, 18.4)."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from typing import Dict, Optional


def repro_info(seed: Optional[int] = None) -> Dict[str, object]:
    info = {
        "datetime_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "seed": seed,
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["device"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def write_model_card(path: str, *, architecture: Dict, training: Dict,
                     metrics: Dict, shortcut_audit: Optional[Dict] = None,
                     interpretability: Optional[Dict] = None,
                     known_failures: Optional[str] = None) -> None:
    """Render a model card (PROJECT.md 18.3 template)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lines = ["# Model card", "", "## Architecture",
             "```json", json.dumps(architecture, indent=2, default=str), "```", "",
             "## Training data", "```json", json.dumps(training, indent=2, default=str), "```", "",
             "## Metrics", "Separated by split and negative tier.", "",
             "```json", json.dumps(metrics, indent=2, default=str), "```", ""]
    if shortcut_audit is not None:
        lines += ["## Shortcut audit",
                  "Length-only, descents-only, Rothe-summary comparisons.",
                  "```json", json.dumps(shortcut_audit, indent=2, default=str), "```", ""]
    if interpretability is not None:
        lines += ["## Interpretability status",
                  "```json", json.dumps(interpretability, indent=2, default=str), "```", ""]
    lines += ["## Known failures", known_failures or "TBD", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))
