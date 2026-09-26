"""Single-writer resource ledger for the Stanley-12 pilot.

The lock is deliberately held for the lifetime of a run.  This pilot values a
correct cumulative budget over parallel throughput; a later scheduler can add
explicit reservations if concurrent scientific runs are needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import time
from typing import Callable


EXACT_KEYS = (
    "sage_table_constructions",
    "gap_character_tables",
    "specht_trace_checks",
)


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def retained_bytes(path: Path) -> int:
    """Return retained regular-file bytes without following symlinks."""

    if not path.exists():
        return 0
    total = 0
    for candidate in path.rglob("*"):
        if candidate.is_file() and not candidate.is_symlink():
            total += candidate.stat().st_size
    return total


@dataclass
class RunBudget:
    project_dir: Path
    run_id: str
    uses_gpu: bool = False
    artifact_dir: Path | None = None
    clock: Callable[[], float] = time.monotonic
    agreement_path: Path = field(init=False)
    ledger_path: Path = field(init=False)
    lock_path: Path = field(init=False)
    agreement: dict = field(init=False)
    previous: list[dict] = field(init=False, default_factory=list)
    exact: dict[str, int] = field(init=False)
    started: float = field(init=False)
    started_utc: str = field(init=False)
    _lock_file: object = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.project_dir = Path(self.project_dir).resolve()
        self.agreement_path = self.project_dir / "resource_agreement.json"
        self.ledger_path = self.project_dir / "RESOURCE_USE.json"
        self.lock_path = self.project_dir / ".budget.lock"
        self.artifact_dir = (
            Path(self.artifact_dir).resolve()
            if self.artifact_dir is not None
            else self.project_dir / "artifacts"
        )
        self.exact = {key: 0 for key in EXACT_KEYS}

    @property
    def limits(self) -> dict:
        return self.agreement["limits"]

    def __enter__(self) -> "RunBudget":
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self._lock_file = self.lock_path.open("a+")
        fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX)
        try:
            self.agreement = _read_json(self.agreement_path, None)
            if not self.agreement or self.agreement.get("status") != "confirmed":
                raise RuntimeError("A confirmed resource agreement is required")
            limits = self.agreement.get("limits", {})
            if limits.get("external_model_calls") != 0:
                raise RuntimeError("This pilot must prohibit external model calls")
            if limits.get("external_api_calls") != 0:
                raise RuntimeError("This pilot must prohibit external API calls")
            if limits.get("paid_services") is not False:
                raise RuntimeError("This pilot must prohibit paid services")
            self.previous = _read_json(self.ledger_path, [])
            if any(record.get("run_id") == self.run_id for record in self.previous):
                raise RuntimeError(f"run_id already recorded: {self.run_id}")
            self._check_cumulative(0.0)
            self.started = self.clock()
            self.started_utc = datetime.now(timezone.utc).isoformat()
            return self
        except Exception:
            self._unlock()
            raise

    def _used(self, key: str) -> float:
        return sum(float(record.get(key, 0)) for record in self.previous)

    def _used_exact(self, key: str) -> int:
        return sum(int(record.get("exact", {}).get(key, 0)) for record in self.previous)

    def _check_cumulative(self, elapsed: float) -> None:
        limits = self.limits
        if self._used("wall_seconds") + elapsed > float(limits["wall_seconds"]):
            raise TimeoutError("cumulative wall-time allowance exceeded")
        gpu_elapsed = elapsed if self.uses_gpu else 0.0
        if self._used("gpu_seconds") + gpu_elapsed > float(limits["gpu_seconds"]):
            raise TimeoutError("cumulative GPU allowance exceeded")
        for key in EXACT_KEYS:
            if self._used_exact(key) + self.exact[key] > int(limits[key]):
                raise RuntimeError(f"cumulative exact-operation cap exceeded: {key}")
        size = retained_bytes(self.artifact_dir)
        if size > int(limits["storage_bytes"]):
            raise RuntimeError("retained artifact storage cap exceeded")

    def check(self) -> None:
        self._check_cumulative(self.clock() - self.started)

    def require_n(self, n: int) -> None:
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError("n must be an integer")
        if n < 1 or n > int(self.limits["max_n"]):
            raise RuntimeError(f"n={n} is outside the confirmed range")

    def charge(self, key: str, count: int = 1) -> None:
        if key not in EXACT_KEYS:
            raise KeyError(key)
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("count must be an integer")
        if count < 0:
            raise ValueError("count must be nonnegative")
        self.exact[key] += count
        try:
            # Charge immediately before the exact operation.  A denied charge
            # is rolled back because the operation must not be executed.
            self.check()
        except Exception:
            self.exact[key] -= count
            raise

    def _unlock(self) -> None:
        if self._lock_file is not None:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None

    def __exit__(self, exception_type, exception, traceback) -> bool:
        elapsed = self.clock() - self.started
        cap_error: Exception | None = None
        try:
            # A run can cross a time or storage boundary after its last
            # explicit check, so enforce every cumulative cap once more before
            # declaring success.
            self._check_cumulative(elapsed)
        except Exception as error:  # preserve and ledger all partial work
            cap_error = error

        status = (
            "passed"
            if exception is None and cap_error is None
            else "failed_or_partial"
        )
        errors = []
        if exception is not None:
            errors.append(repr(exception))
        if cap_error is not None:
            errors.append(f"final budget check: {cap_error!r}")
        record = {
            "run_id": self.run_id,
            "status": status,
            "started_utc": self.started_utc,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": elapsed,
            # Conservatively charge the complete run when GPU is requested.
            "gpu_seconds": elapsed if self.uses_gpu else 0.0,
            "gpu_charge_scope": "entire run wall time" if self.uses_gpu else "none",
            "exact": dict(self.exact),
            "retained_artifact_bytes": retained_bytes(self.artifact_dir),
            "error": "; ".join(errors) if errors else None,
        }
        try:
            # Record even a cap failure or an exception; failed work still counts.
            self.previous.append(record)
            _atomic_json(self.ledger_path, self.previous)
        finally:
            self._unlock()
        if exception is None and cap_error is not None:
            raise cap_error
        return False


def load_summary(project_dir: Path) -> dict:
    """Derive a summary from the canonical ledger rather than hand editing it."""

    project_dir = Path(project_dir)
    records = _read_json(project_dir / "RESOURCE_USE.json", [])
    return {
        "runs": len(records),
        "failed_or_partial_runs": sum(
            record.get("status") == "failed_or_partial" for record in records
        ),
        "wall_seconds": sum(float(record.get("wall_seconds", 0)) for record in records),
        "gpu_seconds": sum(float(record.get("gpu_seconds", 0)) for record in records),
        "exact": {
            key: sum(int(record.get("exact", {}).get(key, 0)) for record in records)
            for key in EXACT_KEYS
        },
        "retained_artifact_bytes": retained_bytes(project_dir / "artifacts"),
    }
