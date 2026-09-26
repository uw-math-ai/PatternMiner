"""Cumulative resource accounting for sequential scripts in this trial.

GPU runs conservatively charge their entire elapsed wall time to the GPU
allowance, including CPU preparation. Instrumented kernel time may be reported
separately. A new script, failed run, or replay never resets the allowance.
"""
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent


class Budget:
    def __init__(self, name, gpu=False):
        self.name, self.gpu = name, gpu
        self.agreement = json.loads((HERE/'resource_agreement.json').read_text())
        if self.agreement.get('status') != 'confirmed':
            raise RuntimeError('The resource agreement is not confirmed')
        self.path = HERE/'RESOURCE_USE.json'
        self.previous = json.loads(self.path.read_text()) if self.path.exists() else []
        if any(r['name'] == name for r in self.previous):
            raise RuntimeError('This run name has already consumed resources; choose a new name')
        self.wall_remaining = self.agreement['wall_seconds']-sum(r['wall_seconds'] for r in self.previous)
        self.gpu_remaining = self.agreement['gpu_seconds']-sum(r['gpu_seconds_charged'] for r in self.previous)
        self.exact_remaining = self.agreement['exact_queries']-sum(r['exact_queries'] for r in self.previous)
        self.exact_queries = 0
        self.start = time.monotonic()
        self.check()

    def check(self):
        elapsed = time.monotonic()-self.start
        if elapsed >= self.wall_remaining:
            raise TimeoutError('Cumulative wall-time allowance exhausted')
        if self.gpu and elapsed >= self.gpu_remaining:
            raise TimeoutError('Conservative cumulative GPU allowance exhausted')
        if self.exact_queries > self.exact_remaining:
            raise RuntimeError('Cumulative exact-query allowance exceeded')

    def exact_query(self):
        self.check()
        if self.exact_queries >= self.exact_remaining:
            raise RuntimeError('Cumulative exact-query allowance exhausted')
        self.exact_queries += 1

    def finish(self, status):
        elapsed = time.monotonic()-self.start
        record = dict(name=self.name, status=status, wall_seconds=elapsed,
                      gpu_seconds_charged=elapsed if self.gpu else 0,
                      gpu_charge_scope='entire process wall time' if self.gpu else 'CPU only',
                      exact_queries=self.exact_queries)
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.previous+[record], indent=2)+'\n')
        tmp.replace(self.path)
        return record
