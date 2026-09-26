"""Finish exact data generation and preparation without GPU competition."""
import json
import subprocess
import sys
import time
import session as s


def run(script):
    with (s.HERE/(script.removesuffix('.py')+'.log')).open('x') as out:
        subprocess.run([sys.executable,str(s.HERE/script)],stdout=out,stderr=subprocess.STDOUT,check=True)


def main():
    while True:
        assert 'Traceback (most recent call last)' not in (s.HERE/'generate_populations.log').read_text()
        path=s.HERE/'DATA.json'
        if path.exists() and json.loads(path.read_text())['status']=='completed':break
        time.sleep(5)
    run('prepare_development.py')
    run('plan_failure_neighborhoods.py')
    run('generate_neighbors.py')
    run('prepare_neighbors.py')
    print(json.dumps(dict(status='completed')),flush=True)


if __name__=='__main__':main()
