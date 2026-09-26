"""Run exact controls without overwriting historical evidence (Python 3.10+)."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
TASKS={'e046': 'e046_replay.py', 'support': 'support_controls.py', 'heredity': 'heredity_controls.py', 'descent': 'descent_controls.py', 'coefficient-eight': 'coefficient_eight.py'}

def stable(value):
    if isinstance(value,dict):
        return {k:stable(v) for k,v in value.items() if k not in {'seconds','elapsed_seconds'}}
    if isinstance(value,list):return [stable(v) for v in value]
    return value

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only',choices=sorted(TASKS),help='Run one named check; default runs every check.')
    parser.add_argument('--output',type=Path,help='Write a new result file; refuses to overwrite an existing file.')
    parser.add_argument('--check-recorded',action='store_true',help='Compare deterministic results with the publication replay.')
    args=parser.parse_args()
    if args.output and args.output.exists():parser.error('output already exists')
    if not __debug__:parser.error('assertions must be enabled; do not run with -O')
    tasks={args.only:TASKS[args.only]} if args.only else TASKS
    results={}
    for name,script in tasks.items():
        start=time.monotonic()
        proc=subprocess.run([sys.executable,'-B',str(ROOT/'research/src'/script)],
                            text=True,capture_output=True,timeout=600,check=True)
        # Support controls emit per-rank JSON lines followed by a final summary.
        try:data=json.loads(proc.stdout)
        except json.JSONDecodeError:data={'records':[json.loads(line) for line in proc.stdout.splitlines() if line.strip()]}
        if args.check_recorded:
            old=json.loads((ROOT/'research/artifacts/publication_replay.json').read_text())['checks'][name]['result']
            assert stable(data)==stable(old),(name,'publication replay differs')
        results[name]={'result':data,'elapsed_seconds':round(time.monotonic()-start,3)}
        print(name+': PASS',file=sys.stderr)
    output=dict(status='PASS',checks=results,python=sys.version.split()[0],
                scope='Exact finite replays; universal proofs and external novelty require separate review.')
    rendered=json.dumps(output,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream:stream.write(rendered)
    else:print(rendered,end='')

if __name__=='__main__':main()
