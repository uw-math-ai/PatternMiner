"""Run the approved analyses and refinement sequentially on the local T4."""
import json
from pathlib import Path
import subprocess
import sys
import time
import session as s


def run(script,arguments=()):
    log=s.HERE/(script.removesuffix('.py')+'_full.log')
    with log.open('x') as stream:
        process=subprocess.run([sys.executable,str(s.HERE/script),*arguments],stdout=stream,stderr=subprocess.STDOUT)
    assert process.returncode==0,(script,process.returncode)


def main():
    path=s.HERE/'ERROR_SAMPLING_COMMON_WINDOW_COMPARISON.json'
    while not path.exists():time.sleep(2)
    comparison=json.loads(path.read_text());assert comparison['status']=='completed'
    winner=comparison['winner'];checkpoint=s.ART/(winner+'_common5000.pt')
    assert s.sha(checkpoint)==comparison['arms'][winner]['checkpoint_sha256']
    run('expanded_decomposition.py',['--checkpoint',str(checkpoint),'--name','EXPANDED_DECOMPOSITION'])
    run('support_controls.py',['--checkpoint',str(checkpoint),'--name','SUPPORT_CONTROLS'])
    while not (s.HERE/'NEIGHBOR_PREPARED.json').exists():time.sleep(2)
    assert json.loads((s.HERE/'NEIGHBOR_PREPARED.json').read_text())['status']=='completed'
    run('audit_neighbor_training.py')
    plan=dict(status='fixed_before_training',arm='neighbor_refinement',initial_checkpoint_sha256=s.sha(checkpoint),
        sampler_choice=winner,sampler_comparison_sha256=s.sha(path),steps=12000,evaluate_every=1500,
        error_refresh_every=3000,mining_samples=350000,wall_reservation=3600,
        learning_rates=dict(base=1e-5,relative_root_input=3e-5,root_action_attention=2e-4),
        ordinary_neighbors_share=.2,fresh_hard_pool_neighbors_share=.5,
        development='Nineteen criteria, with every positive supplied index; difficult pairs require both members correct',
        selection='Minimum development accuracy, then mean; raw and fixed exponential moving average at scheduled evaluations',
        assessment='Open once only if every development criterion is at least .99')
    assert not (s.HERE/'NEIGHBOR_REFINEMENT_PLAN.json').exists();s.write('NEIGHBOR_REFINEMENT_PLAN.json',plan)
    arguments=['--arm','neighbor_refinement','--initial',str(checkpoint),'--steps','12000','--evaluate-every','1500',
        '--wall-limit','3600','--base-lr','1e-5','--extra-lr','3e-5','--root-action-lr','2e-4',
        '--fresh','--adaptive','--screened','--exposed-assessment','--error-refresh-every','3000',
        '--mining-samples','350000','--paired-views','128','--hard-supplied','mixed']
    if winner=='pinned_error_control':arguments.append('--pinned-errors')
    run('train_neighbors.py',arguments)
    report=json.loads((s.HERE/'neighbor_refinement.json').read_text())
    if report['best_criterion'][0]>=.99:
        run('assess_new.py',['--training-report',str(s.HERE/'neighbor_refinement.json')])
    print(json.dumps(dict(status='completed',training_best=report['best_criterion'])),flush=True)


if __name__=='__main__':main()
