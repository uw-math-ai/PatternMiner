"""Retain distinct scheduled development improvements for a bounded stability check."""
import json
from pathlib import Path
import shutil
import time
import torch
import session as s
from full_resources import Budget


def main():
    plan=s.HERE/'REFINEMENT_CHECKPOINT_PLAN.json';assert not plan.exists()
    current=json.loads((s.HERE/'neighbor_refinement.json').read_text())
    s.write(plan.name,dict(status='fixed_before_later_evaluations',current_best_step=current['best_step'],
        policy='Retain each subsequent distinct best checkpoint from the already scheduled evaluations, including the present one, without additional validation evaluations.',
        possible_followup='If the final selected model fails development qualification, test the three fixed convex combinations with weights .25, .5 and .75 of the final best and immediately preceding distinct retained best. These are development comparisons; assessment remains unopened until qualification.',
        source_sha256=s.sha(Path(__file__))))
    retained={}
    while True:
        report=json.loads((s.HERE/'neighbor_refinement.json').read_text());step=report['best_step']
        if step not in retained:
            budget=Budget(f'retain_refinement_{step:06d}',30);status='failed_or_partial'
            try:
                source=s.ART/'neighbor_refinement.pt';before=s.sha(source)
                saved=torch.load(source,map_location='cpu',weights_only=True)
                assert saved['step']==step and s.sha(source)==before
                target=s.ART/f'neighbor_refinement_step{step:06d}.pt';assert not target.exists();shutil.copyfile(source,target)
                assert s.sha(target)==before==s.sha(source)
                frozen=s.ART/f'neighbor_refinement_step{step:06d}_report.json';s.write(frozen.relative_to(s.HERE),report)
                retained[step]=dict(path=str(target.relative_to(s.HERE)),sha256=before,criterion=report['best_criterion'],report_sha256=s.sha(frozen))
                s.write('RETAINED_REFINEMENT_CHECKPOINTS.json',dict(status='running',checkpoints=retained));status='completed'
            finally:print(json.dumps(budget.finish(status)),flush=True)
        if report['status']!='running':break
        time.sleep(1)
    s.write('RETAINED_REFINEMENT_CHECKPOINTS.json',dict(status='completed',checkpoints=retained))


if __name__=='__main__':
    torch.set_num_threads(1);main()
