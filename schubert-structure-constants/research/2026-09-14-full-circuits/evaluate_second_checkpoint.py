"""Natively evaluate a frozen checkpoint on all twenty-seven development criteria."""
import argparse
import json
from pathlib import Path
import torch
import session as s
import richardson_model as rm
import second_training_data as td
from exposed_assessment import append_validation
from train_neighbors import evaluate
from full_resources import Budget


def main(args,budget):
    path=Path(args.checkpoint).resolve();assert path.parent==s.ART.resolve()
    before=s.sha(path);model=rm.load(path).eval();data=append_validation(td.validation(True));assert len(data)==27
    metrics,criterion=evaluate(model,data);assert s.sha(path)==before
    if args.reference_report:
        source=json.loads(Path(args.reference_report).read_text())
        expected=source['history'][-1][args.reference_variant+'_validation']
        assert {k:v['selection_accuracy'] for k,v in metrics.items()}=={k:v['selection_accuracy'] for k,v in expected.items()}
    report=dict(status='completed',configuration=dict(arm=path.stem),checkpoint_sha256=before,
        initial_validation=metrics,best_criterion=criterion,best_step=0,best_variant='native_frozen_checkpoint',history=[],
        source_sha256=s.sha(Path(__file__)),reference_report=args.reference_report,
        role='Twenty-seven declared development populations, including the explicitly exposed older assessments. The current protected assessment is not read.')
    assert not (s.HERE/(args.name+'.json')).exists();s.write(args.name+'.json',report)
    print(json.dumps(dict(status='completed',criterion=criterion,populations={k:v['selection_accuracy'] for k,v in metrics.items()})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--name',required=True)
    parser.add_argument('--reference-report');parser.add_argument('--reference-variant',choices=['raw','moving_average'],default='moving_average');args=parser.parse_args()
    torch.set_num_threads(1);budget=Budget(args.name.lower(),300,gpu=True);status='failed_or_partial'
    try:main(args,budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
