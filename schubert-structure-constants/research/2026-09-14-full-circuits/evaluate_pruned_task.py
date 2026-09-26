"""Evaluate the frozen task-preserving head mask on all 27 development criteria."""
import json
from pathlib import Path
import time
import torch
import session as s
import richardson_model as rm
import circuit_forward as cf
import second_training_data as td
from exposed_assessment import append_validation
from train_neighbors import evaluate
from prune_development_task_heads import zeroed_heads
from full_resources import Budget


class MaskedModel(torch.nn.Module):
    def __init__(self,model,removed):
        super().__init__();self.model=model;self.patches=zeroed_heads(removed)
    def forward(self,x):return cf.explicit(self.model,x,patches=self.patches)


def main(budget):
    selection=json.loads((s.HERE/'TASK_HEAD_PRUNING.json').read_text());assert selection['status']=='completed'
    checkpoint=s.ART/'SECOND_REFINEMENT_READOUTS.pt';digest=s.sha(checkpoint)
    assert digest==selection['checkpoint_sha256']
    model=MaskedModel(rm.load(checkpoint),selection['removed_head_pairs']).eval()
    data=append_validation(td.validation(True));assert len(data)==27
    values,criterion=evaluate(model,data)
    report=dict(status='completed',checkpoint_sha256=digest,selection_sha256=s.sha(s.HERE/'TASK_HEAD_PRUNING.json'),
        source_sha256=s.sha(Path(__file__)),retained_heads=selection['retained_heads'],criterion=criterion,populations=values,
        passes_development_requirement=criterion[0]>=.99,independently_qualified=False,
        scope='All 27 development criteria, every positive supplied index. The mask was fixed before this evaluation. Exposed assessment replay is not new independent qualification.')
    assert not (s.HERE/'TASK_HEAD_PRUNING_NATIVE_VALIDATION.json').exists()
    s.write('TASK_HEAD_PRUNING_NATIVE_VALIDATION.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    while True:
        if (s.HERE/'TASK_HEAD_PRUNING.json').exists():
            active=json.loads((s.HERE/'RESOURCE_RESERVATIONS.json').read_text())
            if 'development_task_head_pruning' not in active:break
        time.sleep(2)
    torch.set_num_threads(1);budget=Budget('task_head_pruning_native_validation',300,gpu=True);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
