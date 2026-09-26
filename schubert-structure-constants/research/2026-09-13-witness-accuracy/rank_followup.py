"""Controlled continuation with explicit permutation rank matrices."""
import json
import numpy as np
import torch
from torch import nn
import accuracy as a
from budget import Budget

USE_RANK = False


def rank_features(x):
    """At token j: r_z(p,j), r_z(p,j+1), ordered by z then column then p."""
    n = (1/x[:, 0, 18]).round().long()
    inverse = torch.stack([torch.cat([x[:, :1, k], x[:, :, k+1]], dim=1)
                           for k in (3, 5, 7)], dim=1)
    inverse = (inverse*n[:, None, None]).round().long()
    cuts = torch.arange(1, a.W+2, device=x.device)
    table = ((inverse[:, :, :, None] > 0) & (inverse[:, :, :, None] <= cuts)).long().cumsum(2)
    table = table.float()/n[:, None, None, None]
    table = table*(cuts[None, None, None, :] <= n[:, None, None, None])
    features = torch.stack([table[:, :, :-1, :], table[:, :, 1:, :]], dim=3)
    features = features.permute(0, 2, 1, 3, 4).reshape(len(x), a.W, 102)
    return features*(x[:, :, :1] != 0)


class RankSelector(a.original.Selector):
    def __init__(self, use_rank):
        super().__init__()
        self.input = nn.Linear(124, 96)
        self.use_rank = use_rank

    def forward(self, x):
        ranks = rank_features(x)
        if not self.use_rank:
            ranks = torch.zeros_like(ranks)
        return self.head(self.encoder(self.input(torch.cat([x, ranks], dim=-1)),
                         src_key_padding_mask=x.abs().sum(-1) == 0)).squeeze(-1)


def init_model(path):
    use_rank = True if path.stem == 'rank_matrices' else False if path.stem == 'rank_control' else USE_RANK
    model = RankSelector(use_rank).cuda()
    state = torch.load(path, map_location='cpu', weights_only=True)['state_dict']
    if state['input.weight'].shape[1] == 22:
        expanded = torch.zeros((96, 124), dtype=state['input.weight'].dtype)
        expanded[:, :22] = state['input.weight']
        state['input.weight'] = expanded
    model.load_state_dict(state)
    return model


def checks():
    rows = a.read(a.TRIAL/'artifacts/evaluation_S9.jsonl.gz')[:32]
    data = a.encode(rows)
    x = torch.as_tensor(data['x'], device='cuda')
    got = rank_features(x).cpu().numpy()
    expected = np.zeros_like(got)
    for k, row in enumerate(rows):
        n = len(row['w'])
        for rev in [0, 1]:
            ps = [row[name] for name in (('v', 'u', 'w') if rev else ('u', 'v', 'w'))]
            for j in range(n-1):
                values = []
                for z in ps:
                    for q in [j+1, j+2]:
                        values.extend(sum(v <= q for v in z[:p])/n if p <= n else 0
                                      for p in range(1, a.W+2))
                expected[2*k+rev, j] = values
    np.testing.assert_array_equal(got, expected)
    old = a.original.Selector().cuda()
    initial = a.ART/'mixed_sizes.pt'
    old.load_state_dict(torch.load(initial, map_location='cpu', weights_only=True)['state_dict'])
    old.eval()
    errors = {}
    for mode in [False, True]:
        model = init_model(initial)
        model.use_rank = mode
        model.eval()
        with torch.inference_mode():
            delta = float((old(x)-model(x)).abs().max())
        assert delta < 1e-5
        errors[str(mode)] = delta
    return dict(rank_entries_checked=int(expected.size), examples=len(x),
                direct_integer_counting_bitwise_equal=True, initial_model_max_errors=errors)


def assess(arm, budget):
    model = init_model(a.ART/f'{arm}.pt')
    values, raw = {}, {}
    for name in ['S7_assessment', 'S8_assessment', 'S9_assessment', 'matched_pairs']:
        data = a.load(name)
        for last in [False, True]:
            budget.check()
            variant = 'last' if last else 'first'
            score = a.predict(model, data, last)
            raw[f'{name}_{variant}'] = score
            if not last:
                for k in ['mask', 'y', 'group']:
                    raw[f'{name}_{k}'] = data[k]
            result = a.metrics(score, data)
            if name == 'matched_pairs':
                pred = np.where(data['mask'], score, -1e20).argmax(1)
                correct = data['y'][np.arange(len(score)), pred]
                result.update(pairs=len(score)//2,
                              both_correct=int((correct[::2] & correct[1::2]).sum()),
                              both_correct_rate=float((correct[::2] & correct[1::2]).mean()))
            values[f'{name}_{variant}'] = result
    values['gate_met'] = all(m['top1'] >= .99 and m.get('both_correct_rate', 1) >= .99 for m in values.values())
    np.savez_compressed(a.HERE/f'{arm}_scores.npz', **raw)
    model = init_model(a.ART/f'{arm}.pt')
    replay = a.predict(model, a.load('S9_assessment'))
    np.testing.assert_array_equal(replay, raw['S9_assessment_first'])
    a.write(a.HERE/f'{arm}_REPLAY.json', dict(checkpoint_sha256=a.sha(a.ART/f'{arm}.pt'),
                                            examples=len(replay), scores_bitwise_equal=True))
    fits = {}
    for name in ['S7_train', 'S8_train']:
        data = a.load(name)
        for last in [False, True]:
            budget.check()
            fits[f"{name}_{'last' if last else 'first'}"] = a.metrics(a.predict(model, data, last), data)
    a.write(a.HERE/f'{arm}_FIT.json', fits)
    print(json.dumps(dict(assessed=arm, metrics=values)), flush=True)
    return values


def main(budget):
    global USE_RANK
    if (a.HERE/'RANK_ASSESSMENT.json').exists():
        raise RuntimeError('Do not overwrite completed results')
    a.write(a.HERE/'RANK_CHECKS.json', checks())
    a.write(a.HERE/'RANK_PROVENANCE.json', dict(source_sha256=a.sha(a.HERE/'rank_followup.py'),
                                              proposal_sha256=a.sha(a.HERE/'RANK_PROPOSAL.md'),
                                              shared_training_source_sha256=a.sha(a.HERE/'accuracy.py')))
    a.init_model = init_model
    runs = [json.loads((a.HERE/'mixed_sizes.json').read_text())]
    for arm, mode in [('rank_control', False), ('rank_matrices', True)]:
        USE_RANK = mode
        runs.append(a.train_arm(arm, a.ART/'mixed_sizes.pt', True, True, 10000, budget))
    selected = max(runs, key=lambda r: r['best_selection_mean'])
    a.write(a.HERE/'RANK_SELECTION.json', dict(selected=selected['arm'],
                                             mean_validation_accuracy=selected['best_selection_mean'],
                                             rule='mean S7/S8 validation first/last accuracy; ties prefer earlier arm'))
    out = dict(models={arm: assess(arm, budget) for arm in ['rank_control', 'rank_matrices']},
               selected_final_model=selected['arm'], threshold=.99,
               exposure='Adaptive diagnostic assessment; same product exclusions as the initial comparison')
    out['gate_met'] = (out['models'][selected['arm']]['gate_met'] if selected['arm'] in out['models']
                       else json.loads((a.HERE/'ASSESSMENT.json').read_text())['gate_met'])
    a.write(a.HERE/'RANK_ASSESSMENT.json', out)


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    budget = Budget('accuracy_rank_followup', gpu=True)
    status = 'failed_or_partial'
    try:
        main(budget)
        status = 'completed'
    finally:
        print(json.dumps(budget.finish(status)))
