"""Bounded, product-group-separated continuations of the Schubert selector.

The experiment proposal fixes the schedule and selection before assessment.
Existing exact support labels are supervision only. A supplied valid index is
the sole label-derived input, exactly as in the historical task.
"""
import argparse
from functools import lru_cache
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'2026-09-08-theorem-search'
TRIAL = HERE.parent/'2026-09-13-autoresearch-circuits'
sys.path.insert(0, str(OLD))
import witness_model as original
from budget import Budget

SEED = 20260913
W = original.WIDTH
CACHE = HERE/'cache'
ART = HERE/'artifacts'
NAMES = {n: f'stable_degree_one_S{n}_{count}_length{length}.jsonl.gz'
         for n, count, length in [(7, 3000, 7), (8, 2000, 9), (9, 500, 11)]}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2)+'\n')
    tmp.replace(path)


def read(path):
    with gzip.open(path, 'rt') as f:
        return [json.loads(line) for line in f]


@lru_cache(maxsize=100000)
def permutation_data(p):
    return original.inv(p), original.length(p)


def encode(rows, both=True, last=False):
    """Same 22 input features as original.encode; all ambient positions retained."""
    count = len(rows)*(2 if both else 1)
    x = np.zeros((count, W, 22), np.float32)
    mask = np.zeros((count, W), bool)
    y = np.zeros_like(mask)
    a = np.zeros_like(mask)
    group = np.zeros(count, np.int32)
    group_ids = {}
    for r, row in enumerate(rows):
        n = len(row['w'])
        assert n-1 <= W
        g = original.group(row)
        gid = group_ids.setdefault(g, len(group_ids))
        pdata = [permutation_data(tuple(row[k])) for k in ('u', 'v', 'w')]
        for rev in range(2 if both else 1):
            k = r*2+rev if both else r
            ps = [pdata[t] for t in ((1, 0, 2) if rev else (0, 1, 2))]
            p = np.array([z[0] for z in ps], dtype=np.float64)
            lengths = np.array([z[1] for z in ps])
            aa, bb = (row['B'], row['A']) if rev else (row['A'], row['B'])
            a[k, :n-1] = np.asarray(aa[:n-1]) > 0
            y[k, :n-1] = np.asarray(bb[:n-1]) > 0
            i = np.flatnonzero(a[k])[-1 if last else 0]
            j = np.arange(n-1)
            z = np.empty((n-1, 22), np.float64)
            z[:, 0] = (j+1)/n
            z[:, 1] = (i+1)/n
            z[:, 2] = (j-i)/n
            z[:, 3:9] = np.stack([p[:, :-1].T, p[:, 1:].T], axis=-1).reshape(n-1, 6)/n
            z[:, 9:15] = p[:, i:i+2].reshape(6)/n
            z[:, 15] = p[0, :-1] > p[0, 1:]
            z[:, 16] = p[2, :-1] > p[2, 1:]
            z[:, 17] = j == i
            z[:, 18] = 1/n
            z[:, 19:22] = lengths/(n*n)
            x[k, :n-1] = z
            mask[k, :n-1] = p[1, :-1] > p[1, 1:]
            group[k] = gid
    assert a.any(1).all() and y.any(1).all()
    assert (~y | mask).all()
    return dict(x=x, mask=mask, y=y, a=a, group=group)


def change_witness(x, indices):
    x = x.clone()
    active = x[:, :, 18] != 0
    nrecip = x[:, :1, 18]
    x[:, :, 1] = (indices[:, None]+1)*nrecip
    x[:, :, 2] = x[:, :, 0]-x[:, :, 1]
    x[:, :, 9:15] = x[torch.arange(len(x), device=x.device), indices, 3:9][:, None, :]
    x[:, :, 17] = torch.arange(W, device=x.device)[None, :] == indices[:, None]
    return x*active[:, :, None]


def save_data(name, rows, both=True):
    data = encode(rows, both=both)
    directory = CACHE/name
    directory.mkdir(parents=True, exist_ok=True)
    for k, v in data.items():
        np.save(directory/f'{k}.npy', v)
    return dict(rows=len(rows), examples=len(data['x']),
                product_groups=len(set(map(original.group, rows))),
                array_sha256={k: sha(directory/f'{k}.npy') for k in data})


def prepare(budget):
    CACHE.mkdir(exist_ok=True)
    ART.mkdir(exist_ok=True)
    if (HERE/'DATA.json').exists():
        raise RuntimeError('Preparation already recorded')
    source = {}
    for n, name in NAMES.items():
        shutil.copyfile(OLD/name, ART/name)
        source[n] = read(ART/name)
        budget.check()
    groups = {n: set(map(original.group, rows)) for n, rows in source.items()}
    pools = {7: source[7], 8: [r for r in source[8] if original.group(r) not in groups[7]]}
    rng = random.Random(SEED)
    rows = {}
    for n in (7, 8):
        for label, buckets, cap in [('train', set(range(7)), None), ('validation', {7}, 5000),
                                    ('assessment', {8, 9}, 10000)]:
            rr = [r for r in pools[n] if original.bucket(original.group(r)) in buckets]
            if n == 8 and cap is not None and len(rr) > cap:
                rr = rng.sample(rr, cap)
            rows[f'S{n}_{label}'] = rr
    excluded = groups[7] | groups[8]
    s9 = read(TRIAL/'artifacts/evaluation_S9.jsonl.gz')
    rows['S9_assessment'] = [r for r in s9 if original.group(r) not in excluded]
    pairs = json.loads((TRIAL/'CONTEXT_PAIRS.json').read_text())['pairs']
    pairs = [pair for pair in pairs if all(original.group(e['row']) not in excluded for e in pair)]
    rows['matched_pairs'] = [e['row'] for pair in pairs for e in pair]
    train_groups = set(map(original.group, rows['S7_train']+rows['S8_train']))
    validation_groups = set(map(original.group, rows['S7_validation']+rows['S8_validation']))
    assert not train_groups & validation_groups
    for name, rr in rows.items():
        if 'assessment' in name or name == 'matched_pairs':
            assert not (train_groups | validation_groups) & set(map(original.group, rr))
    checks = []
    sample = rng.sample(source[7], 8)+rng.sample(source[8], 8)+rng.sample(source[9], 8)
    for last in (False, True):
        data = encode(sample, last=last)
        historical = original.arrays(sample, last=last)
        np.testing.assert_array_equal(data['x'], historical[0])
        np.testing.assert_array_equal(data['mask'], historical[1])
        np.testing.assert_array_equal(data['y'], historical[2])
        checks.append(dict(encoder='last' if last else 'first', examples=len(data['x']), bitwise_equal=True))
    first, last = encode(sample), encode(sample, last=True)
    indices = W-1-np.argmax(last['a'][:, ::-1], axis=1)
    changed = change_witness(torch.from_numpy(first['x']), torch.from_numpy(indices)).numpy()
    error = float(np.max(np.abs(changed-last['x'])))
    assert error < 1e-7
    checks.append(dict(augmentation_max_error=error, examples=len(changed)))
    manifest = dict(seed=SEED, encoder_checks=checks, train_validation_overlap=0,
                    assessment_train_or_validation_overlap=0,
                    S9_excluded_rows=len(s9)-len(rows['S9_assessment']),
                    matched_pairs=len(pairs), sources={name: sha(ART/name) for name in NAMES.values()},
                    split_rule='historical witness-v1 bucket; complete unordered stable factor pairs',
                    exposure='Historical datasets and challenges, not an untouched test', splits={})
    # Retain exact row membership in source order, rather than only a seed.
    lookup = {n: {id(r): i for i, r in enumerate(source[n])} for n in (7, 8)}
    membership = {}
    for name, rr in rows.items():
        if name.startswith(('S7', 'S8')):
            n = int(name[1])
            membership[name] = dict(source=NAMES[n], indices=[lookup[n][id(r)] for r in rr])
        elif name == 'S9_assessment':
            membership[name] = dict(source='../2026-09-13-autoresearch-circuits/artifacts/evaluation_S9.jsonl.gz',
                                    indices=[i for i, r in enumerate(s9) if original.group(r) not in excluded])
        else:
            membership[name] = dict(source='../2026-09-13-autoresearch-circuits/CONTEXT_PAIRS.json',
                                    rows_are_already_oriented=True, pair_count=len(pairs))
        budget.check()
        manifest['splits'][name] = save_data(name, rr, both=name != 'matched_pairs')
        print(json.dumps(dict(prepared=name, **manifest['splits'][name])), flush=True)
    write(HERE/'SPLITS.json', membership)
    write(HERE/'DATA.json', manifest)


def load(name):
    return {k: np.load(CACHE/name/f'{k}.npy', mmap_mode='c') for k in ('x', 'mask', 'y', 'a', 'group')}


def predict(model, data, last=False):
    model.eval()
    chunks = []
    with torch.inference_mode():
        for i in range(0, len(data['x']), 1024):
            x = torch.as_tensor(data['x'][i:i+1024], device='cuda')
            if last:
                a = data['a'][i:i+1024]
                indices = W-1-np.argmax(a[:, ::-1], axis=1)
                x = change_witness(x, torch.as_tensor(indices, device='cuda'))
            chunks.append(model(x).cpu().numpy())
    return np.concatenate(chunks)


def metrics(score, data):
    mask, y, group = data['mask'], data['y'], data['group']
    result = original.metrics(score, mask, y)
    pred = np.where(mask, score, -1e20).argmax(1)
    correct = y[np.arange(len(y)), pred]
    result['correct'] = int(correct.sum())
    result['errors'] = int((~correct).sum())
    counts = np.bincount(group)
    successes = np.bincount(group, weights=correct)
    result['product_groups'] = len(counts)
    result['mean_product_group_top1'] = float(np.mean(successes/counts))
    for name, subset in [('multiple_candidates', mask.sum(1) > 1),
                         ('contains_invalid_candidate', mask.sum(1) > y.sum(1))]:
        result[name] = dict(n=int(subset.sum()), correct=int(correct[subset].sum()),
                            top1=float(correct[subset].mean()) if subset.any() else None)
    # Groups, not rows, are the resampling units. This does not correct adaptive exposure.
    draws = np.random.default_rng(SEED).integers(len(counts), size=(1000, len(counts)))
    boot = successes[draws].sum(1)/counts[draws].sum(1)
    result['product_cluster_bootstrap_95_percent'] = np.quantile(boot, [.025, .975]).tolist()
    return result


def init_model(path):
    model = original.Selector().cuda()
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['state_dict'])
    return model


def train_arm(arm, initial, mixed, support, steps, budget):
    result_path = HERE/f'{arm}.json'
    if result_path.exists():
        raise RuntimeError('Do not overwrite a recorded training run')
    torch.manual_seed(SEED)
    model = init_model(initial)
    names = ['S7_train', 'S8_train'] if mixed else ['S7_train']
    cpu = [load(name) for name in names]
    tr = {k: torch.as_tensor(np.concatenate([d[k] for d in cpu]), device='cuda')
          for k in ('x', 'mask', 'y', 'a')}
    del cpu
    val = {name: load(name) for name in (['S7_validation', 'S8_validation'] if mixed else ['S7_validation'])}
    opt = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01)
    gen = torch.Generator(device='cuda').manual_seed(SEED)
    witness_gen = torch.Generator(device='cuda').manual_seed(SEED+1)
    history = []
    best = -1.
    start = time.monotonic()
    run = dict(arm=arm, seed=SEED, steps=steps, support_loss_weight=.25 if support else 0,
               supplied_witness='uniform valid' if support else 'first valid',
               training=names, checkpoint_selection=list(val), initial_checkpoint_sha256=sha(initial),
               parameters=sum(p.numel() for p in model.parameters()), history=history,
               batch_size=256, learning_rate_start=.0003, learning_rate_end=.00001,
               optimizer='AdamW', weight_decay=.01, source_sha256=sha(Path(__file__)),
               proposal_sha256=sha(HERE/'EXPERIMENT.md'), status='running')
    write(result_path, run)
    for step in range(1, steps+1):
        if step % 100 == 1:
            budget.check()
        model.train()
        idx = torch.randint(len(tr['x']), (256,), generator=gen, device='cuda')
        x, mask, y = (tr[k][idx] for k in ('x', 'mask', 'y'))
        if support:
            i = torch.multinomial(tr['a'][idx].float(), 1, generator=witness_gen).squeeze(1)
            x = change_witness(x, i)
        score = model(x)
        masked = score.masked_fill(~mask, -1e9)
        ranking = (torch.logsumexp(masked, 1)-torch.logsumexp(masked.masked_fill(~y, -1e9), 1)).mean()
        auxiliary = (F.binary_cross_entropy_with_logits(score, y.float(), reduction='none')*mask).sum(1)/mask.sum(1)
        loss = ranking+(.25*auxiliary.mean() if support else 0)
        lr = .00001+(.0003-.00001)*.5*(1+math.cos(math.pi*(step-1)/max(steps-1, 1)))
        for g in opt.param_groups:
            g['lr'] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 1000 == 0 or step == steps:
            vm = {f'{name}_{variant}': original.metrics(predict(model, d, last=variant == 'last'), d['mask'], d['y'])
                  for name, d in val.items() for variant in ('first', 'last')}
            criterion = float(np.mean([m['top1'] for m in vm.values()]))
            item = dict(step=step, training_loss=float(loss.detach()), ranking_loss=float(ranking.detach()),
                        candidate_bce=float(auxiliary.mean().detach()), validation=vm,
                        selection_mean=criterion, elapsed_seconds=time.monotonic()-start)
            history.append(item)
            if criterion > best:
                best = criterion
                run.update(best_step=step, best_selection_mean=best)
                torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                                arm=arm, step=step, selection_mean=best), ART/f'{arm}.pt')
            write(result_path, run)
            print(json.dumps(dict(arm=arm, **item)), flush=True)
    run.update(status='completed', seconds=time.monotonic()-start, checkpoint_sha256=sha(ART/f'{arm}.pt'))
    write(result_path, run)
    return run


def train(budget):
    initial = TRIAL/'artifacts/witness_model_seed0.pt'
    a = train_arm('longer_ranking', initial, False, False, 6000, budget)
    b = train_arm('support_supervision', initial, False, True, 6000, budget)
    selected = max([a, b], key=lambda r: r['best_selection_mean'])
    write(HERE/'SELECTION.json', dict(selected=selected['arm'], selection_mean=selected['best_selection_mean'],
                                    rule='mean S7 validation first/last accuracy; ties prefer ranking'))
    train_arm('mixed_sizes', ART/f"{selected['arm']}.pt", True,
              selected['arm'] == 'support_supervision', 8000, budget)


def assess(budget):
    if (HERE/'ASSESSMENT.json').exists():
        raise RuntimeError('Assessment already recorded')
    names = ['S7_assessment', 'S8_assessment', 'S9_assessment', 'matched_pairs']
    out = dict(threshold=.99, selection='completed before assessment', models={})
    for arm in ['historical', 'longer_ranking', 'support_supervision', 'mixed_sizes']:
        path = TRIAL/'artifacts/witness_model_seed0.pt' if arm == 'historical' else ART/f'{arm}.pt'
        model = init_model(path)
        values, raw = {}, {}
        for name in names:
            data = load(name)
            for last in (False, True):
                budget.check()
                variant = 'last' if last else 'first'
                score = predict(model, data, last)
                raw[f'{name}_{variant}'] = score
                if not last:
                    for k in ('mask', 'y', 'group'):
                        raw[f'{name}_{k}'] = data[k]
                result = metrics(score, data)
                if name == 'matched_pairs':
                    assert len(score) % 2 == 0
                    assert np.all((data['y'][0::2] & data['y'][1::2]).sum(1) == 0)
                    pred = np.where(data['mask'], score, -1e20).argmax(1)
                    correct = data['y'][np.arange(len(score)), pred]
                    result['pairs'] = len(score)//2
                    result['both_correct'] = int((correct[::2] & correct[1::2]).sum())
                    result['both_correct_rate'] = float((correct[::2] & correct[1::2]).mean())
                values[f'{name}_{variant}'] = result
        values['gate_met'] = all(m['top1'] >= .99 and m.get('both_correct_rate', 1) >= .99 for m in values.values())
        np.savez_compressed(HERE/f'{arm}_scores.npz', **raw)
        out['models'][arm] = values
        print(json.dumps(dict(assessed=arm, metrics=values)), flush=True)
    out['selected_final_model'] = 'mixed_sizes'
    out['gate_met'] = out['models']['mixed_sizes']['gate_met']
    out['exposure'] = 'Product groups excluded from optimization and selection; historical and adaptively studied sources'
    write(HERE/'ASSESSMENT.json', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'train', 'assess'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    budget = Budget(f'accuracy_{args.stage}', gpu=args.stage != 'prepare')
    status = 'failed_or_partial'
    try:
        globals()[args.stage](budget)
        status = 'completed'
    finally:
        print(json.dumps(budget.finish(status)), flush=True)
