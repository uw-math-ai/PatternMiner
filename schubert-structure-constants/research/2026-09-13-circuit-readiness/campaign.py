"""Train and assess a Schubert selector before circuit interpretation."""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from budget import Budget
HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent/'2026-09-13-witness-accuracy'
sys.path.insert(0, str(PREVIOUS))
import accuracy as a
import rank_followup as rank
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ART = HERE/'artifacts'
SEED = 20260914


def write(path, obj):
    a.write(path, obj)


class Selector(nn.Module):
    def __init__(self, multiplier=1, dropout=0., full_ranks=False):
        super().__init__()
        self.multiplier = multiplier
        self.full_ranks = full_ranks
        d = 96*multiplier
        self.input = nn.Linear(991 if full_ranks else 124, d)
        layer = nn.TransformerEncoderLayer(d, 4*multiplier, 192*multiplier,
                                           dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Linear(d, 1)

    def feature_map(self, x):
        features = [x, rank.rank_features(x)]
        if self.full_ranks:
            features.append(full_rank_features(x))
        return torch.cat(features, dim=-1)

    def forward(self, x):
        h = self.input(self.feature_map(x))
        return self.head(self.encoder(h, src_key_padding_mask=x.abs().sum(-1) == 0)).squeeze(-1)


def full_rank_features(x):
    """At candidate j, full rank matrices of u, s_j v, and w; no labels."""
    n = (1/x[:, 0, 18]).round().long()
    inverse = torch.stack([torch.cat([x[:, :1, k], x[:, :, k+1]], dim=1)
                           for k in (3, 5, 7)], dim=1)
    inverse = (inverse*n[:, None, None]).round().long()
    cuts = torch.arange(1, a.W+2, device=x.device)
    table = ((inverse[:, :, :, None] > 0) & (inverse[:, :, :, None] <= cuts)).to(torch.int32).cumsum(2, dtype=torch.int32)
    valid = cuts[None, :] <= n[:, None]
    table = table*valid[:, None, :, None]*valid[:, None, None, :]
    table = table.transpose(-2, -1)  # (batch, permutation, position p, value q)
    vpos = inverse[:, 1]
    delta = ((cuts[None, None, :] >= vpos[:, 1:, None]).to(torch.int32)
             -(cuts[None, None, :] >= vpos[:, :-1, None]).to(torch.int32))
    delta = delta*valid[:, None, :]
    diagonal = torch.eye(a.W+1, device=x.device, dtype=torch.int32)[:a.W]
    vj = table[:, 1, None, :, :]+delta[:, :, :, None]*diagonal[None, :, None, :]
    uu = table[:, 0, None, :, :].expand(-1, a.W, -1, -1)
    ww = table[:, 2, None, :, :].expand(-1, a.W, -1, -1)
    features = torch.stack([uu, vj, ww], dim=2).reshape(len(x), a.W, 867)
    return features.float()/n[:, None, None]*(x[:, :, :1] != 0)


def widen(state, factor):
    """Duplicate coordinates and heads, preserving each original head width."""
    out = {}
    for k, v in state.items():
        if k == 'input.weight':
            z = v.repeat(factor, 1)
        elif k == 'input.bias':
            z = v.repeat(factor)
        elif k == 'head.weight':
            z = v.repeat(1, factor)/factor
        elif k == 'head.bias':
            z = v
        elif k.endswith('in_proj_weight'):
            z = v.reshape(3, v.shape[1], v.shape[1]).repeat(1, factor, factor)/factor
            z = z.reshape(3*v.shape[1]*factor, v.shape[1]*factor)
        elif k.endswith('in_proj_bias'):
            z = v.reshape(3, -1).repeat(1, factor).reshape(-1)
        elif v.ndim == 2:
            z = v.repeat(factor, factor)/factor
        else:
            z = v.repeat(factor)
        out[k] = z
    return out


def model_from(path, multiplier=None, dropout=0., noise=0., full_ranks=None, order_bias=None, extra_depth=None):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    state = saved['state_dict']
    old_multiplier = state['input.weight'].shape[0]//96
    multiplier = old_multiplier if multiplier is None else multiplier
    assert multiplier % old_multiplier == 0
    if multiplier != old_multiplier:
        state = widen(state, multiplier//old_multiplier)
    full_ranks = state['input.weight'].shape[1] == 991 if full_ranks is None else full_ranks
    if full_ranks and state['input.weight'].shape[1] == 124:
        expanded = torch.zeros((96*multiplier, 991), dtype=state['input.weight'].dtype)
        expanded[:, :124] = state['input.weight']
        state['input.weight'] = expanded
    order_bias = 'order_bias' in state if order_bias is None else order_bias
    if order_bias:
        from order_attention import OrderSelector
        saved_depth = max(int(k.split('.')[2]) for k in state if k.startswith('encoder.layers.'))-1
        extra_depth = saved_depth if extra_depth is None else extra_depth
        assert extra_depth >= saved_depth
        model = OrderSelector(multiplier, dropout, full_ranks, extra_depth).cuda()
        expanded = model.order_bias.detach().cpu().clone()
        if 'order_bias' in state:
            expanded[:len(state['order_bias'])] = state['order_bias']
        state['order_bias'] = expanded
        for k, v in model.state_dict().items():
            if k.startswith('encoder.layers.') and int(k.split('.')[2]) >= 2+saved_depth:
                state[k] = v.detach().cpu()
    else:
        model = Selector(multiplier, dropout, full_ranks).cuda()
    model.load_state_dict(state)
    if noise:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.ndim == 2:
                    param.add_(torch.randn_like(param)*noise)
    return model


def load(name):
    if name.startswith('fresh_'):
        return {k: np.load(HERE/'cache'/name/f'{k}.npy', mmap_mode='c') for k in ('x', 'mask', 'y', 'a', 'group')}
    return a.load(name)


def hard_records(names):
    membership = json.loads((PREVIOUS/'SPLITS.json').read_text())
    maps, offset = {}, 0
    for name in names:
        if name.startswith('fresh_'):
            continue
        spec = membership[name]
        maps[spec['source']] = {int(v): offset+2*i for i, v in enumerate(spec['indices'])}
        offset += len(spec['indices'])*2
    records = a.read(PREVIOUS/'artifacts/mined_training_errors.jsonl.gz')
    unique = sorted({(maps[e['source']][e['source_row_index']]+int(e['reverse']), e['supplied_i']-1)
                     for e in records if e['source'] in maps})
    return np.asarray(unique, dtype=np.int64), len(records)


def tensors(names):
    cpu = [load(name) for name in names]
    return {k: torch.as_tensor(np.concatenate([d[k] for d in cpu]), device='cuda')
            for k in ('x', 'mask', 'y', 'a')}


def predict(model, data, last=False):
    return a.predict(model, data, last)


def fixed_predict(model, tr, pairs):
    scores = []
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(pairs), 1024):
            idx, supplied = pairs[i:i+1024].unbind(1)
            x = a.change_witness(tr['x'][idx], supplied)
            scores.append(model(x).cpu().numpy())
    score = np.concatenate(scores)
    idx = pairs[:, 0]
    return score, tr['mask'][idx].cpu().numpy(), tr['y'][idx].cpu().numpy()


def validation(model, names):
    metrics = {}
    for name in names:
        data = load(name)
        for last in [False, True]:
            metrics[f"{name}_{'last' if last else 'first'}"] = a.original.metrics(predict(model, data, last), data['mask'], data['y'])
    return metrics


def refresh_errors(model, tr, budget):
    """Find confidently wrong fixed-witness proposals only in the training pool."""
    result = []
    model.eval()
    with torch.inference_mode():
        for last in [False, True]:
            margins, indices, supplied = [], [], []
            for start in range(0, len(tr['x']), 2048):
                budget.check()
                x, mask, y, aa = (tr[k][start:start+2048] for k in ['x', 'mask', 'y', 'a'])
                ii = (a.W-1-aa.flip(1).long().argmax(1)) if last else aa.long().argmax(1)
                x = a.change_witness(x, ii)
                score = model(x).masked_fill(~mask, -1e9)
                pred = score.argmax(1)
                wrong = ~y[torch.arange(len(x), device='cuda'), pred]
                difference = score.max(1).values-score.masked_fill(~y, -1e9).max(1).values
                margins.append(difference[wrong])
                indices.append(torch.arange(start, start+len(x), device='cuda')[wrong])
                supplied.append(ii[wrong])
            if sum(len(x) for x in indices):
                mm, ix, ii = torch.cat(margins), torch.cat(indices), torch.cat(supplied)
                keep = torch.argsort(mm, descending=True, stable=True)[:8192]
                result.append(torch.stack([ix[keep], ii[keep]], dim=1))
    if not result:
        return None
    return torch.unique(torch.cat(result), dim=0, sorted=True)


def train(args, budget):
    path = HERE/f'{args.arm}.json'
    if path.exists():
        raise RuntimeError('Choose a new arm name; previous runs are preserved')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    initial = Path(args.initial)
    model = model_from(initial, args.multiplier, args.dropout, args.noise,
                       args.full_ranks or None, args.order_bias or None, args.extra_depth)
    tr = tensors(args.pools)
    if args.reflect_probability:
        tr['symmetry_allowed'] = torch.as_tensor(np.concatenate([
            np.load(HERE/'cache'/name/'symmetry_allowed.npy') for name in args.pools]), device='cuda')
    hard, records = hard_records(args.pools)
    hard = torch.as_tensor(hard, device='cuda')
    assert tr['a'][hard[:, 0], hard[:, 1]].all()
    pairs = None
    if args.pair_fraction:
        assert not args.hard_fraction and not args.overfit
        pairs = torch.as_tensor(np.load(ART/'training_pairs.npz')['rank_columns'], device='cuda')
        assert not (tr['y'][pairs[:, 0]]&tr['y'][pairs[:, 1]]).any()
    validation_names = ['S7_validation', 'S8_validation']
    if 'fresh_train' in args.pools:
        validation_names.extend(['fresh_S9_validation', 'fresh_S10_validation'])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    batch_gen = torch.Generator(device='cuda').manual_seed(args.seed)
    supplied_gen = torch.Generator(device='cuda').manual_seed(args.seed+1)
    hard_gen = torch.Generator(device='cuda').manual_seed(args.seed+2)
    reflection_gen = torch.Generator(device='cuda').manual_seed(args.seed+3)
    pair_gen = torch.Generator(device='cuda').manual_seed(args.seed+4)
    run = dict(configuration=vars(args), parameters=sum(p.numel() for p in model.parameters()),
               initial_checkpoint_sha256=a.sha(initial), source_sha256=a.sha(Path(__file__)),
               validation=validation_names, raw_error_records=records, distinct_error_inputs=len(hard),
               status='running', history=[], refreshes=[], best_selection_mean=-1.)
    shutil.copyfile(Path(__file__), ART/f'{args.arm}_source.py')
    if hasattr(model, 'order_bias'):
        shutil.copyfile(HERE/'order_attention.py', ART/f'{args.arm}_order_source.py')
    start = time.monotonic()
    write(path, run)
    for step in range(1, args.steps+1):
        if step % 100 == 1:
            budget.check()
        model.train()
        idx = torch.randint(len(tr['x']), (256,), device='cuda', generator=batch_gen)
        ii = torch.multinomial(tr['a'][idx].float(), 1, generator=supplied_gen).squeeze(1)
        count = 256 if args.overfit else int(256*args.hard_fraction)
        if count:
            h = hard[torch.randint(len(hard), (count,), device='cuda', generator=hard_gen)]
            idx[:count], ii[:count] = h[:, 0], h[:, 1]
        pair_count = int(256*args.pair_fraction)//2
        if pair_count:
            selected_pairs = pairs[torch.randint(len(pairs), (pair_count,), device='cuda', generator=pair_gen)]
            idx[:2*pair_count] = selected_pairs.reshape(-1)
            ii[:2*pair_count] = tr['a'][idx[:2*pair_count]].long().argmax(1)
        x = a.change_witness(tr['x'][idx], ii)
        mask, y = tr['mask'][idx], tr['y'][idx]
        if args.reflect_probability:
            import symmetry
            chosen = (torch.rand(256, device='cuda', generator=reflection_gen) < args.reflect_probability)&tr['symmetry_allowed'][idx]
            if pair_count:
                chosen[1:2*pair_count:2] = chosen[:2*pair_count:2]
            if chosen.any():
                x[chosen], mask[chosen], y[chosen] = symmetry.reflect(x[chosen], mask[chosen], y[chosen])
        score = model(x)
        masked = score.masked_fill(~mask, -1e9)
        ranking = (torch.logsumexp(masked, 1)-torch.logsumexp(masked.masked_fill(~y, -1e9), 1)).mean()
        bce = ((F.binary_cross_entropy_with_logits(score, y.float(), reduction='none')*mask).sum(1)/mask.sum(1)).mean()
        loss = ranking+.25*bce
        contrast = score.new_tensor(0.)
        if pair_count:
            ss, tt = score[:2*pair_count:2], score[1:2*pair_count:2]
            ys, yt = y[:2*pair_count:2], y[1:2*pair_count:2]
            ms = torch.logsumexp(ss.masked_fill(~ys, -1e9), 1)-torch.logsumexp(ss.masked_fill(~yt, -1e9), 1)
            mt = torch.logsumexp(tt.masked_fill(~ys, -1e9), 1)-torch.logsumexp(tt.masked_fill(~yt, -1e9), 1)
            contrast = F.softplus(2.-(ms-mt)).mean()
            loss = loss+args.pair_loss_weight*contrast
        lr = .00001+(args.lr-.00001)*.5*(1+math.cos(math.pi*(step-1)/max(args.steps-1, 1)))
        for g in opt.param_groups:
            g['lr'] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 1000 == 0 or step == args.steps:
            if args.overfit:
                ss, mm, yy = fixed_predict(model, tr, hard)
                vm = dict(designated_fitting_set=a.original.metrics(ss, mm, yy))
            else:
                vm = validation(model, validation_names)
            criterion = float(np.mean([m['top1'] for m in vm.values()]))
            item = dict(step=step, loss=float(loss.detach()), ranking_loss=float(ranking.detach()),
                        bce=float(bce.detach()), paired_contrast_loss=float(contrast.detach()),
                        validation=vm, selection_mean=criterion,
                        elapsed_seconds=time.monotonic()-start)
            run['history'].append(item)
            if criterion > run['best_selection_mean']:
                run.update(best_selection_mean=criterion, best_step=step)
                torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                                configuration=vars(args), step=step, selection_mean=criterion), ART/f'{args.arm}.pt')
            write(path, run)
            print(json.dumps(dict(arm=args.arm, **item)), flush=True)
        if args.refresh and step % args.refresh == 0 and step < args.steps and not args.overfit:
            refreshed = refresh_errors(model, tr, budget)
            if refreshed is not None:
                hard = refreshed
            run['refreshes'].append(dict(step=step, distinct_fixed_witness_errors=len(hard)))
            write(path, run)
    run.update(status='completed', seconds=time.monotonic()-start, checkpoint_sha256=a.sha(ART/f'{args.arm}.pt'))
    write(path, run)


def assess(args, budget):
    path = HERE/f'{args.arm}_ASSESSMENT.json'
    if path.exists():
        raise RuntimeError('This assessment is already recorded')
    model = model_from(ART/f'{args.arm}.pt')
    records, raw = {}, {}
    for name in ['S7_assessment', 'S8_assessment', 'S9_assessment', 'matched_pairs']:
        data = load(name)
        for last in [False, True]:
            budget.check()
            variant = 'last' if last else 'first'
            score = predict(model, data, last)
            raw[f'{name}_{variant}'] = score
            if not last:
                for k in ['mask', 'y', 'group']:
                    raw[f'{name}_{k}'] = data[k]
            m = a.metrics(score, data)
            if name == 'matched_pairs':
                pred = np.where(data['mask'], score, -1e20).argmax(1)
                good = data['y'][np.arange(len(score)), pred]
                m.update(pairs=len(score)//2, both_correct=int((good[::2]&good[1::2]).sum()),
                         both_correct_rate=float((good[::2]&good[1::2]).mean()))
            records[f'{name}_{variant}'] = m
    out = dict(arm=args.arm, metrics=records,
               gate_met=all(m['top1'] >= .99 and m.get('both_correct_rate', 1.) >= .99 for m in records.values()),
               exposure='Historical product groups excluded from training and validation; adaptive research assessment')
    np.savez_compressed(HERE/f'{args.arm}_scores.npz', **raw)
    fits = {}
    for name in args.pools:
        data = load(name)
        for last in [False, True]:
            budget.check()
            fits[f"{name}_{'last' if last else 'first'}"] = a.metrics(predict(model, data, last), data)
    out['training_fit'] = fits
    write(path, out)
    print(json.dumps(out), flush=True)


def check(args, budget):
    data = load('S7_validation')
    x = torch.as_tensor(data['x'][:64], device='cuda')
    original = rank.init_model(PREVIOUS/'artifacts/rank_matrices.pt').eval()
    checks = {}
    for multiplier in [1, 2]:
        model = model_from(PREVIOUS/'artifacts/rank_matrices.pt', multiplier).eval()
        with torch.inference_mode():
            error = float((model(x)-original(x)).abs().max())
        assert error < 2e-5
        checks[str(multiplier)] = dict(max_error=error, head_width=24,
                                       parameters=sum(p.numel() for p in model.parameters()))
    pairs, n = hard_records(['S7_train', 'S8_train'])
    checks['mined_records'] = dict(raw=n, distinct_fixed_witness_inputs=len(pairs))
    write(HERE/'IMPLEMENTATION_CHECKS.json', checks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['train', 'assess', 'check'])
    parser.add_argument('--arm', default='implementation')
    parser.add_argument('--initial', default=str(PREVIOUS/'artifacts/rank_matrices.pt'))
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--hard-fraction', type=float, default=0.)
    parser.add_argument('--overfit', action='store_true')
    parser.add_argument('--multiplier', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--noise', type=float, default=0.)
    parser.add_argument('--full-ranks', action='store_true')
    parser.add_argument('--reflect-probability', type=float, default=0.)
    parser.add_argument('--order-bias', action='store_true')
    parser.add_argument('--extra-depth', type=int)
    parser.add_argument('--pair-fraction', type=float, default=0.)
    parser.add_argument('--pair-loss-weight', type=float, default=.25)
    parser.add_argument('--lr', type=float, default=.0003)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--refresh', type=int, default=0)
    parser.add_argument('--pools', nargs='+', default=['S7_train', 'S8_train'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    ART.mkdir(exist_ok=True)
    budget = Budget(f'{args.arm}_{args.stage}', gpu=True)
    status = 'failed_or_partial'
    try:
        globals()[args.stage](args, budget)
        status = 'completed'
    finally:
        print(json.dumps(budget.finish(status)), flush=True)
