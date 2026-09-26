"""Check whether small Levi blocks can detect the retained neural errors."""
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import session as s
from session import c
from exact_roots import root_targets
from expanded_decomposition import bank
from richardson_features import permutations
from full_resources import Budget


def cost(pair):
    return 1+sum(len(c.prior.c.a.original.desc(c.prior.c.a.original.inv(tuple(pair[k])))) for k in ['u','v'])


def main(budget,plan):
    trim=c.prior.c.a.original.trim;results=[];all_records=[];all_roots=[]
    data,names,_=bank()
    for group,anchor in enumerate(plan['anchors']):
        budget.check();pair=anchor['pair']
        for _ in range(cost(pair)):budget.exact_query()
        records,calls,terms,excluded=c.previous_generation.worker((pair,len(pair['u']),group))
        assert calls+1==cost(pair)
        wanted={trim(tuple(error['w'])) for error in anchor['errors']}
        keep=[i for i,row in enumerate(records) if trim(tuple(map(int,row['permutations'][2,:int(row['n'])]))) in wanted]
        records=records[keep];roots,info=root_targets(records,pair,group)
        index={trim(tuple(map(int,row['permutations'][2,:int(row['n'])]))):i for i,row in enumerate(records)}
        assert set(index)==wanted
        errors=[]
        for error in anchor['errors']:
            k=index[trim(tuple(error['w']))];row=records[k];q=roots[k]
            # Neighborhood factors are sorted, whereas diagnostic inputs retain
            # the predictor's orientation. Recover it from the frozen input.
            source_row=np.flatnonzero(data['population']==names.index(error['population']))[0]+error['row']
            original,n=permutations(torch.as_tensor(data['x'][source_row:source_row+1]))
            factors=[trim(tuple(map(int,original[0,side,:int(n[0])].tolist()))) for side in [0,1]]
            canonical=[trim(tuple(pair[key])) for key in ['u','v']]
            reverse=factors!=canonical
            assert factors==canonical[::-1] if reverse else factors==canonical
            assert row['coefficients'][int(reverse),error['supplied_index']]>0
            target=row['coefficients'][1-int(reverse)]
            assert target[error['predicted_index']]==0
            assert np.flatnonzero(target).tolist()==error['valid_indices']
            support=np.flatnonzero(q)+1;assert len(support)
            errors.append(dict(**error,reversed_relative_to_sorted_factors=reverse,Q=q.tolist(),support=support.tolist(),
                smallest_containing_interval_rank=int(support[-1]-support[0]+1),
                no_rank_at_most_three_block_contains_support=bool(support[-1]-support[0]+1>3)))
        results.append(dict(group=group,pair=pair,errors=errors,root_product=info))
        all_records.append(records);all_roots.append(roots)
    archive=s.ART/'ERROR_ROOT_SUPPORT.npz'
    np.savez_compressed(archive,records=np.concatenate(all_records),Q=np.concatenate(all_roots))
    entries=[error for group in results for error in group['errors']]
    result=dict(status='completed',source_sha256=s.sha(Path(__file__)),plan_sha256=s.sha(s.HERE/'NEIGHBOR_PLAN.json'),
        exact_queries=budget.exact_queries,groups=results,incorrect_supplied_inputs=len(entries),
        errors_invisible_to_all_rank_at_most_three_length_tests=sum(e['no_rank_at_most_three_block_contains_support'] for e in entries),
        support_interval_ranks={str(k):sum(e['smallest_containing_interval_rank']==k for e in entries) for k in sorted({e['smallest_containing_interval_rank'] for e in entries})},
        artifact_sha256=s.sha(archive),role='Exact diagnostic on already exposed error factors; these rows do not enter training')
    s.write('ERROR_ROOT_SUPPORT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='groups'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',default='exact_error_root_support');args=parser.parse_args()
    torch.set_num_threads(1);plan=json.loads((s.HERE/'NEIGHBOR_PLAN.json').read_text())
    budget=Budget(args.name,600,query_limit=sum(cost(a['pair']) for a in plan['anchors']));status='failed_or_partial'
    try:main(budget,plan);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
