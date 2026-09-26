"""Measure explicit Levi restrictions against retained development predictions."""
import json
from pathlib import Path
import numpy as np
import torch
import session as s
from expanded_decomposition import bank
from full_resources import Budget


def main(budget):
    data,names,_=bank()
    feature_report=json.loads((s.HERE/'LEVI_FEATURE_CHECKS.json').read_text())
    feature_archive=s.ART/'LEVI_FEATURES_DEVELOPMENT.npz'
    assert s.sha(feature_archive)==feature_report['artifact_sha256']
    features=np.load(feature_archive)['features']
    forbidden=(features[:,:,6::8]>1e-6).any(-1)&data['mask']
    assert not (forbidden&data['y']).any()
    prior=json.loads((s.PREVIOUS/'ROOT_ATTENTION_ABLATION.json').read_text())
    archive=s.PREVIOUS/'artifacts/ROOT_ATTENTION_ABLATION.npz'
    assert s.sha(archive)==prior['artifact_sha256']
    saved=dict(np.load(archive));old_names=list(prior['populations'])
    rows=np.asarray([np.flatnonzero(data['population']==names.index(old_names[p]))[0]+r for p,r,i in saved['examples']])
    mask=data['mask'][rows];y=data['y'][rows];valid=mask&~forbidden[rows]
    assert valid.any(1).all()
    full=np.where(mask,saved['full_scores'],-1e9).argmax(1)
    filtered=np.where(valid,saved['full_scores'],-1e9).argmax(1)
    good=y[np.arange(len(rows)),full];after=y[np.arange(len(rows)),filtered]
    populations={}
    for population,name in enumerate(old_names):
        include=saved['examples'][:,0]==population
        r=saved['examples'][include,1];count=int((data['population']==names.index(name)).sum())
        wrong=np.bincount(r,weights=~good[include],minlength=count)>0
        wrong_after=np.bincount(r,weights=~after[include],minlength=count)>0
        pair=~wrong[::2]&~wrong[1::2];pair_after=~wrong_after[::2]&~wrong_after[1::2]
        populations[name]=dict(pairs=len(pair),correct_before=int(pair.sum()),correct_after=int(pair_after.sum()),
            accuracy_before=float(pair.mean()),accuracy_after=float(pair_after.mean()))
    output=s.ART/'LEVI_RESTRICTION_DIAGNOSTIC.npz'
    np.savez_compressed(output,rows=rows,excluded=forbidden,unfiltered_choices=full,filtered_choices=filtered,correct_before=good,correct_after=after)
    report=dict(status='completed',role='Explicit theorem-based filtering of already exposed predictions; not a newly trained model or independent qualification',
        source_sha256=s.sha(Path(__file__)),checkpoint_sha256=prior['checkpoint_sha256'],
        prediction_archive_sha256=s.sha(archive),feature_report_sha256=s.sha(s.HERE/'LEVI_FEATURE_CHECKS.json'),
        inputs=len(rows),errors_before=int((~good).sum()),errors_after=int((~after).sum()),
        corrected=int((~good&after).sum()),lost=int((good&~after).sum()),
        wrong_choices_excluded=int(forbidden[rows,full][~good].sum()),populations=populations,artifact_sha256=s.sha(output))
    assert report['lost']==0
    s.write('LEVI_RESTRICTION_DIAGNOSTIC.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);budget=Budget('levi_restriction_prediction_diagnostic',120);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
