"""Validate and freeze the completed circuit experiments and cumulative ledger."""
import ast
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import circuit_context as s
from circuit_resources import Budget


def read(name):return json.loads((s.HERE/name).read_text())


def storage():
    seen=set();sizes={}
    for name in [*read('PRIOR_EVIDENCE_INTEGRITY.json')['prior_manifests'],s.HERE.name]:
        total=0
        for parent,dirs,files in os.walk(s.HERE.parent/name,followlinks=True):
            dirs[:]=[d for d in dirs if d!='__pycache__']
            for name2 in files:
                stat=(Path(parent)/name2).stat();key=(stat.st_dev,stat.st_ino)
                if key not in seen:seen.add(key);total+=stat.st_size
        sizes[name]=total
    return dict(scope='Seven experiment directories including symlinked caches; each physical inode counted once.',
        bytes_by_directory=sizes,total_bytes=sum(sizes.values()),
        root_free_bytes=os.statvfs(s.HERE).f_bavail*os.statvfs(s.HERE).f_frsize,
        data_volume_free_bytes=os.statvfs('/mnt/data').f_bavail*os.statvfs('/mnt/data').f_frsize)


def audit(budget):
    checkpoint=s.sha(s.CHECKPOINT)
    assert checkpoint=='19d12b2bcdb0ad6ccf4f40dd3e2ed89662beab425449daf7808af05785500894'
    files={'INTERFACE_CHECKS':'check_interfaces.py','DISCOVERY_SCAN':'circuit_scan.py',
        'HEAD_CONFIRMATION':'circuit_confirm.py','MLP_CONFIRMATION':'circuit_path.py',
        'PATH_TRACE':'trace_paths.py','PATH_TRACE_ALL_HEADS':'trace_all_heads.py',
        'MATCHED_PATH_TRACE':'trace_matched_features.py','ENDPOINT_CONFIRMATION':'endpoint_confirmation.py',
        'COALITIONS_DISCOVERY':'head_coalitions.py','COALITIONS_CONFIRMATION':'head_coalitions.py',
        'SECOND_HOP':'second_hop.py','RESULTS':'summarize_results.py'}
    for name,source in files.items():
        budget.check();report=read(name+'.json')
        assert report['source_sha256']==s.sha(s.HERE/source),name
        assert report['checkpoint_sha256']==checkpoint,name
        if 'artifact_sha256' in report:assert report['artifact_sha256']==s.sha(s.ART/(name+'.npz')),name
    assert read('INTERFACE_CHECKS.json')['status']=='passed'
    for name in ['PATH_TRACE','PATH_TRACE_ALL_HEADS','MATCHED_PATH_TRACE']:
        assert max(read(name+'.json')['checks'].values())==0.
    for name in ['COALITIONS_DISCOVERY','COALITIONS_CONFIRMATION']:
        report=read(name+'.json');assert max(report['checks'].values())==0.
        for mode in ['keys','values','keys_and_values']:
            assert report['results'][mode]['shapley_efficiency_max_error']<1e-9
    assert read('MLP_CONFIRMATION.json')['summaries']['all']['selected_minus_random_16']['group_bootstrap_95_percent'][0]>0
    assert read('PRIOR_EVIDENCE_INTEGRITY.json')['status']=='passed'
    assert len(read('PRIOR_EVIDENCE_INTEGRITY.json')['prior_manifests'])==6
    assert s.sha(s.HERE.parents[1]/'paper/build/main.pdf')==read('PRIOR_EVIDENCE_INTEGRITY.json')['manuscript_pdf_sha256']
    baseline=json.loads((s.FULL/'RESOURCE_USE.json').read_text());ledger=read('RESOURCE_USE.json')
    assert ledger[:len(baseline)]==baseline
    assert all(record['status']=='completed' and record['exact_queries']==0 for record in ledger[len(baseline):])
    assert set(read('RESOURCE_RESERVATIONS.json'))=={budget.name}
    manifest_files={}
    for path in sorted(s.HERE.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.name in ['RESOURCE_LOCK','EVIDENCE_MANIFEST.json']:continue
        budget.check();assert not path.is_symlink()
        if path.suffix=='.json':json.loads(path.read_text())
        if path.suffix=='.py':ast.parse(path.read_text(),filename=str(path))
        if path.suffix=='.npz':
            with np.load(path) as data:
                for key in data.files:
                    value=data[key]
                    if np.issubdtype(value.dtype,np.number):assert np.isfinite(value).all(),(str(path),key)
        assert path.stat().st_size<100_000_000
        manifest_files[str(path.relative_to(s.HERE))]=dict(bytes=path.stat().st_size,sha256=s.sha(path))
    sizes=storage();assert sizes['total_bytes']<read('resource_agreement.json')['storage_bytes']
    return manifest_files,sizes


def main():
    assert not (s.HERE/'EVIDENCE_MANIFEST.json').exists()
    budget=Budget('circuit_evidence_freeze',120);status='failed_or_partial'
    try:files,sizes=audit(budget);status='completed'
    finally:record=budget.finish(status)
    ledger=read('RESOURCE_USE.json');agreement=read('resource_agreement.json')
    total={k:sum(v[k] for v in ledger) for k in ['wall_seconds','gpu_seconds_charged','exact_queries']}
    remaining={k:agreement[cap]-total[k] for k,cap in [('wall_seconds','wall_seconds'),('gpu_seconds_charged','gpu_seconds'),('exact_queries','exact_queries')]}
    assert min(remaining.values())>=0 and not read('RESOURCE_RESERVATIONS.json')
    baseline=json.loads((s.FULL/'FINAL_RESOURCE_AUDIT.json').read_text())['cumulative_use']
    increment={k:total[k]-baseline[k] for k in total}
    resource=dict(status='passed',cumulative_use=total,this_continuation=increment,remaining=remaining,storage=sizes,source_sha256=s.sha(Path(__file__)))
    s.write('FINAL_RESOURCE_AUDIT.json',resource)
    for name in ['RESOURCE_USE.json','RESOURCE_RESERVATIONS.json','FINAL_RESOURCE_AUDIT.json']:
        path=s.HERE/name;files[name]=dict(bytes=path.stat().st_size,sha256=s.sha(path))
    s.write('EVIDENCE_MANIFEST.json',dict(date='2026-09-17',
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=s.HERE,text=True).strip(),
        checkpoint_sha256=s.sha(s.CHECKPOINT),protocol_amendment_sha256=s.sha(s.HERE/'PROTOCOL_AMENDMENT.json'),
        circuit_analysis_completed=True,algebraic_algorithm_identified=False,
        prior_integrity=read('PRIOR_EVIDENCE_INTEGRITY.json'),resource_use=total,resource_remaining=remaining,
        files=files))
    print(json.dumps(dict(status='passed',files=len(files),bytes=sum(v['bytes'] for v in files.values()),
        this_continuation=increment,remaining=remaining,storage_bytes=sizes['total_bytes'])),flush=True)


if __name__=='__main__':main()
