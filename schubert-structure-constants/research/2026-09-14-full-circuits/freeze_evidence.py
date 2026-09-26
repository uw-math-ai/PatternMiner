"""Audit completed results and freeze the staged evidence with its resource ledger."""
import ast
import argparse
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import session as s
from full_resources import Budget

ROOT=s.HERE.parents[2]
NAMES=['2026-09-13-autoresearch-circuits','2026-09-13-witness-accuracy',
       '2026-09-13-circuit-readiness','2026-09-14-reliable-circuits',
       '2026-09-14-richardson-circuits',s.HERE.name]


def read(name):return json.loads((s.HERE/name).read_text())


def storage():
    seen=set();sizes={}
    for name in NAMES:
        total=0
        for parent,dirs,files in os.walk(s.HERE.parent/name,followlinks=True):
            dirs[:]=[d for d in dirs if d!='__pycache__']
            for filename in files:
                stat=(Path(parent)/filename).stat();key=(stat.st_dev,stat.st_ino)
                if key not in seen:seen.add(key);total+=stat.st_size
        sizes[name]=total
    return dict(scope='Six experiment directories, including symlinked caches; count each physical inode once.',
        bytes_by_directory=sizes,total_bytes=sum(sizes.values()),
        root_free_bytes=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize,
        data_volume_free_bytes=os.statvfs('/mnt/data').f_bavail*os.statvfs('/mnt/data').f_frsize)


def audit(budget):
    assert set(read('RESOURCE_RESERVATIONS.json'))=={budget.name}
    selected=read('SECOND_MODEL_SELECTION.json');assert selected['status']=='selected'
    digest=s.sha(s.HERE/selected['checkpoint']);assert digest==selected['checkpoint_sha256']
    native=read('SECOND_NATIVE_VALIDATION.json')
    assert native['status']=='completed' and native['checkpoint_sha256']==digest
    assert len(native['initial_validation'])==27 and native['best_criterion'][0]>=.99
    assert selected['interpolation_sha256']==s.sha(s.HERE/'SECOND_INTERPOLATION.json')
    assert read('SECOND_INTERPOLATION.json')['status']=='completed'
    assessment=read('SECOND_ASSESSMENT.json')
    assert assessment['status']=='completed' and assessment['qualification']['checkpoint_sha256']==digest
    formal=read('SECOND_FORMAL_STUDY_STATUS.json')
    if assessment['passes']:
        assert formal['status']=='completed' and formal['checkpoint_sha256']==digest
        for name in ['FORMAL_DISCOVERY.json','FORMAL_CONFIRMATION.json','FORMAL_MLP_PATH.json']:
            report=read(name);assert report['checkpoint_sha256']==digest
        assert read('FORMAL_MLP_PATH.json')['status']=='completed'
        assert not (s.HERE/'SECOND_ASSESSMENT_EXPOSURE.json').exists()
    else:
        assert (s.HERE/'SECOND_ASSESSMENT_EXPOSURE.json').exists()
        assert not (s.HERE/'FORMAL_DISCOVERY_OPENED.json').exists()
        assert not (s.HERE/'FORMAL_CONFIRMATION_OPENED.json').exists()
    for name in ['neighbor_refinement.json','NEIGHBOR_DATA.json','PREPARED.json','NEIGHBOR_PREPARED.json']:
        assert read(name)['status']=='completed'
    assert not read('FRESH_ASSESSMENT.json')['passes']
    assert (s.HERE/'FRESH_ASSESSMENT_EXPOSURE.json').exists()
    for name in ['SECOND_DATA.json','SECOND_NEIGHBOR_DATA.json','SECOND_NEIGHBOR_PREPARED.json',
                 'FINAL_DEVELOPMENT_REPLAY_STATUS.json','FINAL_COMPONENT_SCAN.json']:
        assert read(name)['status']=='completed'
    for name in ['FINAL_DEVELOPMENT_DECOMPOSITION.json','FINAL_SUPPORT_CONTROLS.json','FINAL_COMPONENT_SCAN.json']:
        assert read(name)['checkpoint_sha256']==digest
    assert read('DEVELOPMENT_HEAD_CONTROLS.json')['status']=='completed'
    assert read('DEVELOPMENT_HEAD_CONTROLS.json')['checkpoint_sha256']==digest
    for name in ['HEAD_PRUNING.json','TASK_HEAD_PRUNING.json','TASK_HEAD_PRUNING_NATIVE_VALIDATION.json','PRUNING_SUMMARY.json']:
        assert read(name)['status']=='completed' and read(name)['checkpoint_sha256']==digest
    assert read('FINAL_OUTCOME.json')['checkpoint_sha256']==digest
    integrity=read('PRIOR_EVIDENCE_INTEGRITY.json');assert integrity['status']=='passed'
    protected=[str((s.HERE.parent/name).relative_to(ROOT)) for name in NAMES[:-1]]
    protected.append(str((s.HERE.parents[1]/'paper').relative_to(ROOT)))
    assert not subprocess.check_output(['git','diff','HEAD','--',*protected],cwd=ROOT)
    assert s.sha(s.HERE.parents[1]/'paper/build/main.pdf')==integrity['manuscript_pdf_sha256']
    paths=subprocess.check_output(['git','ls-files','--cached','-z','--',str(s.HERE.relative_to(ROOT))],cwd=ROOT).decode().split('\0')
    files={}
    for filename in filter(None,paths):
        path=ROOT/filename;relative=str(path.relative_to(s.HERE))
        if relative=='EVIDENCE_MANIFEST.json':continue
        budget.check();assert path.is_file() and not path.is_symlink(),relative
        if path.suffix=='.json':json.loads(path.read_text())
        elif path.suffix=='.py':ast.parse(path.read_text(),filename=str(path))
        elif path.suffix=='.npz':
            with np.load(path) as archive:
                for key in archive.files:archive[key]
        size=path.stat().st_size;assert size<100_000_000,(relative,size)
        files[relative]=dict(bytes=size,sha256=s.sha(path))
    assert files
    sizes=storage();assert sizes['total_bytes']<=read('resource_agreement.json')['storage_bytes']
    return files,sizes,selected,assessment,formal,integrity


def main(budget_name):
    assert not (s.HERE/'EVIDENCE_MANIFEST.json').exists()
    assert subprocess.check_output(['git','ls-files','--cached','--',str(s.HERE.relative_to(ROOT))],cwd=ROOT)
    budget=Budget(budget_name,240);status='failed_or_partial'
    try:
        files,sizes,selected,assessment,formal,integrity=audit(budget);status='completed'
    finally:charge=budget.finish(status)
    ledger=read('RESOURCE_USE.json');agreement=read('resource_agreement.json')
    assert len({item['name'] for item in ledger})==len(ledger)
    totals={key:sum(item[key] for item in ledger) for key in ['wall_seconds','gpu_seconds_charged','exact_queries']}
    remaining={key:agreement[cap]-totals[key] for key,cap in [('wall_seconds','wall_seconds'),('gpu_seconds_charged','gpu_seconds'),('exact_queries','exact_queries')]}
    assert min(remaining.values())>=0 and not read('RESOURCE_RESERVATIONS.json')
    s.write('FINAL_RESOURCE_AUDIT.json',dict(status='passed',cumulative_use=totals,remaining=remaining,storage=sizes,source_sha256=s.sha(Path(__file__))))
    for name in ['RESOURCE_USE.json','RESOURCE_RESERVATIONS.json','FINAL_RESOURCE_AUDIT.json']:
        path=s.HERE/name;files[name]=dict(bytes=path.stat().st_size,sha256=s.sha(path))
    manifest=dict(date='2026-09-15',base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        independently_qualified=assessment['passes'],qualification_scope='Recorded finite assessment populations and stated metrics; not a universal correctness guarantee.',
        formal_study_status=formal['status'],confirmed_algebraic_algorithm=False,
        selected_checkpoint=selected['checkpoint'],selected_checkpoint_sha256=selected['checkpoint_sha256'],
        final_selection_sha256=s.sha(s.HERE/'SECOND_MODEL_SELECTION.json'),independent_assessment_sha256=s.sha(s.HERE/'SECOND_ASSESSMENT.json'),
        prior_integrity=integrity,resource_use=totals,resource_remaining=remaining,storage=sizes,files=files)
    s.write('EVIDENCE_MANIFEST.json',manifest)
    print(json.dumps(dict(files=len(files),bytes=sum(v['bytes'] for v in files.values()),independently_qualified=assessment['passes'],formal_study_status=formal['status'],remaining=remaining,storage_bytes=sizes['total_bytes'],audit_charge=charge)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--budget-name',default='full_evidence_freeze')
    main(parser.parse_args().budget_name)
