"""Verify all five frozen manifests and the manuscript before finalizing this continuation."""
import json
from pathlib import Path
import session as s
from full_resources import Budget


def main(budget):
    reports={}
    for name in ['2026-09-13-autoresearch-circuits','2026-09-13-witness-accuracy','2026-09-13-circuit-readiness','2026-09-14-reliable-circuits','2026-09-14-richardson-circuits']:
        parent=s.HERE.parent/name;path=parent/'EVIDENCE_MANIFEST.json';manifest=json.loads(path.read_text());count=0
        for filename,spec in manifest['files'].items():
            budget.check();file=parent/filename
            assert file.is_file() and file.stat().st_size==spec['bytes'],str(file)
            assert s.sha(file)==spec['sha256'],str(file);count+=1
        reports[name]=dict(files=count,manifest_sha256=s.sha(path))
        print(json.dumps(dict(directory=name,files=count,status='passed')),flush=True)
    pdf=s.HERE.parents[1]/'paper/build/main.pdf'
    assert s.sha(pdf)=='1bb1c40c018df10338145103be651ef63e0929d6ec505e86a90cdfe882272347'
    report=dict(status='passed',source_sha256=s.sha(Path(__file__)),prior_manifests=reports,manuscript_pdf_sha256=s.sha(pdf))
    s.write('PRIOR_EVIDENCE_INTEGRITY.json',report)


if __name__=='__main__':
    budget=Budget('prior_evidence_integrity',180);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
