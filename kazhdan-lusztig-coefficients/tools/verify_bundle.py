"""Read-only publication checks; no mathematical replays or network access."""
from pathlib import Path
import ast
import hashlib
import importlib
import json
import re
import sys
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]

def digest(data):return hashlib.sha256(data).hexdigest()

def inventory():
    return {p.relative_to(ROOT).as_posix():p for p in ROOT.rglob('*') if p.is_file()
            and not p.is_symlink() and '__pycache__' not in p.parts
            and 'replay-output' not in p.parts and p.suffix!='.pyc'
            and p.name not in {'MANIFEST.json','.DS_Store'}}

def stable(value):
    if isinstance(value,dict):return {k:stable(v) for k,v in value.items() if k not in {'seconds','elapsed_seconds'}}
    if isinstance(value,list):return [stable(v) for v in value]
    return value

def markdown_links(path):
    text=path.read_text()
    tick=chr(96)
    text=re.sub(tick*3+r'.*?'+tick*3,'',text,flags=re.S)
    text=re.sub(tick+r'+[^'+tick+r']*'+tick+r'+','',text)
    count=0
    for url in re.findall(r'\[[^\]\n]+\]\(([^)\n]+)\)',text):
        if re.match(r'https?://|mailto:',url):continue
        target,_,anchor=url.partition('#')
        other=(path.parent/unquote(target)).resolve() if target else path
        assert other.exists(),(str(path.relative_to(ROOT)),url)
        assert other.is_relative_to(ROOT),('link leaves publication package',url)
        if anchor and other.suffix=='.md':
            headings=re.findall(r'^#+\s+(.+)$',other.read_text(),re.M)
            ids={re.sub(r'[^\w\- ]','',h.lower()).replace(' ','-') for h in headings}
            assert anchor in ids,('missing heading',url)
        count+=1
    return count

def evidence():
    run=json.loads((ROOT/'research/artifacts/publication_replay.json').read_text())
    checks=run['checks']
    assert run['status']=='PASS'
    if (ROOT/'research/e046_config.json').exists():
        r=checks['e046']['result']
        cfg=json.loads((ROOT/'research/e046_config.json').read_text())
        assert len(r['training_rows'])==cfg['training_rows']==33
        assert len(r['holdout_rows'])==2
        assert r['training_label_histogram']=={'0':17,'1':16}
        assert [x['label'] for x in r['holdout_rows']]==cfg['holdout_exact_labels']
        assert r['holdout_predictions']==cfg['holdout_predictions']
        for row in r['training_rows']+r['holdout_rows']:
            assert set(row)-{'source','descent','label'}==set(cfg['features'])
        assert [x[1:] for x in r['best_rules']]==cfg['expected_best_rules']
        assert all(x[0]==3 for x in r['best_rules'])
        assert checks['support']['result']['records'][-1]['intervals']==4017
        assert checks['coefficient-eight']['result']['P']==[1,10,29,29,8]
        assert checks['coefficient-eight']['result']['introductory_example']['P']==[1,1]
        for name in ['heredity','descent']:
            old=json.loads((ROOT/f'research/artifacts/historical/{name}.json').read_text())
            assert stable(checks[name]['result'])==stable(old),name
    else:
        old=json.loads((ROOT/'research/artifacts/historical/width_two.json').read_text())
        assert stable(checks['width-two']['result'])==stable(old)
        assert checks['width-two']['result']['counts']['full_poset_coefficients']==776891
        rows=checks['rectangles']['result']['witnesses']
        assert [r['positive_terms'] for r in rows]==[38,51,42,33,15]
        assert [r['total_multiplicity'] for r in rows]==[50,75,50,40,15]
        assert all(r['permutations_checked']==720 for r in rows)
    return len(checks)

def main():
    if not __debug__:raise SystemExit('Assertions must be enabled; do not run with -O.')
    sys.dont_write_bytecode=True
    actual=inventory()
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    expected={entry['path']:entry for entry in manifest['files']}
    assert actual.keys()==expected.keys(),{'missing':sorted(expected.keys()-actual.keys()),'extra':sorted(actual.keys()-expected.keys())}
    for name,path in actual.items():
        raw=path.read_bytes()
        assert len(raw)==expected[name]['bytes'] and digest(raw)==expected[name]['sha256'],name
        assert not re.search(rb'/(?:home|mnt|Users)/|BEGIN [A-Z ]*PRIVATE KEY|gh[pousr]_[A-Za-z0-9]{20,}',raw),name
        if path.suffix=='.py':ast.parse(raw,filename=name)
        if path.suffix=='.json':json.loads(raw)
    provenance=json.loads((ROOT/'research/SOURCE_MAP.json').read_text())
    index=provenance['campaign_index']
    ids=[row['id'] for row in index['experiments']]
    assert len(ids)==len(set(ids))==index['experiment_ids']==62
    assert sum(len(row['reports']) for row in index['experiments'])==index['report_files']==63
    assert set(ids)=={f'E{i:03}' for i in range(1,63)}
    definitions=0
    for entry in provenance['files']:
        path=ROOT/entry['destination']
        assert digest(path.read_bytes())==entry['published_sha256'],entry['destination']
        if entry['mode']=='verbatim':assert entry['source_sha256']==entry['published_sha256']
        if entry['mode']=='scoped-adaptation':
            text=path.read_text()
            for old,new in reversed(entry['details']['replacements']):
                if not new:break
                assert new in text
                text=text.replace(new,old)
            else:assert digest(text.encode())==entry['source_sha256']
        if entry['mode']=='definition-extraction':
            nodes={n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
            for name,wanted in entry['details']['definitions'].items():
                assert digest(ast.dump(nodes[name],include_attributes=False).encode())==wanted
                definitions+=1
    src=ROOT/'research/src'
    sys.path.insert(0,str(src))
    modules={p.stem for p in src.glob('*.py')}
    for name in sorted(modules):
        tree=ast.parse((src/(name+'.py')).read_text())
        for node in ast.walk(tree):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) else []
            for module in names:
                assert module.split('.')[0] in modules|sys.stdlib_module_names,('unbundled import',name,module)
        importlib.import_module(name)
    links=sum(markdown_links(p) for p in actual.values() if p.suffix=='.md')
    checks=evidence()
    print(json.dumps(dict(status='PASS',files=len(actual),source_entries=len(provenance['files']),
          unchanged_definition_fingerprints=definitions,imported_modules=len(modules),local_links=links,
          saved_check_groups=checks,scope='Integrity, provenance, syntax, imports, links and saved-result checks; no mathematics rerun.'),indent=2))

if __name__=='__main__':main()
