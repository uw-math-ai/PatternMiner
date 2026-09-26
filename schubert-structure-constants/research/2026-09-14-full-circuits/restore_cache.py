"""Restore compact data from retained gzip shards, without exact queries."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def restore(cache,verify_only=False):
    data=json.loads((HERE/'DATA.json').read_text());neighbors=json.loads((HERE/'NEIGHBOR_DATA.json').read_text())
    assert data['status']==neighbors['status']=='completed'
    jobs=[(name+'.bin',spec['archives'],spec['sha256']) for name,spec in data['populations'].items()]
    jobs.append(('neighbors_train.bin',neighbors['archives'],neighbors['data_sha256']))
    jobs.append(('neighbors_Q.npy',[dict(name='neighbors_Q.npy.gz',sha256=neighbors['root_archive_sha256'])],neighbors['root_sha256']))
    results={}
    for name,archives,expected in jobs:
        target=cache/'data'/name
        if target.exists():assert sha(target)==expected,str(target)
        temporary=target.with_suffix(target.suffix+'.restoring')
        output=None
        if not verify_only and not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True);output=temporary.open('xb')
        digest=hashlib.sha256();size=0
        try:
            for spec in archives:
                source=HERE/'artifacts'/spec['name'];assert sha(source)==spec['sha256']
                local=hashlib.sha256()
                with gzip.open(source,'rb') as stream:
                    for block in iter(lambda:stream.read(1<<20),b''):
                        local.update(block);digest.update(block);size+=len(block)
                        if output is not None:output.write(block)
                if 'uncompressed_sha256' in spec:assert local.hexdigest()==spec['uncompressed_sha256']
            assert digest.hexdigest()==expected,name
        finally:
            if output is not None:output.close()
        if output is not None:
            assert sha(temporary)==expected;temporary.replace(target)
        results[name]=dict(bytes=size,sha256=expected,cache_present=target.exists())
    return dict(status='passed',verification_only=verify_only,files=results)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cache',type=Path,default=HERE/'cache');parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    print(json.dumps(restore(args.cache,args.verify_only),indent=2))
