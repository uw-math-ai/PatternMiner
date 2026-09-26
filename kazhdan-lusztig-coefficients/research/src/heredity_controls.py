#!/usr/bin/env python3
"""Independent standard-library finite controls; writes JSON to stdout only."""
import itertools, json, time
from pathlib import Path
from collections import Counter


def inv(p):
    return sum(p[i] > p[j] for i in range(len(p)) for j in range(i+1,len(p)))


def leq(a,b):
    return all(sum(v >= t for v in a[:i]) <= sum(v >= t for v in b[:i]) for i in range(1,len(a)+1) for t in range(len(a)))


def cycles(p):
    unseen=set(range(len(p))); out=0
    while unseen:
        out+=1; i=next(iter(unseen))
        while i in unseen:
            unseen.remove(i); i=p[i]
    return out


def relative(a,b):
    ai={v:i for i,v in enumerate(a)}
    return tuple(ai[v] for v in b)


def components(n,edges):
    blocks=[{i} for i in range(n)]
    for i,j in edges:
        bi=next(b for b in blocks if i in b); bj=next(b for b in blocks if j in b)
        if bi is not bj:
            bi.update(bj); blocks.remove(bj)
    return tuple(sorted(tuple(sorted(b)) for b in blocks))


def word_stats(n,word):
    p=list(range(n))
    for i,j in word: p[i],p[j]=p[j],p[i]
    k=len(components(n,word)); c=cycles(p)
    g2=len(word)-n-c+2*k
    assert g2 >= 0 and g2%2 == 0
    return g2//2


def flatten(p,block):
    vals=sorted(p[i] for i in block); ranks={v:i for i,v in enumerate(vals)}
    return tuple(ranks[p[i]] for i in block)


def naked(a,b):
    return [i for i in range(len(a)) if a[i]==b[i] and sum(v>=a[i] for v in a[:i])==sum(v>=a[i] for v in b[:i])]


def run():
    started=time.monotonic(); counts=Counter(); hist=Counter()
    for n,maxlen in [(1,0),(2,7),(3,7),(4,6)]:
        letters=list(itertools.combinations(range(n),2))
        for length in range(maxlen+1):
            for word in itertools.product(letters,repeat=length):
                g=word_stats(n,word); counts['words']+=1
                assert word_stats(n,word[::-1])==g
                if word: assert word_stats(n,word[:-1]) <= g
                for i in range(length+1):
                    for j in range(i,length+1):
                        assert word_stats(n,word[i:j]) <= g
                        counts['contiguous_subwords']+=1
    for n in range(1,6):
        ps=list(itertools.permutations(range(n))); lens={p:inv(p) for p in ps}
        comp={(a,b):leq(a,b) for a in ps for b in ps}
        up={a:[] for a in ps}
        for a in ps:
            for i,j in itertools.combinations(range(n),2):
                b=list(a); b[i],b[j]=b[j],b[i]; b=tuple(b)
                if lens[b]==lens[a]+1: up[a].append((b,(i,j)))
        stats={}
        for a in ps:
            for b in ps:
                if not comp[a,b]: continue
                counts['intervals']+=1
                # Compute support from ALL internal covers, independently of one chain.
                es=[e for z in ps if comp[a,z] and comp[z,b] for zz,e in up[z] if comp[zz,b]]
                blocks=components(n,es); c=cycles(relative(a,b)); d=lens[b]-lens[a]
                g2=d-n-c+2*len(blocks)
                assert g2>=0 and g2%2==0
                g=g2//2; stats[a,b]=(g,blocks,c,d); hist[f'S{n}:g{g}']+=1
                # One selected saturated chain reconstructs the independently computed full-cover support.
                z=a; chain=[]
                while z!=b:
                    zz,e=next((zz,e) for zz,e in up[z] if comp[zz,b]); chain.append(e); z=zz
                assert components(n,chain)==blocks
                assert word_stats(n,chain)==g
                total=0
                for block in blocks:
                    aa=flatten(a,block); bb=flatten(b,block)
                    assert leq(aa,bb)
                    dg=inv(bb)-inv(aa); cg=cycles(relative(aa,bb)); gg2=dg-len(block)-cg+2
                    assert gg2>=0 and gg2%2==0
                    total+=gg2//2
                    if len(block)>1:
                        assert not naked(aa,bb)
                        counts['canonical_nonsingleton_factors']+=1
                assert total==g
        for (a,b),(g,blocks,c,d) in stats.items():
            facets=[(aa,b) for aa,e in up[a] if comp[aa,b]]
            facets += [(a,bb) for bb in ps if comp[a,bb] and any(z==b for z,e in up[bb])]
            for ab in facets:
                gg,bs,cc,dd=stats[ab]
                assert dd==d-1 and gg<=g
                counts['endpoint_facet_heredity']+=1
                if g==0 and len(blocks)==1:
                    assert (len(bs),cc) in [(1,c-1),(2,c+1)]
                    counts['genus_zero_connected_facets']+=1
            if n<=4:
                for (aa,bb),(gg,_,_,_) in stats.items():
                    if comp[a,aa] and comp[bb,b]:
                        assert gg<=g
                        counts['all_nested_intervals_S_le_4']+=1
    # Full-support proper-subinterval classification is an exact integer consequence.
    frontier=[]
    for d in range(14):
        for c in range(1,13):
            if d-12-c+2==0:
                frontier.append([d,c])
    assert frontier==[[11,1],[12,2],[13,3]]
    return {'status':'PASS','counts':dict(counts),'genus_histogram':dict(hist),'twelve_position_genus_zero_pairs_d_le_13':frontier,'elapsed_seconds':round(time.monotonic()-started,3),'scope':'Independent finite controls; symbolic audit supplies universal reasoning. No KL coefficient bound tested or implied.'}

if __name__=='__main__':
    out=run(); print(json.dumps(out,indent=2))
