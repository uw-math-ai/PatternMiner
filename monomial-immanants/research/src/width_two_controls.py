#!/usr/bin/env python3
"""Independent T053 audit. Standard library only; no originating imports."""
from functools import lru_cache
from itertools import product, permutations, combinations
from math import comb, factorial
from collections import Counter
import json
from pathlib import Path

@lru_cache(None)
def parts(n, cap=None):
    if n == 0: return ((),)
    if cap is None: cap=n
    return tuple((a,)+q for a in range(min(n,cap),0,-1) for q in parts(n-a,a))

@lru_cache(None)
def kostka(shape, content):
    content=tuple(v for v in content if v)
    if not content: return int(not shape)
    if sum(shape)!=sum(content): return 0
    k=content[-1]
    # Largest letter occupies a horizontal strip: interlacing predecessor.
    result=0
    ranges=[range(shape[i+1] if i+1<len(shape) else 0, v+1) for i,v in enumerate(shape)]
    for row in product(*ranges):
        if sum(row)==sum(shape)-k:
            result += kostka(tuple(v for v in row if v),content[:-1])
    return result

def choose(n,k):
    if k<0: return 0
    if n>=0: return comb(n,k) if k<=n else 0
    return (-1)**k*comb(k-n-1,k)

def series(r,o,p,q):
    if p<0 or q<0: return 0
    k=r-o-1
    return sum(comb(o,i)*comb(o-i,j)*choose(k,p-i)*choose(k,q-j) for i in range(min(o,p)+1) for j in range(min(o-i,q)+1))

def coeff(shape,r,o):
    tail=shape[1:]; b=tail.count(2); c=tail.count(1); p=b+c
    return series(r,o,p,b)-series(r,o,p+1,b-1)

@lru_cache(None)
def sytdesc(shape):
    if not shape: return ((-1,0),)
    n=sum(shape); out=[]
    for i,v in enumerate(shape):
        if i+1<len(shape) and shape[i+1]==v: continue
        smaller=list(shape); smaller[i]-=1; smaller=tuple(x for x in smaller if x)
        for row,bits in sytdesc(smaller):
            out.append((i,bits | ((1<<(n-2)) if n>=2 and i>row else 0)))
    return tuple(out)

def polyadd(a,b,scale=1):
    c=Counter(a)
    for ij,v in b.items(): c[ij]+=scale*v
    return {ij:v for ij,v in c.items() if v}
def polymul(a,b):
    c=Counter()
    for (i,j),v in a.items():
        for (k,l),w in b.items():c[i+k,j+l]+=v*w
    return {ij:v for ij,v in c.items() if v}
def power(a,n):
    out={(0,0):1}
    for _ in range(n):out=polymul(out,a)
    return out

def main():
    counts=Counter()
    for N in range(2,13):
        for theta in parts(N):
            if theta[0]<2 or (len(theta)>1 and theta[1]>2):continue
            for alpha in parts(N):
                if len(alpha)>7:continue
                assert kostka(theta,alpha)==coeff(theta,len(alpha),alpha.count(1)),(theta,alpha)
                counts['content_series_checks']+=1
                if theta[1:].count(2)>=2:counts['multiple_tail_twos_contents']+=1
                if alpha.count(1)==N:counts['all_ones_contents']+=1
    U={(0,0):1,(1,0):1,(0,1):1,(1,1):1};V={(0,0):1,(1,0):1,(0,1):1};W={(1,1):1}
    for s in range(1,8):
        hs={(i,s-i):1 for i in range(s+1)}
        hm={(i,s-1-i):1 for i in range(s)}
        hp={(i,s+1-i):1 for i in range(s+2)}
        for a,b in product(range(2),repeat=2):
            actual={}; e=a+b; ell=s-1+e
            for mask in range(1<<s):
                cov=sum(bool(mask&(3<<i)) for i in range(s-1))+a*bool(mask&1)+b*bool(mask&(1<<(s-1)))
                term=polymul(power(U,s-mask.bit_count()+cov),power(V,ell-cov))
                actual=polyadd(actual,term,(-1)**mask.bit_count())
            expected=polymul(power(U,s-1),polyadd(hs,polymul(W,hm))) if e==0 else polymul(power(U,s),hs if e==1 else polyadd(hs,hp))
            assert actual==expected,(s,a,b)
            counts['exact_path_polynomial_identities']+=1
    for n in range(1,6):
        perms=list(permutations(range(n)))
        fibers={}
        for w in perms:
            h=tuple(max(w[:j+1]) for j in range(n));fibers.setdefault(h,[]).append(w)
        hs=sorted(fibers,key=lambda h:sum(h),reverse=True)
        # Every partition pair in this explicit finite domain; all rows nonempty.
        for mu in product(range(1,5),repeat=n):
            if any(mu[i]<mu[i+1] for i in range(n-1)):continue
            for nu in product(*(range(v) for v in mu)):
                if any(nu[i]<nu[i+1] for i in range(n-1)):continue
                N=sum(mu)-sum(nu)
                if N<2 or N>11:continue
                counts['skew_shapes']+=1
                bs=[mu[i]+n-1-i for i in range(n)];cs=[nu[i]+n-1-i for i in range(n)]
                contents={};ones={};rs={}
                for h,ws in fibers.items():
                    allalpha=[tuple(bs[w[j]]-cs[j] for j in range(n)) for w in ws]
                    feasible=[all(v>=0 for v in alpha) for alpha in allalpha]
                    assert len(set(feasible))==1
                    if not feasible[0]:continue
                    alpha=allalpha[0];contents[h]=alpha
                    ones[h]={j for j in range(n) if (j==0 or h[j]>h[j-1]) and bs[h[j]]-cs[j]==1}
                    rs[h]=sum(v>0 for v in alpha)
                    assert all((sum(v>0 for v in al),al.count(1))==(rs[h],len(ones[h])) for al in allalpha)
                    counts['permutation_fiber_checks']+=len(ws)
                    corners=[j for j in range(n-1) if h[j]<h[j+1] and bs[h[j]+1]>=cs[j]]
                    losses=[]; z=0; inert=False; corner_data=[]
                    for j in corners:
                        hh=tuple(v+(i==j) for i,v in enumerate(h))
                        # Compute the record statistics directly, before subsequent h is visited.
                        oj={i for i in range(n) if (i==0 or hh[i]>hh[i-1]) and bs[hh[i]]-cs[i]==1}
                        rj=n-sum((i==0 or hh[i]>hh[i-1]) and bs[hh[i]]==cs[i] for i in range(n))
                        u=rs[h]-rj
                        corner_data.append((u, ones[h]-oj, oj-ones[h]))
                        assert u in (0,1)
                        if not u:
                            assert ones[h]<=oj and len(oj-ones[h])<=1
                            if oj==ones[h]:inert=True
                            else:z+=1
                        else:
                            assert oj<=ones[h] and len(ones[h]-oj)<=2
                            losses.append((j,ones[h]-oj))
                    for mask in range(1<<len(corners)):
                        chosen={corners[k] for k in range(len(corners)) if mask>>k&1}
                        hh=tuple(v+(i in chosen) for i,v in enumerate(h))
                        actual_ones={i for i in range(n) if (i==0 or hh[i]>hh[i-1]) and bs[hh[i]]-cs[i]==1}
                        actual_r=n-sum((i==0 or hh[i]>hh[i-1]) and bs[hh[i]]==cs[i] for i in range(n))
                        removed=set();added=set();u_total=0
                        for k,(u,lost,new) in enumerate(corner_data):
                            if mask>>k&1:removed|=lost;added|=new;u_total+=u
                        assert actual_r==rs[h]-u_total
                        assert actual_ones==(ones[h]-removed)|added
                        counts['exact_loss_set_cube_vertices']+=1
                    if not inert:
                        edges=[(i,j) for i in range(len(losses)) for j in range(i+1,len(losses)) if losses[i][1]&losses[j][1]]
                        assert all(losses[j][0]==losses[i][0]+1 and len(losses[i][1]&losses[j][1])==1 for i,j in edges)
                        comps=[];unseen=set(range(len(losses)))
                        while unseen:
                            cc={unseen.pop()}
                            while True:
                                more={j for i,j in edges if i in cc}|{i for i,j in edges if j in cc}
                                if more<=cc:break
                                cc|=more;unseen-=more
                            comps.append(cc)
                        c0=0
                        for cc in comps:
                            union=set().union(*(losses[i][1] for i in cc));e=len(union)-(len(cc)-1)
                            assert e in (0,1,2)
                            c0+=e==0
                        B=rs[h]-len(ones[h])
                        if B:
                            assert B>=z+c0+1,(mu,nu,h,B,z,c0)
                            counts['spare_big_entry_checks']+=1
                    counts['legal_cubes']+=1
                for theta in parts(N):
                    width=theta[0]>=2 and (len(theta)==1 or theta[1]<=2)
                    if not width:continue
                    F={h:kostka(theta,tuple(v for v in contents[h] if v)) if h in contents else 0 for h in hs}
                    coefficients={}
                    for h in hs:
                        ah=F[h]-sum(v for g,v in coefficients.items() if all(g[i]>=h[i] for i in range(n)))
                        assert ah>=0,(mu,nu,theta,h,ah)
                        coefficients[h]=ah
                        corners=[i for i in range(n-1) if h[i]<h[i+1]]
                        alt=0
                        for mask in range(1<<len(corners)):
                            chosen={corners[k] for k in range(len(corners)) if mask>>k&1}
                            hh=tuple(v+(i in chosen) for i,v in enumerate(h))
                            alt+=(-1)**mask.bit_count()*F[hh]
                        assert alt==ah
                        counts['full_poset_coefficients']+=1
                        if theta[1:].count(2)>=2:counts['multiple_tail_twos_coefficients']+=1
                    if all(mu[i]-nu[i]==1 for i in range(n)):
                        A=sum(1<<i for i in range(n-1) if cs[i]-cs[i+1]==1)
                        exact=sum(bits&A==A for row,bits in sytdesc(theta))
                        assert coefficients[tuple(range(n))]==exact
                        counts['zero_based_descent_checks']+=1
    output={'verdict':'PASS','counts':dict(counts),'scope':'Symbolic audit is separately recorded in review; finite controls are not proof.'}
    print(json.dumps(output,indent=2))
if __name__=='__main__':main()
