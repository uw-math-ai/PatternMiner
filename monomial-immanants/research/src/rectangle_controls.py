"""Independent full-poset replay of the retained six-row rectangle witnesses."""
import json
from itertools import permutations
from pathlib import Path
from width_two_controls import kostka, coeff

def main():
    # D=(2,1), phi^(2)(identity)=0, phi^(2)(transposition)=2.
    illustration=[]
    for theta in [(3,),(2,1),(1,1,1)]:
        value=sum((0 if w==(0,1) else 2)*kostka(theta,tuple((2,1)[w[j]]+j-w[j] for j in range(2)))
                  for w in permutations(range(2)))
        illustration.append(dict(shape=theta,coefficient=value))
    assert [r['coefficient'] for r in illustration]==[2,0,0]
    frozen=json.loads((Path(__file__).resolve().parents[1]/'artifacts/historical/rectangles.json').read_text())
    n=6
    perms=list(permutations(range(n)))
    fibers={}
    for w in perms:
        h=tuple(max(w[:j+1]) for j in range(n))
        fibers.setdefault(h,[]).append(w)
    hs=sorted(fibers,key=lambda h:(sum(h),h),reverse=True)
    assert len(hs)==132
    rows=[]
    for witness in frozen['witnesses']:
        theta=tuple(witness['shape'])
        weights={}
        for h,ws in fibers.items():
            values=set()
            for w in ws:
                alpha=tuple(2+j-w[j] for j in range(n))
                value=0 if min(alpha)<0 else kostka(theta,alpha)
                if min(alpha)>=0:
                    assert value==coeff(theta,sum(v>0 for v in alpha),alpha.count(1))
                values.add(value)
            assert len(values)==1
            weights[h]=values.pop()
        coefficients={}
        for h in hs:
            value=weights[h]-sum(a for g,a in coefficients.items() if all(x>=y for x,y in zip(g,h)))
            assert value>=0
            coefficients[h]=value
        actual={h:a for h,a in coefficients.items() if a}
        expected={tuple(term['h']):term['coefficient'] for term in witness['terms']}
        assert actual==expected
        for w in perms:
            alpha=tuple(2+j-w[j] for j in range(n))
            direct=0 if min(alpha)<0 else kostka(theta,alpha)
            assert direct==sum(a for h,a in actual.items() if all(w[j]<=h[j] for j in range(n)))
        rows.append(dict(shape=theta,positive_terms=len(actual),total_multiplicity=sum(actual.values()),permutations_checked=len(perms)))
    assert [r['positive_terms'] for r in rows]==[38,51,42,33,15]
    assert [r['total_multiplicity'] for r in rows]==[50,75,50,40,15]
    print(json.dumps(dict(status='PASS',hessenberg_functions=132,witnesses=rows,
        introductory_example=dict(skew_shape=(2,1),immanant_index=(2,),schur_coefficients=illustration),
        scope='Fresh full-poset inversion and all-permutation equality checks of the five frozen witnesses.'),indent=2))

if __name__=='__main__':main()
