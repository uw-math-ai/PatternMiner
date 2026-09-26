"""Recompute the C018 coefficient-eight example with both exact engines."""
import json
from exact_kl import P
from independent_kl import polynomial, length
from descent_controls import cycles, naked, components

def main():
    illustration=polynomial((0,1,2,3),(2,3,0,1))
    assert illustration==P((0,1,2,3),(2,3,0,1))==(1,1)
    x=(0,4,3,2,1,8,7,6,5,9)
    w=(4,6,7,0,8,9,1,2,3,5)
    a=P(x,w)
    b=polynomial(x,w)
    assert a==b==(1,10,29,29,8)
    assert length(w)-length(x)==10 and cycles(x,w)==2
    assert not naked(x,w) and components(x,w)==1
    print(json.dumps(dict(status='PASS',x=x,w=w,P=a,gap=10,relative_cycles=2,
        support_components=1,genus=0,independent_engines_agree=True,
        introductory_example=dict(x=(0,1,2,3),w=(2,3,0,1),P=illustration),
        scope='Refutes the generic even-facet top-coefficient bound of four, not the source-sensitive corrected-gain conjecture.'),indent=2))

if __name__=='__main__':main()
