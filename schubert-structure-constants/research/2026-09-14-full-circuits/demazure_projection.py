"""Compute the Bruhat projection by right downward Demazure operators."""


def inversion_length(p):
    return sum(a>b for i,a in enumerate(p) for b in p[i+1:])


def block_projection(h,w,left,right):
    """Inclusive value interval [left,right], in one-based permutation notation."""
    n=len(w);assert len(h)==n and 1<=left<right<=n
    positions=[i for i,value in enumerate(w) if left<=value<=right]
    z=list(w)
    for i,value in zip(positions,range(left,right+1)):z[i]=value
    a=list(h)
    # Sorting z by adjacent right descents lists a reduced word for z^{-1}.
    for end in range(n-1,0,-1):
        for i in range(end):
            if z[i]>z[i+1]:
                z[i],z[i+1]=z[i+1],z[i]
                if a[i]>a[i+1]:a[i],a[i+1]=a[i+1],a[i]
    assert z==list(range(1,n+1))
    assert all(value==i+1 for i,value in enumerate(a) if not left<=i+1<=right)
    local=[value-left+1 for value in a[left-1:right]]
    assert sorted(local)==list(range(1,right-left+2))
    x=[w[i]-left+1 for i in positions]
    inverse=[local.index(value) for value in range(1,len(local)+1)]
    descents=[inverse[i]>inverse[i+1] for i in range(len(local)-1)]
    return local,inversion_length(local),descents,inversion_length(x)


def all_levi_restrictions(u,v,w):
    """Largest necessary-length violation over every Levi subset containing i."""
    n=len(w);blocks={}
    for left in range(1,n):
        for right in range(left+1,n+1):
            _,a,_,x=block_projection(u,w,left,right)
            _,b,desc,y=block_projection(v,w,left,right);assert x==y
            blocks[left,right]=(a+b-x,desc)
    # A Levi subset is a set of disjoint nontrivial value intervals.
    prefix=[0]*(n+1);suffix=[0]*(n+2)
    for right in range(1,n+1):
        prefix[right]=max([prefix[right-1]]+[prefix[left-1]+blocks[left,right][0] for left in range(1,right)])
    for left in range(n,0,-1):
        suffix[left]=max([suffix[left+1]]+[blocks[left,right][0]+suffix[right+1] for right in range(left+1,n+1)])
    violations=[];witnesses=[]
    for i in range(1,n):
        value,left,right=max((prefix[left-1]+blocks[left,right][0]-int(blocks[left,right][1][i-left])+suffix[right+1],left,right)
            for left in range(1,i+1) for right in range(i+1,n+1))
        violations.append(value);witnesses.append((left,right))
    return violations,blocks
