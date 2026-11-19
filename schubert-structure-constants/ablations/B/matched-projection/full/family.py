"""A uniform family with retained-dimension depth n-2 for every n >= 4.

No oracle, stored examples, fitted parameters, or search is used here.
"""

def example(n):
    if not isinstance(n,int) or n < 4:
        return None
    m=n//2
    u=list(range(m,0,-1))+list(range(n,m,-1))
    t=list(range(2,m+1))+[1,m+2,m+1]+list(range(m+3,n+1))
    w=[n+1-x for x in t]
    return u,u.copy(),w
