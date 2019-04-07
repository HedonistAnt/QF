import sympy as sp
import numpy as np
import math
from numpy.linalg import solve



def Richardson(S,H,m):
    r=len(H)
    P=np.array(range(int(m)+1,int(m)+1+r-1))
    Ht = np.empty((0, r), dtype=np.double)
    for i in range(len(H)):
        K=[]
        for j in range(len(P)):
            K.append(H[i]**P[j])
        K.append(-1)
        Ht = np.append(Ht, [K], axis=0)
    print(Ht)
    C = np.linalg.inv(Ht) @ np.transpose(-np.array(S))

    return C[0]

def Richardson(S,H,m,l):
    d1 = H[0]**(m) - H[1]**(m)
    k1 = H[0]**(m+1) - H[1]**(m+1)
    d2 = H[1]**(m) - H[2]**(m)
    k2 = H[1]**(m+1) - H[2]**(m+1)
    return (-S[1]+S[2])/d2 - (d2*(-S[0]+S[1]) - S[1]*d1 + S[2]*d1)*k2/((k1-k2*d1)*d2)



def Eitken(S,l):
    return -np.log(abs((S[2]-S[1])/(S[1]-S[0])))/np.log(l)

def compute_n_sums (n,a,b,r):
    integsum=0;
    h=(b-a)/n
    for i in range(n):
        B = a + (i + 1) * h
        A = B - h
        p = 1 / t ** (3 / 7)
        M = pmoment(p, A, B, r)
        N = np.linspace(A, B, r)
        integsum += newtonqf(M, N)
    return integsum;


def compositeqf(a,b,r,eps):
    p = 1/t**(-3/7)
    M = pmoment(p, a, b, r)
    N =  np.linspace(a,b,r,dtype=np.double)
    integsum_p = newtonqf(M, N)
    n=4
    P=148.825
    l=2
    niter=0
    m=1
    S=np.array([],dtype=np.double)
    H=np.array([],dtype=np.double)
    Rh=np.array([],dtype=np.double)
    Ms=np.array([],dtype=np.double)
    C=100
    while (niter<100):
        niter += 1
        print("N_iter=", niter)
        if (len(H)!=0):
            n*=l
        h=(b-a)/n
        print("h=", h, " nsteps=", n)
        integsum=compute_n_sums(n,a,b,r)
        print("S=", integsum)
        S=np.append(S,integsum)
        H=np.append(H,h)
        if (r % 2 == 0):
            Rn = (integsum - integsum_p) / (l ** (r - 1) - 1)
        else:
            Rn = (integsum - integsum_p) / (l ** (r) - 1)
        Rh=np.append(Rh,Rn)
        print("Rn=", Rn)
        if len(S)==3:
            mprev = m
            m=Eitken(S,l)
            Ms = np.append(Ms,m)
            ho=hopt(H[1],Rh[1],m,eps)
            if len(Ms) > 2:
               Ms=np.delete(Ms,0)
            Cprev = C
            C=Richardson(S,H,m,l)
            if abs(C-Cprev)<ho and (niter>=3):
                break;
            print("C=", C)
            n = int((b - a)/ho)
            print("hopt=",ho, "m=",m, "nsteps=", n, "C=",C)
            S =np.array([],dtype=np.double)
            H = np.array([],dtype=np.double)
            Rh = np.array([],dtype=np.double)
        integsum_p = integsum

    print("S=",compute_n_sums(n,a,b,r),"P=",P,"nsteps=",n, "h=", (b-a)/n, "C=", C)
    if (r % 2 == 0):
        Rn = (integsum - integsum_p) / (l ** (r - 1) - 1)
    else:
        Rn = (integsum - integsum_p) / (l ** (r) - 1)
    print("Rn=",Rn)


#def hopt(m,eps,h,l,S1,S2):
#    return h*(eps*(1-l**(-m))/abs(S2-S1))**(1/(int(m)+1))
def hopt(h1,Rh1,m,eps):
    return h1 * (eps / abs(Rh1)) ** (1 / (abs(int(m)) +1))
def pmoment(p, a, b, r):
    # p - weight function
    # r - count
    # a,b - limits
    m = lambda k: (b**(k+10/7))/(k+10/7) - (a**(k+10/7))/(k+10/7)
    M = np.array(list(map(m,range(0,r))),dtype=np.double)
    #sp.integrate(p*x**j)
    return M

def newtonqf(M,N):
    #N - nodes
    #M - moments
    T = np.empty((0,len(N)),dtype=np.double)
    for i in range(len(N)):
       T=  np.append(T, [N**i], axis=0)
    A=np.linalg.inv(T)@np.transpose(M)
    S=0
    for i in range(len(N)):
        S+=A[i]*lft(N[i])
    return S





def gaussqf(M, p, a, b, lf):
    l = len(M)  # count of moments
    k = l // 2  # count of equations
    M_ = []
    # Making system for a1..an

    for i in range(k - 1, 2 * k - 1):
        S = []
        for j in range(0, k):
            S.append(M[i - j])

        M_.append(S)

    m = M[k:l]

    m = [-1 * x for x in m]
    m = np.array(m, dtype='float')
    M_ = np.array(M_, dtype='float')
    x = sp.symbols('x')
    A = solve(M_, m)
    n = len(A)  # polynomial degree
    A = list(A)
    A = list(reversed(A))
    A.append(1)

    x = sp.symbols("x")
    eq = 0
    # Make equation
    for i in range(n, -1, -1):
        eq += A[i] * x ** i
    # Find nodes.
    # Nodes must be different and must be in [a,b]

    X = list(sp.solve(eq))
    X = [sp.re(x) for x in X]
    # omega and derivative of omega
    w = 1
    for i in range(len(X)):
        w = w * (x - X[i])
    dw = sp.diff(w, x)
    dw = sp.lambdify(x, dw)
    A = []
    print(w)
    # find coef of the quadrature formula
    for i in range(len(X)):
        A.append(sp.Integral(p * w / ((x - X[i]) * (dw(X[i]))), (x, a, b)).as_sum(20, method="midpoint").n())

    X = np.array(X, dtype=np.float)
    A = np.array(A, dtype=np.float)

    integ_sum = 0
    for i in range(len(A)):
        integ_sum = integ_sum + lf(X[i]) * A[i]
    print(integ_sum)


def main():
    a = 0
    b = 1.5
    r = 5
    alpha = 0
    betta = 3 / 7
    eps=10**(-6)
    p = (x - a) ** (-alpha) * (b - x) ** (-betta)
    pt = 1 / t ** (-3 / 7)
    # lft = lambda t: 2.7 * np.cos(3.5*(4.3-t))*np.exp(-7*(4.3-t)/3) + 4.4 * np.sin(2.5*(4.3-t)) * np.exp(5*(4.3-t)/3) + 2
    M = pmoment(pt, a, b, r)
    print(newtonqf(np.array(M), np.linspace(a, b, r)))
    compositeqf(0, 1.5, r,eps)

    #print(gaussqf(M,p,a,b,lft))



if __name__ == "__main__":
    x = sp.symbols("x")
    t = sp.symbols("t")
    lf = lambda xk: 2.7 * np.cos(3.5 * xk) * np.exp(-7 * xk / 3) + 4.4 * np.sin(2.5 * xk) * np.exp(5 * xk / 3) + 2
    lft = lambda t: lf(4.3 - t)
    #print("hopt=", hopt(6.882590642777202,0.000001,-0.09465219184999334,0.75))
    #print(Eitken([153.84629292132496,148.66255049472562,148.79681589976022],2))
    #Richardson([148.85026938543876,148.82540222835274,148.82503458704028],[0.75,0.375,0.1875],int(Eitken([148.85026938543876,148.82540222835274,148.82503458704028],2))+1)
    main()