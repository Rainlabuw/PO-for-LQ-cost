import numpy as np 
import matplotlib.pyplot as plt
import control as ct


alpha = lambda A: np.max(np.real(np.linalg.eig(A)[0])) # spectral abscissa

def sigma_min(A: np.ndarray) -> float:
    """Minimum singular value of A
    A: (np.ndarray) arbitrary matrix
    
    Returns: (float) min sing val of A"""
    _, S, _ = np.linalg.svd(A)
    return np.min(S)

def sigma_max(A):
    _, S, _ = np.linalg.svd(A)
    return np.max(S)

def beta_S(A: np.ndarray, bound: float=10, N: int=10000) -> float:
    """Computes the distance between A and the space of Hurwitz unstable 
    matrices.
    
    A: (np.ndarray) matrix of which we wish to compute the instability distance
    bound: (float) bound for which we solve min_{mu in R} sigma_min(A + mu*j*I)
    N: (int) number of iterates in brute force minimization procedure

    Returns: (float) Hurwitz instability distance of A"""
    n = A.shape[0]
    mu_span = np.linspace(-bound, bound, N)
    I = np.eye(n)
    min_s = -1
    for mu in mu_span:
        s = sigma_min(A - mu*1j*I)
        if min_s == -1 or s < min_s:
            min_s = s
    if min_s == -1:
        return np.inf
    return min_s

def test_lower_bound(K, A, B, bound=20, N=100000):
    n = A.shape[0]
    mu_span = np.linspace(-bound, bound, N)
    I = np.eye(n)
    min_s = -1
    P = np.linalg.inv(B.T@B)@B.T
    for mu in mu_span:
        s = sigma_min(P@(A + B@K - mu*1j*I))
        if min_s == -1 or s < min_s:
            min_s = s
    if min_s == -1:
        return np.inf
    return min_s

n = 3
m = 2
A = np.random.randn(n,n)
while alpha(A) >= 0:
    A = np.random.randn(n,n)
E = np.random.randn(n,n)
E = E/np.linalg.norm(E)

aux = sigma_min(A)/sigma_max(E)
bound = 10
N = 100000
t_span = np.linspace(-bound, bound, N)
min_t = -1
for t in t_span:
    a = alpha(A + t*E)
    if a >= 0:
        if abs(t) < min_t or min_t == -1:
            min_t = abs(t)
            print(np.round(min_t,3), np.round(min_t - aux,3))
