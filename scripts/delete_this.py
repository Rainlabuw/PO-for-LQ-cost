import numpy as np 
import matplotlib.pyplot as plt
import control as ct

alpha = lambda A: np.max(np.real(np.linalg.eig(A)[0]))
def sigma_min(A):
    _, S, _ = np.linalg.svd(A)
    return np.min(S)

def beta_S(A, bound=10, N=10000):
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

n = 5
m = 3
A = np.random.randn(n,n)
while alpha(A) >= 0:
    A = np.random.randn(n,n)

B = np.random.randn(n,m)
B_norm = np.linalg.norm(B)
Q = np.eye(n)
R = np.eye(m)
K, _, _ = ct.lqr(A,B,Q,R)
beta_G_lower_bound = beta_S(A - B@K)/B_norm

min_r = -1
print(beta_G_lower_bound)
for _ in range(1000):
    V = np.random.randn(m,n)
    V = V/np.linalg.norm(V)
    r = 0
    L = K + r*V
    count = 0
    while alpha(A - B@L) < 0 and count <= 1000:
        r += .01
        L = K + r*V
        count += 1

    if min_r == -1 or r < min_r:
        min_r = r
        print(min_r)