from methods import StabilizingGainManifold
import numpy as np 
import matplotlib.pyplot as plt
import control as ct
import itertools
from scipy.linalg import block_diag
from typing import Union, Tuple
n = 5
m = 4
p = 3
A, B = StabilizingGainManifold.rand_controllable_matrix_pair(n,m)
C = np.random.randn(p,n)
Sigma = np.eye(n)

Q = np.eye(n)
R = np.eye(m)
C_dist = np.eye(n)
C_noise = np.eye(p)
A_bar = block_diag(A, np.zeros((n,n)))
B_bar = block_diag(B, np.eye(n))
C_bar = block_diag(C, np.eye(n))

def compose(K: tuple[np.ndarray]) -> np.ndarray:
    A_K, B_K, C_K = K
    m = C_K.shape[0]
    d = B_K.shape[1]
    return np.block([
        [np.zeros((m,d)), C_K],
        [B_K, A_K]
    ])

def decompose(K: np.ndarray) -> tuple[np.ndarray]:
    A_K = K[m:, p:]
    B_K = K[m:, :p]
    C_K = K[:m, p:]
    return (A_K, B_K, C_K)

def A_cl(K: np.ndarray) -> np.ndarray:
    return A_bar + B_bar@K@C_bar

def Q_tilde(K: np.ndarray) -> np.ndarray:
    C_K = decompose(K)[2]
    return block_diag(Q, C_K.T@R@C_K)

def P(K: np.ndarray):
    return StabilizingGainManifold.lyapunov_map(A_cl(K).T, Q_tilde(K))

def f(K: np.ndarray) -> float:
    if np.max(np.abs(np.linalg.eig(A_cl(K))[0])) >= 1:
        return np.inf
    return np.trace(P(K)@block_diag(Sigma, np.zeros((n,n))))

def inner(V: np.ndarray, W: np.ndarray) -> float:
    return np.trace(V.T@W)

def randvec() -> np.ndarray:
    V_A = np.random.randn(n,n)
    V_B = np.random.randn(n,p)
    V_C = np.random.randn(m,n)
    V = compose((V_A, V_B, V_C))
    return V/np.sqrt(inner(V,V))

def dP(K: np.ndarray, V: np.ndarray) -> np.ndarray:
    C_K = decompose(K)[2]
    V_C = decompose(V)[2]
    M1 = C_bar.T@V.T@B_bar.T@P(K)@A_cl(K)
    M2 = block_diag(np.zeros((n,n)), C_K.T@R@V_C + V_C.T@R@C_K)
    return StabilizingGainManifold.lyapunov_map(
        A_cl(K).T,
        M1 + M1.T + M2
    )

def df(K: np.ndarray, V: np.ndarray) -> np.ndarray:
    return np.trace(dP(K,V)@block_diag(Sigma, np.zeros((n,n))))
    # C_K = decompose(K)[2]
    # V_C = decompose(V)[2]
    # M1 = C_bar.T@V.T@B_bar.T@P(K)@A_cl(K) + block_diag(
    #     np.zeros((n,n)), V_C.T@R@C_K
    # )

    # M2 = StabilizingGainManifold.lyapunov_map(
    #     A_cl(K), block_diag(Sigma, np.zeros((n,n)))
    # )
    # return 2*np.trace(M1@M2)

def grad_f(K: np.ndarray) -> np.ndarray:
    out = np.zeros((m + n, n + p))
    for i in range(m + n):
        for j in range(n + p):
            if i > m or j > p:
                E_ij = np.zeros((m + n, n + p))
                E_ij[i,j] = 1
                out[i,j] = df(K, E_ij)
    return out


Mc = StabilizingGainManifold(A, B, Q, R, Sigma)
Mo = StabilizingGainManifold(
    A.T, C.T, C_dist, C_noise, Sigma
)
F0 = Mc.dlqr()
L0 = Mo.dlqr().T
A_K0 = A - B@F0 - L0@C
B_K0 = L0
C_K0 = -F0
K0 = compose([A_K0, B_K0, C_K0])

F = ct.place(A, B, np.random.rand(n))
L = ct.place(A.T, C.T, np.random.rand(n))
L = L.T
A_K = A - B@F - L@C
B_K = L
C_K = -F

K_list = (A_K, B_K, C_K)
K = compose(K_list)
V = randvec()

error = []
for t in range(1000):
    alpha = 1e-15
    grad_f_K = grad_f(K)
    f_K = f(K)
    # while f(K - alpha*grad_f_K) > f_K:
    #     alpha = alpha/10
    K = K - alpha*grad_f_K
    error.append(f_K)
    print(t, np.round(f_K), alpha, np.linalg.norm(K - K0))

plt.figure()
plt.semilogy(error)
plt.grid()
plt.show()




