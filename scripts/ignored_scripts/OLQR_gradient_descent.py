from methods import StabilizingGainManifold
import numpy as np 
import matplotlib.pyplot as plt
import control as ct
from scipy.linalg import solve_discrete_lyapunov, block_diag
from typing import Tuple

n = 5
m = 4
p = 3

A, B2 = StabilizingGainManifold.rand_controllable_matrix_pair(n, m)
B1 = np.eye(n)
B = np.block([B1, B2])
C2 = np.random.randn(p,n)
C1 = np.eye(n)
C = np.block([
    [C1],
    [C2]
])
D11 = np.zeros((n,n))
D12 = np.zeros((n,m))
D21 = np.zeros((p,n))
D22 = np.zeros((p,m))
D = np.block([
    [D11, D12],
    [D21, D22]
])


Sigma = np.eye(n)
Q = np.eye(n)
R = np.eye(m)
W = .1*np.eye(n)
V = .1*np.eye(p)

F, _, _ = ct.dlqr(A, B2, Q, R)
L, _, _ = ct.dlqr(A.T, C2.T, W, V)
F = -F
L = -L.T

A_K = A + B2@F + L@C2 + L@D22@F
B_K = -L
C_K = F
print(B2.shape)
print(A_K.shape, B_K.shape, C_K.shape)

A1 = np.block([
    [A, np.zeros((n,n))],
    [np.zeros((n,n)), A_K]
])

A2 = np.block([
    [B2, np.zeros((n,m))],
    [np.zeros((n,p)), B_K]
])

def decompose(K: np.ndarray) -> np.ndarray:
    A_K = K[m:,p:]
    B_K = K[m:,:p]
    C_K = K[:m,p:]
    return A_K, B_K, C_K

def compose(A_K: np.ndarray, B_K: np.ndarray, C_K: np.ndarray) -> np.ndarray:
    return np.block([
        [np.zeros((m, p)), C_K],
        [B_K, A_K]
    ])

def stability_constraint(K):
    A_K, B_K, C_K = decompose(K)
    return np.block([
        [A, B@C_K],
        [B_K@C, A_K]
    ])

def X(K: np.ndarray) -> np.ndarray:
    A_K, B_K, C_K = decompose(K)
    bias = block_diag(W, B_K@V@B_K.T)
    X_K = StabilizingGainManifold.lyapunov_map(stability_constraint(K), bias)
    return X_K

