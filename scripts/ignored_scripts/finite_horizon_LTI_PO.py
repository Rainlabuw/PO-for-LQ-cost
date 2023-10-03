import numpy as np
import control as ct
import matplotlib.pyplot as plt
import math
from methods import rand_controllable_matrix_pair, finite_horizon_LQR_cost

n = 2
m = 1
N = 5
Q = np.eye(n)
R = np.eye(m)
A, B = rand_controllable_matrix_pair(n,m)
x0 = np.random.randn(n)
u0 = np.random.randn(m,N)
eps = 1e-5

def J(x0, u):
    return finite_horizon_LQR_cost(x0, u, A, B, Q, R, N)

def grad_J(x0, u):
    m = u.shape[0]
    N = u.shape[1]
    J0 = J(x0, u)
    out = np.zeros((m,N))
    for i in range(m):
        for k in range(N):
            E_ik = np.zeros((m,N))
            E_ik[i,k] = 1
            out[i,k] = (J(x0, u + eps*E_ik) - J0)/eps
    return out
K, _, _ = ct.dlqr(A,B,Q,R)
u_opt = np.zeros((m,N))
xk = x0
for k in range(N):
    u_opt[:,k] = -K@xk
    xk = (A - B@K)@xk
J_opt = J(x0, u_opt)


u = u0
num_iter = int(1e6)
error = np.zeros(num_iter)
for t in range(num_iter):
    u = u - eps*grad_J(x0, u)
    error[t] = J(x0,u)
    print(t, np.round(error[t],3), np.round(J_opt,3))
print(error[-1])

