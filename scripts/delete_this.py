from methods import StabilizingGainManifold
import numpy as np 
import matplotlib.pyplot as plt
import control as ct
import scipy as sp
from itertools import product 
from scipy.linalg import solve_discrete_lyapunov

n = 4
m = 3
eps = .8
while True:
    A, B = StabilizingGainManifold.rand_controllable_matrix_pair(n, m)
    Sigma = np.eye(n)
    Q = np.eye(n)
    R = np.eye(m)
    M = StabilizingGainManifold(A, B, Q, R, Sigma, True)
    K_opt = M.dlqr()
    print(M.gain_spectral_radius(K_opt))
    
    if M.gain_spectral_radius(K_opt) > eps + .1:
        break
print(M.gain_spectral_radius(K_opt))

# K = M.rand()
# while M.gain_spectral_radius(K) >= eps:
#     K = M.rand()


# tol = 1e-3
# norm_grad_hist = []
# while True:
#     alpha = .001
#     f_K = M.f(K)
#     grad_f_K = M.grad_f_2(K, eps)
#     K = K - alpha*M.grad_f_2(K, eps)
#     norm_grad_hist.append(np.linalg.norm(grad_f_K)**2)
#     out = [np.linalg.norm(grad_f_K)**2, f_K, M.gain_spectral_radius(K), np.linalg.norm(K - K_opt)**2]
#     out = np.round(out, 2)
#     print(
#         f"sq norm of grad: {out[0]}, f(K): {out[1]}, rho(A + BK): {out[2]}, |K - K_opt|^2: {out[3]}"
#     )
#     if out[0] < tol:
#         break

# plt.semilogy(norm_grad_hist)
# plt.grid()
# plt.show()
