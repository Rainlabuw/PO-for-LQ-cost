from methods import StabilizingGainManifold
import numpy as np 
import matplotlib.pyplot as plt
import control as ct
from scipy.linalg import solve_discrete_lyapunov

def sigma_max(A):
    return np.linalg.svd(A)[1][0]

def sigma_min(A):
    return np.linalg.svd(A)[1][-1]

def stability_margin(A, res=1000):
    n = A.shape[0]
    f = lambda theta: sigma_min(A - np.exp(1j*theta)*np.eye(n))
    tspan = np.linspace(0,2*np.pi,res)
    y = [] 
    for t in np.linspace(0,2*np.pi,res):
        y.append(f(t))
    return min(y)


n = 5
m = 3

while True:
    A, B = StabilizingGainManifold.rand_controllable_matrix_pair(n, m)
    Sigma = np.eye(n)
    Q = np.eye(n)
    R = np.eye(m)
    M = StabilizingGainManifold(A, B, Q, R, Sigma)
    Kopt = M.dlqr()
    if M.spectral_radius(A - B@Kopt) > .8:
        break
tol = 1e-7
alpha = 1e-3
K = M.rand()
eps = .4
rho_hist = []
cost_hist = []
while True:
    A_cl = A - B@K
    rho = M.spectral_radius(A_cl)
    rho_hist.append(rho)
    cost_hist.append(M.f(K))
    print(rho,M.f(K))

    if rho > eps:
        evals, evecs = np.linalg.eig(A_cl)
        for i, eval in enumerate(evals):
            if np.abs(eval) > eps:
                evals[i] = eval/np.abs(eval)*eps
        K = ct.place(A, B, evals)

    alpha = .01
    # while M.f(K - alpha*M.grad_f(K)) >= M.f(K):
    #     alpha = alpha/10
    K = K - alpha*M.grad_f(K)



    if np.linalg.norm(K - Kopt) < tol:
        break

plt.subplot(2,1,1)
plt.semilogx(rho_hist)
plt.semilogx(M.spectral_radius(A - B@Kopt)*np.ones(len(rho_hist)))
plt.title('spectral radius')
plt.grid()

plt.subplot(2,1,2)
plt.semilogx(cost_hist)
plt.grid()
plt.title('cost')
plt.show()