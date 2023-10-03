There are research notes related to my studies on #robust-control-theory, #policy-optimization.

# Latex Commands
$\newcommand{\inner}[1]{\langle #1 \rangle}$
$\newcommand{\mat}[1]{\begin{bmatrix} #1 \end{bmatrix}}$
$\newcommand{\bb}[1]{\mathbb{#1}}$
$\newcommand{\cal}[1]{\mathcal{#1}}$
$\newcommand{\tr}{\text{tr}}$
$\newcommand{\E}{\bb{E}}$
$\newcommand{\Bern}{\text{Bernoulli}}$
$\newcommand{\R}{\mathbb{R}}$
$\newcommand{\C}{\mathbb{C}}$
$\newcommand{\ad}{\text{ad}}$
$\newcommand{\diag}{\text{diag}}$

# Journal

## 10/2/23
Suppose we have $x_{k+1}=Ax_k+Bu_k$ and $B \in \R^{n \times n}$ is invertible. Then the map $g(K) =A+BK$ is invertible with $g^{-1}(K)=B^{-1}(M-A)$. Suppose I want to solve the following problem:

$$\begin{align} \min_{K \in \cal{G}} f(K) \\ 
\rho(A+BK) \geq \epsilon \end{align}$$

This can probably be solved with projected gradient descent. Define $\cal{G}_\epsilon:=\{K \in \R^{m \times n}:\epsilon \leq \rho(A+BK) <1\}$. Let $K \in \cal{G}-\cal{G}_\epsilon$. What is $$\Pi_{\cal{G}_\epsilon}(K) := \min_{L \in \cal{G}_\epsilon}\|K-L\|$$ where $\|.\|$ is the Frobenius norm. Note $\|K-L\|=\|B^{-1}(V-W)\|$, where $V=g^{-1}(K)$ and $W=g^{-1}(L)$. So $\|K-L\|\leq \|B^{-1}\|\|V-W\|$. It follows $$\Pi_{\cal{G}_\epsilon}(K) \leq \|B^{-1}\|\min_{W \in \cal{S}^c}\|V-W\|=\|B^{-1}\|\Pi_{\cal{S}_\epsilon}(V),$$ where $\cal{S}:=\{V:\rho(V) <1 \}$ and $\cal{S}_{\epsilon}:=\{V \in \R^{n \times n}:\epsilon \leq \rho(V) <1 \}$. I find this inequality very exciting since $A$ makes no appearance.

Furthermore, there are ways of computing $\Pi_{\cal{S}_\epsilon}(V)$

## 9/29/23
I came up with a cool concept. Let $$\cal{S}:=\{A \in \R^{n \times n}:\rho(A)<1\}$$ be the space of Schur stable matrices. Define $$\cal{G}:=\{K \in \R^{m \times n}:\rho(A + BK)<1\}$$ be the space of stabilizing gain matrices for controllable matrix pair $(A,B)$.

Lately I've been interested in calculating the following quantity. Given $K \in \cal{G}$, let $$s_{\cal{G}}(K):=\inf_{L \in \cal{G}^c}\|K-L\|$$ be the distance of the nearest destabilizing gain matrix. I denote this with $s_{\cal{G}}(.)$ since this is akin to a stability radius. I'll denote this quantity the stability margin of the stabilizing gain space. There are some interesting papers related to this concept. 

A related concept is the stability margin of the schur stable space: $$s_{\cal{S}}(A)=\inf_{U \in \cal{S}^c}\|A-U\|.$$

The above has a closed form expression: $$s_{\cal{S}}(A)=\min_{0 \leq \theta \leq 2\pi} \sigma_{\min}(A-e^{j\theta}I)$$

Since $s_{\cal{S}}$ is easily computable, can I somehow relate it to $s_{\cal{G}}$, which is seemingly much more difficult to compute? Or at least bound it? The answer is yes! 
Let $K \in \cal{G}$. Observe that $$s_{\cal{S}}(A+BK)= \inf_{U \in \cal{S}^c} \|A+BK-U\|$$ Now, given any $U \in \cal{S}^c \subset \R^{n \times n}$, can I find $L \in \cal{G}^c \subset \R^{m \times n}$ such that $U=A+BL$? This deserves some investigation. The answer might be no since $L \mapsto A+BL$ is mapping a lower dimensional linear space $\R^{m \times n}$ into a higher dimensional linear space $\R^{n \times n}$. 

How about this. Given an unstable $U$, can I find $L \in \R^{m \times n}$ such that $\|A+BK-U\|=\|B(K-L)\|$?

Let's assume that it is true nonetheless. Then I can re-write the above as 
$$=\inf_{L \in \cal{G}^c}\|A+BK-(A+BL)\|=\inf_{L \in \cal{G}^c}\|B(K-L)\|\leq\sigma_{\max}(B) s_{\cal{G}}(K).$$ Therefore, if true, we have the neat inequality $$s_{\cal{S}}(A+BK)=\min_{\theta} \sigma_\min(A-e^{j\theta}I)\leq \sigma_\max(B)\cdot s_{\cal{G}}(K)$$

What's interesting about this is the inequality is invariant of $A$!

However, that assumption is a big one. That requires an investigation in of itself.

Here is an optimization problem. Suppose $f(K)=\tr(P_K\Sigma)$ is my usual infinite horizon LQR cost. Suppose I want to solve $$\begin{align}&\min_{K \in \cal{G}} f(K) \\ &s_{\cal{S}}(A+BK) \geq \epsilon \end{align}$$ Or $$\begin{align}&\min_{K \in \cal{G}} f(K) \\ &s_{\cal{G}}(K) \geq \epsilon \end{align}$$

What would such an answer look like? Could I take advantage of the fact that $f$ satisfies the PL inequality?

Suppose also I want to find a stabilizing gain matrix that stabilizes $(A+\Delta A,B)$ for all $\Delta A$ such that $\|\Delta A\|\leq \epsilon$. What would that look like geometrically? This sorta changes the boundary of $\cal{G}$ doesn't it? Is there a relation between that and the functions $s_{\cal{S}}(.),s_{\cal{G}}(.)$?

Here is another problem I think I might have a chance at. 
$$\begin{align}&\min_{K \in \cal{G}} f(K) \\ &\rho(A+BK) \leq \epsilon \end{align}$$

This has a lot of relation to the stability margin. Furthermore, I think there's a way I can solve it quickly. 

Define $\cal{S}_\epsilon:=\{A:\rho(A) \leq \epsilon\}$ and $\cal{G}_\epsilon:=\{K:\rho(A+BK) \leq \epsilon\}=\{K:A+BK \in \cal{S}_\epsilon\}$.

In order to solve that minimization program, we might have to do projected gradient descent. That isn't necessarily a bad thing. Furthermore, the constraint set $\cal{G}_\epsilon$ is very natural and related to $\cal{G}$ since it's just that set contracted a bit. 

The paper **Nearest Ä-stable matrix via Riemannian optimization** by Noferini, Poloni mgiht be of huge help. These people developed an algorithm that can find the nearest matrix to a specific set of matrices whose eigenvalue lie within a closed subset of the complex numbers. 

Suppose $K \in \cal{G} - \cal{G}_\epsilon$. What is the nearest matrix in $\cal{G}_\epsilon$? That is, what matrix $L \in \cal{G}_\epsilon$ minimizes $\|K-L\|_F$? Suppose $A^* \in \cal{S}_\epsilon$ minimizes $\|A+BK-A^*\|_F$. Then maybe I can find $L \in \cal{G}$ such that $A^*=A+BL$? 

Suppose $\rho(A^*) \leq \epsilon<1$. How can I find $K \in \R^{m \times n}$ such that $A^* = A+BK$? 
$K=B^{-1}(A^*-A)$

It might make mroe sense to study the program $$\begin{align}\min f(K) \\ s_{\cal{S}}(A+BK) \geq \epsilon\end{align}$$ since we have a closed-form expression: $$s_{\cal{S}}(A+BK)=\min_{0 \leq \theta \leq 2\pi} \sigma_\min(A+BK-e^{j\theta}I)$$