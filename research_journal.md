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
$\newcommand{\rank}{\text{rank}}$
$\newcommand{\diag}{\text{diag}}$
$\newcommand{\cone}{\text{cone}}$
$\newcommand{\co}{\text{co}}$
$\newcommand{\Int}{\text{int}}$
$\newcommand{\cl}{\text{cl}}$
$\newcommand{\spec}{\text{spec}}$

# Journal

## 10/10/23
I have lots of updates. Let $$\alpha(A):=\max_{\lambda \in \spec(A)} \Re(\lambda)$$ be the **spectral abscissa**. So, $A \in \R^{n \times n}$ is Hurwitz stable if and only if $\alpha(A) < 0$. 
Define $$\beta_{\cal{S}}(A):=\min_{E \in \R^{n\times n}}\{\|E\|_F:\alpha(A+E)=0\}$$ as the distance to space of Hurwitz unstable matrices. We call $\beta_{\cal{S}}$ the **instability distance**. Also, define $$\beta_{\cal{G}}(K):=\min_{L\in \R^{m \times n}}\{\|L\|_F:\alpha(A+BK+BL)=0\}$$ as the distance to the nearest destabilizing gain. We'll call $\beta_{\cal{G}}$ the **gain instability distance**.

It was proven that $$\beta_\cal{S}(A):=\min_{E \in \R^{n\times n}}\{\|E\|_{F}:\exists \mu \in \R \ \st \det(A-\mu j I + E)=0\}$$$$=\min_{\mu\in \R}\sigma_{\min}(A-\mu j I).$$
This follows from the fact that $\alpha(A + E)=0$ if and only if there exist $\mu \in \R$ such that $\det(A-\mu j I + E) = 0$. 

Observe the following:
$$\begin{align*}
\|B\|_{F}\cdot \beta_{\cal{G}}(K)&= \|B\|_{F}\cdot\min_{L\in \R^{m \times n}}\{\|L\|_F:\alpha(A+BK+BL)=0\} \\
&\geq \min_{L \in \R^{m \times n}}\{\|BL\|_{F}:\alpha(A+BK+BL)=0\} \\
& \geq \min_{E \in \R^{n \times n}} \{\|E\|_{F}:\alpha(A+BK+E)=0\} = \beta_{\cal{S}}(A+BK)
\end{align*}$$

Therefore $$\beta_{\cal{G}}(K) \geq \|B\|_{F}^{-1} \beta_{\cal{S}}(A+BK)$$


## 10/5/23
Recall $S_+^n$ is a proper cone in the space of symmetric matrices $S^n$. By proper, we mean it is closed, convex, solid, and pointed. 
Let $$\pi(S_+^n) := \{L \in\cal{L}(S^n,S^n):L(S_+^n) \subset S_+^n$$ be the vector space of linear operators on $S^n$ that are $S_+^n$-invariant. Recall that $\pi(S_+^n)$ is itself a proper cone in $\cal{L}(S^n,S^n)$. Also $$\pi(S_+^n)^\circ=\{L \in \cal{L}(S^n,S^n):L(S_+^n-\{0\}) \subset S_{++}^n\}.$$ This implies that operators in $\pi(S_+^n)^\circ$ are $S_{++}^n$-invariant.

Define $\bb{L}:\cal{S} \to \cal{L}(S^n,S^n)$ as follows: $$\bb{L}(A)Q\equiv L_AQ \equiv \sum_{k=0}^\infty A^kQ(A^T)^k.$$ Note that $\bb{L}$ is nonlinear, injective, and analytic.  Define $$\cal{C}:=\bb{L}(\cal{S}) \subset \pi(S_+^n).$$

Suppose I equip $\pi(S_+^n)$ with a Riemannian metric so that it is geodesically convex, such as a log-barriar function. 

## 10/3/23
Mehran liked some of the stuff I showed him below. He wasn't sold on it completely, but he saw potential. He doesn't like the idea of an invertible $B$. I don't blame him cause I don't either. He also again pointed to another application. If I could equip $\cal{S}$ with a Riemannian metric so that $\cal{S}$ is geodesically convex, then it could make projection a much much easier process. It's an interesting high level idea. 

### Notes on Cones
I want to write a bit about cones real quick. I took these from the paper **Matrices which Leave a Cone Invariant** by Abraham Berman. 

Let $V$ be a vector space of finite dimensions. A **cone** $K \subset V$ is a set closed under conic combinations. That is, for $x_i \in K$ and $\alpha_i \geq 0$, we have $\alpha_i x_i \in K$. So, $K$ is a cone iff $\cone(K) = K$. 

Some authors instead only require $\alpha_i > 0$, implying that cones do not need to contain $0$. The paper I read requires $\alpha_i \geq 0$. In the latter case, open cones are a possibility. 


A cone $K$ is **convex** if it's closed under convex combinations. That is, for $x_i \in K$ and $\alpha_i \geq 0$ with $\alpha_1+...+\alpha_n=1$, we have $\alpha_i x_i \in K$. In other words, $K$ is a closed convex cone iff $\cone(K)=K$ and $\co(K) = K$. 

A cone $K$ is **pointed** if $K \cap (-K) = \{0\}$. 

A cone $K$ if **solid** if $\Int(K) \not = \emptyset$. 

A cone is **polyhedral** if there exists a finite set of points $x_1,...,x_n \in K$ such that $\cone(\{x_i\})=K$.

The **dual** of a cone $K$ is defined as $K^*:=\{y \in V: \inner{x,y} \geq 0 \ \forall x \in V\}$.

A cone is **proper** if it is pointed, closed, solid, and convex. 

We have the following lemmas. Let $S \subset V$.  
* $S^{**}=\overline{\cone(S)}$
* $K$ is a closed, convex cone iff $K = K^{**}$.
* A closed convex cone $K$ is pointed iff $K^*$ is solid. 
* Let $K$ be a pointed, closed, convex cone. Then $\Int(K^*)=\{y \in K^*:x \in K-\{0\} \implies \inner{x,y} > 0 \}$

A proper cone induces a partial order on $V$: $y \preceq_K x$ iff $x-y \in K$. 

Let $K$ and $F \subset K$ be pointed closed cones. Then $F$ is called a **face** of $K$ if for all $x \in F$, we have the implication $0 \preceq y \preceq x \implies y \in F$. We call $\{0\},K$ **trivial** faces; with all other faces are  **non-trivial**.

Let $K_1$ and $K_2$ be proper cones of vector spaces $V$ and $W$, respectively. Define $\pi(K_1,K_2)$ as the set of linear maps $A:V \to W$ such that $AK_1 \subset K_2$. Then $\pi(K_1,K_2)$ is a proper cone in $\cal{L}(V,W)$. Furthermore, if $K_1,K_2$ are polyhedral, then $\pi(K_1,K_2)$ is too. If $K_1=K_2=K$, we just write $\pi(K)=\pi(K,K)$ for brevity. 

If $K$ is a proper cone, then $\pi(K)$ is closed under matrix multiplication, in addition to being a proper cone. 

The set $V = S^n$, the vector space of symmetric $n \times n$ matrices. Let $S^n_+$ be the set of positive semi-definite matrices. Note $S^n_+$ is proper; that is, closed, convex, and pointed. 

The space of positive-definite matrices, $S^n_{++}$, forms an open cone, if we assume that conic combinations consists only of positive weighted combinations of the points therein. There's a particular metric call the Hilbert metric designed for convex cones. 

Let $C$ be a closed, convex, pointed cone. We will denote $\Omega := C^\circ$. Recall $x \preceq y$ iff $y-x \in C$ So $(V,\preceq)$ is a partially ordered linear space. Furthermore, it is Archemedean. That is, if $ny \preceq y$ for all $n=0,1,2,...$, then $y \preceq 0$. 

For $x \in V$ and $y \in \Omega$, we let $$M(x,y) := \inf\{\lambda : x \preceq \lambda y\}$$ and $$m(x,y) := \sup\{\mu : \mu y \preceq x\}.$$ Hilbert's projective pseudometric is defined as 

### Spectral properties of matrices in $\pi(K)$
The matrices in $\pi(K)$ are called $K$**-nonnegative** and are said to **leave** $K$ **invariant**. $A \in \pi(K)$ is called $K$**-positive** if $A(K - \{0\}) \subset \Int(K)$.

If $A \in \pi(K)$, then
* $\rho(A)\in \spec(A)$, where $\rho(.)$ denotes spectral radius.
* if $\lambda \in \spec(A)$ such that $|\lambda| = \rho(A)$, then $\deg(\lambda) \leq \deg(\rho(A))$, where $\deg(.)$ denotes algebraic multiplicity. 
* $K$ contains an eigenvector of $A$ corresponding to $\rho(A)$. 
* $K^*$ contains an eigenvector of $A^T$ corresponding to $\rho(A)$. 

Let $A$ be some linear operator on $V$. Suppose $\rho(A)$ is an eigenvalue of $A$ and suppose $\deg(\rho(A)) \geq \deg(\lambda)$ for all $\lambda \in \spec(A)$ with $|\lambda| = \rho(A)$. Then there exists a proper cone $K \subset V$ for which $A \in \pi(K)$. 


### Lyapunov stability manifold

Let $\cal{S}$ denote the set of Schur stable $n \times n$ matrices. For $A \in \cal{S}$, define the map $L(A,Q)=X$, where $X$ is the unique solution to $A^TXA+Q=X$. Recall $L(A,Q)=\sum_{k=0}^\infty (A^T)^kQA^k$. Set $L_A(Q):=L(A,Q)$. Obviously, the inverse of $L_A$ is $L_A^{-1}(X)=A^TXA-X$. Recall that $L_A:S^n \to S^n$ is invertible, $S_+^n$-invariant, and $S_+^n$-positive (wrt $S^n$). It is also invertible on $\R^{n \times n}$. 

I'm fairly confident that $\cal{S}$ can be equipped with the following function, which is probably a Riemannian metric: $\inner{V,W}_A:=\tr(V^TWL(A,I))$. This metric has coercive properties, and probably has some interesting properties. I'll denote this the **Lyapunov metric**.

However, I'm interested in another metric. Let $\cal{L}(S^n):= \cal{L}(S^n,S^n)$ be the set of linear operators on the vector space of $n \times n$ symmetric matrices. Recall $S_+^n \subset S^n$ is a proper cone. So, $\pi(S_+^n) \subset \cal{L}(S^n,S^n)$ is also a proper cone that is closed under function composition. The cone of $S^n_+$-invariant operators. We can go one step further and define the cone of $S^n_+$-positive operators, denoted $\pi(S_+^n)^\circ$.  

Now, can we equip $\pi(S_+^n)^\circ$ with some cone-friendly metric? in particularly, since it is proper, can we find a metric such that is becomes a Hadamard manifold? I bet we can. Let's put a pin in that.

Now, define the set $\cal{C}:=\{L_A:A \in \cal{S}\}$. This set has some interesting properties. First, $\cal{C} \subset \pi(S_+^n)^\circ$. Second, $\partial \cal{C} \subset \partial \pi(S_+^n)^\circ$ (I need to confirm this). What does $\cal{C}$ equipped with the subset metric look like?

I call this restriction the conical Lyapunov metric. 
## 10/2/23
Suppose we have $x_{k+1}=Ax_k+Bu_k$. I grown interest in solving the following problem:
$$\begin{align} \min_{K \in \cal{G}} f(K) \\ 
\rho(A+BK) \leq \epsilon \end{align}$$

Note that the robust constraint $\rho(A+BK)$ has other forms, such as $s_{\cal{G}}(K)\geq \epsilon$ or $s_{\cal{S}}(A+BK)\geq \epsilon$. All 3 are heavily related to one another. The latter also has a closed form expression:
$$s_{\cal{S}}(A+BK)=\min_{\theta \in (-\pi,\pi]} \sigma_\min(A+BK-e^{j\theta}I)$$

However the constraint $\rho(A+BK)$ has some interesting properties I could take advantage of. The problem could be solved via gradient descent. 

To make the problem more approachable, I will assume $B \in \R^{n \times n}$ is invertible. So, the map $g:K \mapsto A+BK$ is invertible with $g^{-1}(V)=B^{-1}(V-A)$. Define $$\cal{G}_\epsilon:=\{K \in \R^{m \times n}:0\leq \rho(A+BK) \leq \epsilon \}.$$ Let $K \in \cal{G}-\cal{G}_\epsilon$. What is $$\Pi_{\cal{G}_\epsilon}(K) := \min_{L \in \cal{G}_\epsilon}\|K-L\|,$$ where $\|.\|$ is the Frobenius norm? Note $\|K-L\|=\|B^{-1}(V-W)\|$, where $V=A+BK$ and $W=A+BL$. So $\|K-L\|\leq \|B^{-1}\|\|V-W\|$. It follows $$\Pi_{\cal{G}_\epsilon}(K) \leq \|B^{-1}\|\min_{W \in \cal{S}^c}\|V-W\|=\|B^{-1}\|\Pi_{\cal{S}_\epsilon}(V),$$ where $\cal{S}:=\{V:\rho(V) <1 \}$ and $\cal{S}_{\epsilon}:=\{V \in \R^{n \times n}:0 \leq \rho(V) \leq \epsilon \}$. I find this upper bound very exciting since $A$ makes no appearance.

Furthermore, there are ways of computing $\Pi_{\cal{S}_\epsilon}(V)$. I'm going to investigate them and see if they help with computing the projected gradient descent. 

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

It might make mroe sense to study the program $$\begin{align}\min f(K) \\ s_{\cal{S}}(A+BK) \leq \epsilon\end{align}$$ since we have a closed-form expression: $$s_{\cal{S}}(A+BK)=\min_{0 \leq \theta \leq 2\pi} \sigma_\min(A+BK-e^{j\theta}I)$$