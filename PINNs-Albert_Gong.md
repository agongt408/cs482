# Optimal control of PDEs using physics-informed neural networks

Albert Gong

May 4, 2023
 
This blog post was written for Yale’s CPSC 482: Current Topics in Applied Machine Learning course. I will first walk through the paper: https://arxiv.org/abs/2111.09880. Later I will provide my own thoughts and comments on the paper.

## Motivation

We aim to solve a PDE-constrained minimization problem. That is, we 

General/forward PDE problem: Find $u(x,t)$ such that
- [Volume] $\mathcal{F}(u(x, t), x, t)=0$ for all $x \in \Omega, t \in[0, T]$
- [Boundary] $\mathcal{B}(u(x, t), x, t)=0$ for all $x \in$ boundary of $\Omega, \mathrm{t} \in[0, T]$
- [Initial] $\mathcal{I}(u(x, 0), x)=0$ for all $x \in \Omega$

Examples:
1. Laplace equations (models steady-state of heat dissipation)

$$
F(u(x, t), x, t)=\frac{\partial^2 u}{\partial x^2}+\frac{\partial^2 u}{\partial t^2}=0
$$

$$
B(u(x, t), x, t)=\begin{cases}
\sin (\pi x), & \text { if } t=1 \\
0, & \text { if } t=0 \\
0, & \text { if } x=0 \\
0, & \text { if } x=1
\end{cases}
$$

2. Burgers equations (simple model of viscous fluid motions)
\begin{aligned}
& F(u(x, t), x, t)=\frac{\partial u}{\partial t}+u \frac{\partial u}{\partial x}-\nu \frac{\partial^2 u}{\partial x^2}\\
& B(u(x,t),x,t)=\text{periodic boundary conditions} \\
& I(u(x,t),x,t)=u(x,t) - \text{"analytic solution at t=0"}
\end{aligned}

3. Kuromoto-Sivashinsky equations (models the diffusive–thermal instabilities in a laminar flame front)
\begin{aligned}
& F(u(x, t), x, t)=\frac{\partial u}{\partial t}+u \frac{\partial u}{\partial t}+\frac{\partial^2 u}{\partial x^2}+\frac{\partial^4 u}{\partial x^4}-f(x, t) \\
& B(u(x,t),x,t)=\text{periodic boundary conditions} \\
& I(u(x,t),x,t)=u(x,0)-\cos \left(\frac{2 \pi x}{10}\right)-\operatorname{sech}\left(\frac{x-L / 2}{5}\right)
\end{aligned}

4. Navier-Stokes equations (general model of viscous fluid motions)
\begin{aligned}
& F_1(u(x, t), x, t)=(u \cdot \nabla) u+\nabla p-\frac{1}{\operatorname{Re}} \nabla^2 u \\
& F_2(u(x, t), x, t)=\nabla \cdot u
\end{aligned}


Optimal control PDE problem: 
Solve $\text{argmin}_{c\in \mathcal{C}} \mathcal{J}(u)$, where
- [Volume] $\mathcal{F}(u(x, t), x, t; c_v(x,t))=0$ for all $x \in \Omega, t \in[0, T]$
- [Boundary] $\mathcal{B}(u(x, t), x, t; c_b(x,t))=0$ for all $x \in$ boundary of $\Omega, \mathrm{t} \in[0, T]$
- [Initial] $\mathcal{I}(u(x, 0), x; c_0(x))=0$ for all $x \in \Omega$

Examples:
1. Laplace equations + control
$$
\mathcal{J}(u)=\int_0^1\left|\frac{\partial u}{\partial y}(x, 1)-q_d(x)\right|^2 d x, \quad q_d(x)=\cos (\pi x)
$$

2. Burgers equations + control
$$\mathcal{J}(u)=\frac{1}{2} \int_0^L\left|u(x, T)-u_a(x, T)\right|^2 d x $$

3. Kuromoto-Sivashinsky equations + control
$$\mathcal{J}(u, f)=\frac{1}{2} \int_0^T \int_0^L\left(|u(x, t)|^2+\sigma|f(x, t)|^2\right) d x d t$$

4. Navier-Stokes equations + control
$$\mathcal{J}(\mathbf{u})=\frac{1}{2} \int_0^{L_y}\left(\left|u\left(L_x, y\right)-u_{\text {parab }}(y)\right|^2+\left|v\left(L_x, y\right)\right|^2\right) d y, \quad u_{\text {parab }}(y)=\frac{4}{L_y^2} y(1-y)$$

## Methodology

Model $u(x,t)$ using a fully-connected neural network $u_{NN}(x,t; \theta_u)$ with parameters $\theta$ and inputs $x$ and $t$. Model $c(x,t)$ using fully-connected neural network $u_{NN}(x,t; \theta_c)$ with parameters $\theta_c$, respectively, and inputs $x$ and $t$. Approximate the optimal control PDE problem by the following optimization problem:
\begin{equation}
\text{argmin}_{\theta_u,\theta_c} \mathcal{L}\left(\theta_{u}, \theta_{c} \right)
\end{equation}
where
\begin{equation}
\begin{aligned}
\mathcal{L}\left(\boldsymbol{\theta}_{\mathbf{u}}, \boldsymbol{\theta}_{\mathbf{c}}\right)= & \frac{w_r}{N_r} \sum_{i=1}^{N_r}\left|\mathcal{F}\left[\mathbf{u}_{\mathrm{NN}}\left(\mathbf{x}_i^r, t_i^r ; \boldsymbol{\theta}_{\mathbf{u}}\right) ; \mathbf{c}_{\mathrm{NN}}\left(\mathbf{x}_i^r, t_i^r ; \boldsymbol{\theta}_{\mathbf{c}}\right)\right]\right|^2+\frac{w_b}{N_b} \sum_{i=1}^{N_b}\left|\mathcal{B}\left[\mathbf{u}_{\mathrm{NN}}\left(\mathbf{x}_i^b, t_i^b ; \boldsymbol{\theta}_{\mathbf{u}}\right) ; \mathbf{c}_{\mathrm{NN}}\left(\mathbf{x}_i^b, t_i^b ; \boldsymbol{\theta}_{\mathbf{c}}\right)\right]\right|^2 \\
& +\frac{w_0}{N_0} \sum_{i=1}^{N_0}\left|\mathcal{I}\left[\mathbf{u}_{\mathrm{NN}}\left(\mathbf{x}_i^0, 0 ; \boldsymbol{\theta}_{\mathbf{u}}\right); \mathbf{c}_{\mathrm{NN}}\left(\mathbf{x}_i^0, 0 ; \boldsymbol{\theta}_{\mathbf{c}}\right)\right]\right|^2+w_{\mathcal{J}} \mathcal{L}_{\mathcal{J}}\left(\boldsymbol{\theta}_{\mathbf{u}}, \boldsymbol{\theta}_{\mathbf{c}}\right),
\end{aligned}
\end{equation}
$\left\{\mathbf{x}_i^r, t_i^r\right\}_{i=1}^{N_r},\left\{\mathbf{x}_i^b, t_i^b\right\}_{i=1}^{N_b},\left\{\mathbf{x}_i^0\right\}_{i=1}^{N_0}$ represent training samples to estimate the volume, boundary, and initial conditions, and $w_r,w_b,w_0,w_{\mathcal{J}}$ are loss weights.

Line search method:

Solve forward problem once to tune network architecture, distribution of residual points, training hyperparameters (number of epochs, batch size, etc), and weights 𝑤_𝑟, 𝑤_𝑏, and 𝑤_0.

For each 𝑤_𝒥  in a range of values:
Fixing u_NN, train c_NN*
Fixing c_NN*, train u_NN’
Fixing u_NN’, train c_NN’
Return c_NN’ corresponding to the lowest value of J(u_NN’,c_NN’)

### Adjoint-based methods

TODO: Explain adjoint method
- iterative methods, such as the direct-adjoint-looping algorithm

We can enforce the PDE constraints using the method of Lagrange multipliers:
$$
\mathcal{L}(\mathbf{u}, \mathbf{c}, \boldsymbol{\lambda})=\mathcal{J}(\mathbf{u}, \mathbf{c})-\langle\boldsymbol{\lambda}, \mathcal{F}[\mathbf{u} ; \mathbf{c}]\rangle
$$
where the inner product is defined by
$$
\langle\mathbf{a}, \mathbf{b}\rangle=\int_{0}^{T} \int_{\Omega} \mathbf{a}(\mathbf{x}, t)^{\top} \mathbf{b}(\mathbf{x}, t) d \mathbf{x} d t
$$

Then, the constrained problem (2) is equivalent to the unconstrained problem

$$
\mathbf{u}^{*}, \mathbf{c}^{*}, \boldsymbol{\lambda}^{*}=\arg \min _{\mathbf{u}, \mathbf{c}, \boldsymbol{\lambda}} \mathcal{L}(\mathbf{u}, \mathbf{c}, \boldsymbol{\lambda})
$$
whose solution is given by the stationary point(s) of the Lagrangian. This yields the following stationarity conditions:

Stationary condition $u(x,t)$
$$
\left\langle\frac{\partial \mathcal{L}}{\partial \mathbf{u}}, \delta \mathbf{u}\right\rangle=\left\langle\frac{\partial \mathcal{J}}{\partial \mathbf{u}}, \delta \mathbf{u}\right\rangle-\left\langle\boldsymbol{\lambda}, \frac{\partial \mathcal{F}}{\partial \mathbf{u}} \delta \mathbf{u}\right\rangle=\left\langle\frac{\partial \mathcal{J}}{\partial \mathbf{u}}-\frac{\partial \mathcal{F}^{\dagger}}{\partial \mathbf{u}} \boldsymbol{\lambda}, \delta \mathbf{u}\right\rangle=0 \quad \forall \delta \mathbf{u}
$$
This implies
$$
\frac{\partial \mathcal{J}(\mathbf{u}, \mathbf{c})}{\partial \mathbf{u}}-\frac{\partial \mathcal{F}[\mathbf{u}, \mathbf{c}]^{\dagger}}{\partial \mathbf{u}} \boldsymbol{\lambda}=0
$$


Stationarity condition $c(x,t)$
$$
\left\langle\frac{\partial \mathcal{L}}{\partial \mathbf{c}}, \delta \mathbf{c}\right\rangle=\left\langle\frac{\partial \mathcal{J}}{\partial \mathbf{c}}, \delta \mathbf{c}\right\rangle-\left\langle\boldsymbol{\lambda}, \frac{\partial \mathcal{F}}{\partial \mathbf{c}} \delta \mathbf{c}\right\rangle=\left\langle\frac{\partial \mathcal{J}}{\partial \mathbf{c}}-\frac{\partial \mathcal{F}^{\dagger}}{\partial \mathbf{c}} \boldsymbol{\lambda}, \delta \mathbf{c}\right\rangle=0 \quad \forall \delta \mathbf{c}
$$
This implies
$$
\frac{\partial \mathcal{J}(\mathbf{u}, \mathbf{c})}{\partial \mathbf{c}}-\frac{\partial \mathcal{F}[\mathbf{u}, \mathbf{c}]^{\dagger}}{\partial \mathbf{c}} \boldsymbol{\lambda}=0
$$

Stationarity condition $\lambda(x,t)$
$$
\left\langle\frac{\partial \mathcal{L}}{\partial \boldsymbol{\lambda}}, \delta \boldsymbol{\lambda}\right\rangle=-\langle\delta \boldsymbol{\lambda}, \mathcal{F}\rangle=0 \quad \forall \delta \boldsymbol{\lambda},
$$

where we have defined the adjoint $\mathcal{A}^{\dagger}$ of a linear operator $\mathcal{A}$ as
$$
\langle\mathbf{a}, \mathcal{A} \mathbf{b}\rangle=\left\langle\mathcal{A}^{\dagger} \mathbf{a}, \mathbf{b}\right\rangle \quad \forall \mathbf{a}, \mathbf{b}
$$
This implies
$$
\mathcal{F}[\mathbf{u}, \mathbf{c}] = 0
$$

When the stationarity conditions for $u(x,t)$ and $c(x,t)$ are satisfied, we have $\mathcal{J}=\mathcal{L}$, and $14 \mathrm{~b}$ therefore gives the total gradient of the cost objective with respect to the control $\mathbf{c}$,

$$
\frac{\mathrm{d} \mathcal{J}(\mathbf{u}, \mathbf{c})}{\mathrm{d} \mathbf{c}}=\frac{\partial \mathcal{L}(\mathbf{u}, \mathbf{c})}{\partial \mathbf{c}}=\frac{\partial \mathcal{J}(\mathbf{u}, \mathbf{c})}{\partial \mathbf{c}}-\frac{\partial \mathcal{F}[\mathbf{u}, \mathbf{c}]^{\dagger}}{\partial \mathbf{c}} \boldsymbol{\lambda}
$$

For the optimal solution, $\mathrm{d} \mathcal{J}\left(\mathbf{u}^{*}, \mathbf{c}^{*}\right) / \mathrm{d} \mathbf{c}=0$ holds.

We can solve this using the direct-adjoint looping (DAL) iterative algorithm. At each iteration $k$:
1. given the current control $\mathbf{c}^{k}$, solve the forward PDE for $\mathbf{u}^{k}$
    - Use finite-volume method implemention in OpenFOAM
2. given $\mathbf{u}^{k}$ and $\mathbf{c}^{k}$, solve the adjoint PDE (16) for $\boldsymbol{\lambda}^{k}$ in backward time since the adjoint PDE contains a terminal condition instead of an initial condition.

3. Update the control via
$$
\mathbf{c}^{k+1}=\mathbf{c}^{k}-\beta \frac{\mathrm{d} \mathcal{J}\left(\mathbf{u}^{k}, \mathbf{c}^{k}\right)}{\mathrm{d} \mathbf{c}}
$$

### Implementation

The PINN solutions are trained on one GPU (Tesla V100) in TensorFlow. The DAL solutions are all computed using a single CPU core (Core i7-4980HQ or Xeon E5-2683), using the C++ finite-volume solver OpenFOAM for the Laplace and Navier-Stokes equations, and a spectral Python code for the Burgers and Kuramoto-Sivashinsky equations.

## Results

### Forward Problem

Laplace equation:
- Training strategy: we sample 10000 residual training points using a Latin hypercube sampling strategy and we select 160 equally-spaced boundary training points on the boundary of the domain
- 6k epochs

Burgers equation:

Kuramoto-Sivashinsky equation:

Navier-Stokes equations:

### Optimal Control Problem

Laplace equation:
- Training strategy: we sample 10000 residual training points using a Latin hypercube sampling strategy and we select 160 equally-spaced boundary training points on the boundary of the domain

LHS: a square grid containing sample positions is a Latin square if (and only if) there is only one sample in each row and each column

- 10k epochs
- We repeat this procedure for 11 values of $w_J$ between $10^{-3}$ and $10^7$ (**downside: need to select this hyperparameter in practice**)

Burgers equation:

Kuramoto-Sivashinsky equation:

Navier-Stokes equations:

### Computational Efficiency

- For the simpler problems based on the Laplace and Burgers equations, the DAL solution was obtained in much shorter time than the PINN solution. 
- For the more complex problems based on the Kuramoto-Sivashinsky and Navier-Stokes equations, the situation reverses and the PINN solution is obtained in shorter time.

|                       | PINN          | DAL           |
|-----------------------|---------------|---------------|
| Laplace               | 9 min         | 25 min        |
| Burgers               | 19 min        | 1 min         |
| Kuramoto-Sivashinsky  | 2 hours 4 min | 2 hours 55 min|
| Navier-Stokes         | 8 hours 20 min| 28 hours      |

## Thoughts and Comments
* Time performance tradeoff
* Question: what is the right way to evaluate performance on examples which do not have analytic solutions? Can we come up with benchmarks?

For problems with a known analytic solution, they use relative error:
$$\lVert u-u_a\rVert_2/ \lVert u_a\rVert_2$$
where $u_a$ is the analytic solution

the L2 error is estimated using grid (independent from the residual training points)

Question: is relative L2 error the conventional performance metric for DAL? (TODO: look at their original paper)

* TODO: Comparison with neural ODEs
* TODO: Comparison with PDE solvers that solve in one forward pass (Neural ODEs and Fourier Neural Operators)

* Their claims: For the Burgers and Navier-Stokes equations, the optimal control distributions found by DAL yielded a lower cost objective but were less smooth than the ones obtained from PINNs.
    - They only used four examples, probably limited by computational efficiency of more complicated problems
    - How do you test claim of "smoothness"?

* Would like to see a comparison of energy efficiency for PINN (GPU) vs DAL (single CPU score)

* Question: Can OpenFOAM be parallelized?
    - https://www.openfoam.com/documentation/user-guide/3-running-applications/3.2-running-applications-in-parallel


Advantages of PINNs:

Availability of fast deep learning frameworks allow PINNs to perform competitively compared to standard numerical approaches, especially for PDEs that are difficult to solve due to significant nonlinearities, convection dominance, or shocks.

The PINN framework is very flexible in terms of the type of governing equations, boundary conditions, geometries, and cost objective functions that it allows.

AD calculates the exact derivatives of the network output uNN(x,t;θu)
with respect to its inputs x and t. Thus, the various loss components in (4) can be computed exactly without inheriting the truncation error incurred by standard numerical discretization schemes. 

Another advantage of computing derivatives with AD is that the residual points {xi, ti}Nr can be chosen arbitrarily, conferring i=1
PINNs their convenient mesh-free nature.

Disadvantages of PINNs:

Cannot incorporate training data

### Potential Improvements
To improve time efficiency, use PDE approach that learns from training data to generate initial solution, then apply PINNs or DAL.
