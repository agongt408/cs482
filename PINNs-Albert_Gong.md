# Physics-Inspired Neural Networks

## Problem

General/forward PDE problem: Find $u(x,t)$ such that
1. [Volume] $\mathcal{F}(u(x, t), x, t)=0$ for all $x \in \Omega, t \in[0, T]$
2. [Boundary] $\mathcal{B}(u(x, t), x, t)=0$ for all $x \in$ boundary of $\Omega, \mathrm{t} \in[0, T]$
3. [Initial] $\mathcal{I}(u(x, 0), x)=0$ for all $x \in \Omega$

Examples:
\begin{itemize}
\item Laplace equations (models steady-state of heat dissipation)
\begin{aligned}
& F(u(x, t), x, t)=\frac{\partial^2 u}{\partial x^2}+\frac{\partial^2 u}{\partial t^2}=0 \\
& B(u(x, t), x, t)=\left\{\begin{array}{cl}
\sin (\pi x), & \text { if } t=1 \\
0, & \text { if } t=0 \\
0, & \text { if } x=0 \\
0, & \text { if } x=1
\end{array}\right.
\end{aligned}

\item Burgers equations (simple model of viscous fluid motions)
\begin{aligned}
& F(u(x, t), x, t)=\frac{\partial u}{\partial t}+u \frac{\partial u}{\partial x}-\nu \frac{\partial^2 u}{\partial x^2}\\
& B(u(x,t),x,t)=\text{periodic boundary conditions} \\
& I(u(x,t),x,t)=u(x,t) - \text{"analytic solution at t=0"}
\end{aligned}

\item Kuromoto-Sivashinsky equations (models the diffusive–thermal instabilities in a laminar flame front)
\begin{aligned}
& F(u(x, t), x, t)=\frac{\partial u}{\partial t}+u \frac{\partial u}{\partial t}+\frac{\partial^2 u}{\partial x^2}+\frac{\partial^4 u}{\partial x^4}-f(x, t) \\
& B(u(x,t),x,t)=\text{periodic boundary conditions} \\
& I(u(x,t),x,t)=u(x,0)-\cos \left(\frac{2 \pi x}{10}\right)-\operatorname{sech}\left(\frac{x-L / 2}{5}\right)
\end{aligned}

\item Navier-Stokes equations (general model of viscous fluid motions)
\begin{aligned}
& F_1(u(x, t), x, t)=(u \cdot \nabla) u+\nabla p-\frac{1}{\operatorname{Re}} \nabla^2 u \\
& F_2(u(x, t), x, t)=\nabla \cdot u
\end{aligned}

\end{itemize}

Optimal control PDE problem: 
Solve $\text{argmin}_{c\in \mathcal{C}} \mathcal{J}(u)$, where
1. [Volume] $\mathcal{F}(u(x, t), x, t; c_v(x,t))=0$ for all $x \in \Omega, t \in[0, T]$
2. [Boundary] $\mathcal{B}(u(x, t), x, t; c_b(x,t))=0$ for all $x \in$ boundary of $\Omega, \mathrm{t} \in[0, T]$
3. [Initial] $\mathcal{I}(u(x, 0), x; c_0(x))=0$ for all $x \in \Omega$

Examples:
\begin{itemize}
\item Laplace equations + control
$$
\mathcal{J}(u)=\int_0^1\left|\frac{\partial u}{\partial y}(x, 1)-q_d(x)\right|^2 d x, \quad q_d(x)=\cos (\pi x)
$$

\item Burgers equations + control
$$\mathcal{J}(u)=\frac{1}{2} \int_0^L\left|u(x, T)-u_a(x, T)\right|^2 d x $$

\item Kuromoto-Sivashinsky equations + control
$$\mathcal{J}(u, f)=\frac{1}{2} \int_0^T \int_0^L\left(|u(x, t)|^2+\sigma|f(x, t)|^2\right) d x d t$$

\item Navier-Stokes equations + control
$$\mathcal{J}(\mathbf{u})=\frac{1}{2} \int_0^{L_y}\left(\left|u\left(L_x, y\right)-u_{\text {parab }}(y)\right|^2+\left|v\left(L_x, y\right)\right|^2\right) d y, \quad u_{\text {parab }}(y)=\frac{4}{L_y^2} y(1-y)$$
\end{itemize}

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