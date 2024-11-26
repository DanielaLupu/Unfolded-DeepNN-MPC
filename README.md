# Unfolded-DeepNN-MPC
This repo contains a unfolded deep neural network methods for solving the linear MPC problem.

## MPC problem formulation

Let us consider  the  following discrete-time linear system:

$$x_{t+1} = Ax_t + B u_t,$$

where $x_t \in \mathbb{R}^{n_x}$ denotes the state at time $t$ and $u_t \in \mathbb{R}^{n_u}$ is the associated control input.

We consider input and state constraints: $u_t \in \mathbb{U}$  and $x_t \in \mathbb{X}$ for all $t\geq 0$, where $\mathbb{U} \subseteq  \mathbb{R}^{n_u}$ and $\mathbb{X} \subseteq  \mathbb{R}^{n_x}$ are given  polyhedral sets. More precisely, we consider box type  constraints on the states and inputs, i.e.:
$$\mathbb{U} =[\underline{u}, \overline{u}] \quad  \text{and} \quad   \mathbb{X}=[\underline{x}, \overline{x}].$$  

For the aforementioned system, we assume convex quadratic stage and final costs of the form:

$$\ell(x, u)= 1/2 \left( x^{\top} Q x+u^{\top} R u \right), \quad \ell_f (x)=1/2 x^{\top} P x,$$

respectively. Thus, the matrices $P, Q \in \mathbb{R}^{n_x \times n_x}$  and $R \in \mathbb{R}^{n_u \times n_u}$ are positive (semi-)definite. Further, for a given initial state $x_0$, we formulate the model predictive control (MPC) problem as follows:

$$ \min_{x_t, u_t} \sum_{t=0}^{N-1} \ell\left(x_t, u_t\right) + \ell_f\left(x_N\right)$$
$$\text{s.t.:}  \quad x_{t+1}=A x_t+B u_t,    u_t \in \mathbb{U}  \forall t=0:N-1, \;\; x_0 \; \text{given}$$
$$  x_t \in \mathbb{X}\;\;  \forall t=1:N,$$
