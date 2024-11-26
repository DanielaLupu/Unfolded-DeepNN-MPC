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
$$\text{s.t.:}  \quad x_{t+1}=A x_t+B u_t,$$
$$               u_t \in \mathbb{U}  \forall t=0:N-1,  x_0 \text{given}$$
$$               x_t \in \mathbb{X}  \forall t=1:N,$$

where we denoted the length of the prediction horizon with $N$.
This problem can be reformulated as a quadratic programming (QP) problem. See in Bibliography paper 1 or 2 for the formulation.

## Guide to run the application
Run the code

    python app.py

Requirements to run the application:

    numpy
    scipy
    PIL
    matplotlib
    tkinter

## Steps to use the application as exe
This application, generates data for training the unfolded deep network. Once the network is trained, one can generate fast MPC laws with the Neural Networks.

<img width="280" alt="image" src="https://github.com/user-attachments/assets/f86c237a-9685-4f98-85e5-c450cbbebff5">

Step 1: Generate data for training the network. See MPC formulation section for the inputs the user needs to provide.
If there are no constrains on the state, the QP problem is solved by using first order optimization algorithm, i.e the accelerated gradient projection algorithm (see paper 1 in Bibliography). If the user checks the checkbox for state constrains we employ a primal dual optimization algorithm , named PD3O (see paper 2 in Bibliography).

<img width="295" alt="image" src="https://github.com/user-attachments/assets/c342e819-6e77-4bdb-8806-b6433ef4a678">

Step 2: Train the network. The Neural Network architecture is inspired by the optimization algorithm used in step 1.

Step 3: Give a initial state of the system and a simulation horizont. A figure with the inputs and states bahaviour will be produced.

We provide for testing the CIGRE system (see paper 3 in Bibliography for the model).

## Bibliography
1. D. Lupu, I. Necoara, "Exact representation and efficient approximations of linear model predictive control laws via HardTanh type deep neural networks", Systems and Control Letters (Q1), 186, doi: 10.1016/j.sysconle.2024.105742, 2024.
2. D. Lupu, I. Necoara, L. Toma, "Deep unfolding primal-dual architectures: application to linear model predictive control", submitted to European Control Conference, Greece, 2025.
3. R. M. Hermans, M. Lazar and A. Jokić, "Distributed predictive control of the 7-machine CIGRÉ power system," Proceedings of the 2011 American Control Conference, San Francisco, CA, USA, 2011, pp. 5225-5230, doi: 10.1109/ACC.2011.5991135.
   
## Authors
Daniela Lupu & Ion Necoara


