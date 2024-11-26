import numpy as np

def fom_agpd(H, q, lb, ub, alpha=0.05, epsilon=1e-8, beta=1e-3, iterMax =500):
    """
    Accelerated Gradient Projection Descent (AGPD) method.

    Solves the problem:
        min_u 1/2 u^T H u + q^T u
        subject to u_lb <= u <= u_ub

    Inputs:
        H       - Quadratic term in the objective function
        q       - Linear term in the objective function
        lb      - Lower bound for u
        ub      - Upper bound for u
        alpha   - Step size (default: 0.05)
        epsilon - Convergence tolerance (default: 1e-8)
        beta    - Acceleration parameter (default: 1e-3)

    Outputs:
        u_c     - Solution vector
    """

    # Initialization
    n = H.shape[0]
    u_c = np.random.randn(n, 1)  # Random initial guess
    y = u_c.copy()
    crit = 1  # Convergence criterion
    iter =0
    while crit > epsilon and iter <= iterMax:
        # Gradient descent step
        u_new = (np.eye(n) - alpha * H) @ y - alpha * q

        # Project onto the feasible set [lb, ub]
        u_new = np.minimum(ub, np.maximum(u_new, lb))

        # Update the extrapolation
        y = (1 + beta) * u_new - beta * u_c

        # Check convergence
        crit = np.linalg.norm(u_new - u_c)

        # Update the current solution
        u_c = u_new
        iter +=1
    return u_c


def pd(lbd, gamma1, H, q, dx0, C, u_ub, u_lb, gamma2= 0, iterMax =500):
    """
    Condat-Vu and PD30 algorithm for MPC optimization.

    Solves:
        min_u 1/2 u^T H u + q^T u
        subject to C*u <= dx0, u_lb <= u <= u_ub

    Parameters:
        lbd      - Step size for dual variable update
        gamma1   - Primal step size
        gamma2   - Scaling factor for H in dual update
        iterMax  - Maximum number of iterations
        H        - Quadratic term in the objective
        q        - Linear term in the objective
        dx0      - Inequality constraint bounds
        C        - Constraint matrix
        u_ub     - Upper bounds for u
        u_lb     - Lower bounds for u

    Returns:
        uNew     - Optimized solution
        iter     - Number of iterations performed
    """
    epsilon = 1e-4  # Convergence tolerance
    cond = 1  # Convergence criterion
    iter = 0  # Iteration counter

    # Initialization
    miuCrnt = np.random.rand(C.shape[0], 1)  # Initialize dual variable
    uPrev = np.random.rand(H.shape[0], 1)    # Initialize previous primal variable
    uCrnt = np.random.rand(H.shape[0], 1)    # Initialize current primal variable

    while cond >= epsilon and iter <= iterMax:
        # Dual variable update (miu)
        miuNew = miuCrnt + (lbd / gamma1) * (C @ (2 * uCrnt - uPrev - gamma2 * H @ (uCrnt - uPrev)) - dx0)
        miuNew = np.maximum(miuNew, 0)  # Enforce non-negativity

        # Primal variable update (u)
        uNew = uCrnt - gamma1 * (H @ uCrnt + q + C.T @ miuNew)
        uNew = np.minimum(u_ub, np.maximum(uNew, u_lb))  # Enforce box constraints

        # Convergence check
        cond = np.linalg.norm(uNew - uCrnt)

        # Update variables for next iteration
        uPrev = uCrnt
        uCrnt = uNew
        miuCrnt = miuNew
        iter += 1

    return uNew

