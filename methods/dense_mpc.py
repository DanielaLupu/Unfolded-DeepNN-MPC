import numpy as np

def dense_mpc(A, B, Q, R, P, z0, N, u_ub, u_lb, z_ub, z_lb):
    """
    Dense QP formulation for MPC.
    
    Inputs:
        A, B    - System dynamics matrices
        Q, R    - Cost function matrices
        P       - Terminal cost matrix
        z0      - Initial state
        N       - Horizon length
        u_ub, u_lb - Input bounds
        z_ub, z_lb - State bounds
    
    Outputs:
        H, q    - Quadratic objective function terms
        Czz, dzz - State inequality constraints
        C, d    - Combined inequality constraints
    """
    # Dimensions
    nz, nu = B.shape  # Size of state and input variables

    # Create the cost matrices
    QQ = np.kron(np.eye(N - 1), Q)
    QQ = np.block([[QQ, np.zeros((QQ.shape[0], P.shape[1]))],
                   [np.zeros((P.shape[0], QQ.shape[1])), P]])
    RR = np.kron(np.eye(N), R)

    # Initialize bb and AA matrices
    bb = np.zeros((N * nz, nz))
    bb[:nz, :] = A
    AA = np.kron(np.eye(N), B)

    # Formulate the equality constraint: Ax = b
    for i in range(1, N):
        bb[i * nz:(i + 1) * nz, :] = np.linalg.matrix_power(A, i + 1)
        AA += np.kron(np.diag(np.ones(N - i), -i), 
                      bb[(i - 1) * nz:i * nz, :] @ B)

    bb = bb @ z0

    # Formulate the inequality constraint: Cx <= d
    #Cu = np.kron(np.eye(N), np.vstack((np.eye(nu), -np.eye(nu))))
    #du = np.kron(np.ones(N), np.hstack((u_ub, -u_lb)))

    Cz = np.kron(np.eye(N), np.vstack((np.eye(nz), -np.eye(nz))))
    dz = np.kron(np.ones(N), np.hstack((z_ub, -z_lb)))

    Czz = Cz @ AA
    dzz = dz - Cz @ bb

    #C = np.vstack((Cu, Czz))
    #d = np.hstack((du, dzz))

    # Calculate H and q
    H = AA.T @ QQ @ AA + RR
    q = AA.T @ QQ @ bb

    return H, q, Czz, dzz
