import numpy as np

# Parametrii generatorului
H = np.array([100, 30.3, 35.8, 28.6, 26, 34.8, 26.4])
D = np.array([0.8, 0.85, 0.8, 0.8, 0.9, 0.7, 0.8])
tau_T = np.array([0.5, 0.4, 0.5, 0.5, 0.4, 0.5, 0.5])
r = np.array([1/20, 1/23, 1/19, 1/21, 1/21, 1/18, 1/20])

# Dimensiunea sistemului
n_generators = len(H)
state_dim = 4 * n_generators  # Dimensiunea stării totale

# Matricele A și B globale
A = np.zeros((state_dim, state_dim))
B = np.zeros((state_dim, n_generators))

# Matricele locale A_ii și B_ii
A_ii = np.zeros((n_generators, 4, 4))
B_ii = np.zeros((n_generators, 4, 1))
for i in range(n_generators):
    A_ii[i] = np.array([
        [0, 1, 0, 0],
        [0, -D[i] / H[i], 1 / H[i], 0],
        [0, 0, -1 / tau_T[i], 1 / tau_T[i]],
        [0, -1 / (r[i] * tau_T[i]), 0, -1 / tau_T[i]]
    ])
    B_ii[i] = np.array([
        [0],
        [0],
        [0],
        [1 / (r[i] * tau_T[i])]
    ])

# Inserarea blocurilor A_ii și B_ii în matricele globale
for i in range(n_generators):
    start_idx = i * 4
    A[start_idx:start_idx + 4, start_idx:start_idx + 4] = A_ii[i]
    B[start_idx:start_idx + 4, i:i + 1] = B_ii[i]

# Matricele de cuplare (pe baza susceptanțelor b_{ij})
b_ij = {
    (1, 3): 24.5, (1, 4): 24.5, (2, 3): 62.6, (3, 4): 40, 
    (3, 6): 28, (4, 5): 10, (4, 6): 10, (6, 7): 31.8
}

for (i, j), b in b_ij.items():
    i_idx, j_idx = i - 1, j - 1  # Indici zero-based
    if i_idx < n_generators and j_idx < n_generators:  # Verificare limite
        coupling = np.array([
            [0, 0, 0, 0],
            [b / H[i_idx], 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        # Actualizare blocuri pentru A[i, j] și A[j, i] (interconexiuni reciproce)
        start_i, start_j = i_idx * 4, j_idx * 4
        A[start_i:start_i + 4, start_j:start_j + 4] = coupling
        A[start_j:start_j + 4, start_i:start_i + 4] = coupling



# Matricele de cost Q și R
Q = np.kron(np.eye(n_generators), 100 * np.diag([5, 5, 0, 0]))

P = Q;
R = 0.1 *np.eye (n_generators)

print("Matricea globală A:\n", A)
print("Matricea globală B:\n", B)
print("Matricele de cost Q:\n", Q)
print("Matricele de cost R:\n", R)
