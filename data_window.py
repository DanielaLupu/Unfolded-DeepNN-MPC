import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import os
from methods.dense_mpc import dense_mpc  # Assuming dense_mpc is in the methods folder
from opt_methods import pd, fom_agpd  # Assuming your optimization functions are in opt_methods.py




class DatasubWindow:
    def __init__(self, root, name):
        self.root = root
        self.root.title(name)
        self.root.geometry("400x400")
        self.state_checked = tk.BooleanVar(value=False)

        # Titlu
        ttk.Label(self.root, text="Problem formulation", font=("Helvetica", 12)).pack(pady=10)

        # Buton pentru introducerea a două matrice
        ttk.Button(self.root, text="System Dynamics: A, B", command=self.input_two_matrices).pack(pady=10)

        # Buton pentru introducerea a trei matrice
        ttk.Button(self.root, text="Cost matrices Q, P, R", command=self.input_three_matrices).pack(pady=10)

        # Checkbox pentru opțiunea de stare
        ttk.Checkbutton(
            self.root, text="Check if state is constrained",
            variable=self.state_checked,
            command=self.on_state_checked
        ).pack(pady=10)

        # Buton pentru introducerea vectorilor
        ttk.Button(self.root, text="Define bounds", command=self.input_vectors).pack(pady=10)

        # Generare de date
        ttk.Button(self.root, text="Generate Data", command=self.generate_data).pack(pady=10)

    def input_two_matrices(self):
        """Deschide o fereastră pentru introducerea a două matrice."""
        self.open_matrix_input_window(2)

    def input_three_matrices(self):
        """Deschide o fereastră pentru introducerea a trei matrice."""
        self.open_matrix_input_window(3)

    def input_vectors(self):
        """Deschide o fereastră pentru introducerea vectorilor."""
        vec_window = tk.Toplevel(self.root)
        vec_window.title("Input Vectors")
        vec_window.geometry("500x500")

        # Vectors u_lb and u_ub
        ttk.Label(vec_window, text="Enter control law lower bound (comma-separated):").pack(pady=5)
        entry_ulb = ttk.Entry(vec_window)
        entry_ulb.pack()

        ttk.Label(vec_window, text="Enter control law upper bound (comma-separated):").pack(pady=5)
        entry_uub = ttk.Entry(vec_window)
        entry_uub.pack()

        # Vectors z_lb and z_ub (only if primal-dual is selected)
        if self.state_checked.get():
            ttk.Label(vec_window, text="Enter state lower bound (comma-separated):").pack(pady=5)
            entry_zlb = ttk.Entry(vec_window)
            entry_zlb.pack()

            ttk.Label(vec_window, text="Enter state upper bound (comma-separated):").pack(pady=5)
            entry_zub = ttk.Entry(vec_window)
            entry_zub.pack()
        else:
            entry_zlb, entry_zub = None, None

        def submit_vectors():
            global u_ub, u_lb, z_ub, z_lb, state_vectors
            try:
                # Read and parse the u_lb and u_ub vectors
                u_lb = np.array([float(x) if x not in ['inf', '-inf'] else float(x.replace('inf', 'inf')) 
                                 for x in entry_ulb.get().split(',')])
                u_ub = np.array([float(x) if x not in ['inf', '-inf'] else float(x.replace('inf', 'inf')) 
                                 for x in entry_uub.get().split(',')])

                if B is not None:
                    if u_lb.size != B.shape[1] or u_ub.size != B.shape[1]:
                        messagebox.showerror("Error", "control law bounds must match the number of columns in Matrix B.")
                        return

                # Read and parse the z_lb and z_ub vectors if primal-dual is checked
                if self.state_checked.get():
                    z_lb = np.array([float(x) if x not in ['inf', '-inf'] else float(x.replace('inf', 'inf')) 
                                     for x in entry_zlb.get().split(',')])
                    z_ub = np.array([float(x) if x not in ['inf', '-inf'] else float(x.replace('inf', 'inf')) 
                                     for x in entry_zub.get().split(',')])

                    if A is not None:
                        if z_lb.size != A.shape[0] or z_ub.size != A.shape[0]:
                            messagebox.showerror("Error", "State bounds must match the number of rows in Matrix A.")
                            return

                    state_vectors = {"z_lb": z_lb, "z_ub": z_ub}

                else:
                    state_vectors = None

                messagebox.showinfo("Success", "Vectors saved successfully!")
                vec_window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Invalid input. Please ensure numerical values are entered. 'inf' is accepted for infinity.")

        ttk.Button(vec_window, text="Submit", command=submit_vectors).pack(pady=20)

    def open_matrix_input_window(self, matrix_count):
        """Deschide o fereastră pentru introducerea matricilor."""
        matrix_window = tk.Toplevel(self.root)
        matrix_window.title(f"Input Matrices")
        matrix_window.geometry("400x400")
        matrix_list = ("A", "B", "Q", "P", "R")
        entries = []
        
        if matrix_count == 2:
            for i in range(matrix_count):
                ttk.Label(matrix_window, text=f"Enter Matrix {matrix_list[i]} (rows separated by ';', values by ','):").pack(pady=5)
                entry = ttk.Entry(matrix_window)
                entry.pack(pady=5)
                entries.append(entry)
        else:
            for i in range(matrix_count):
                ttk.Label(matrix_window, text=f"Enter Matrix {matrix_list[i+2]} (rows separated by ';', values by ','):").pack(pady=5)
                entry = ttk.Entry(matrix_window)
                entry.pack(pady=5)
                entries.append(entry)

        def submit_matrices():
            global A, B, Q, P, R
            try:
                matrices = []
                for entry in entries:
                    matrix = np.array([list(map(float, row.split(','))) for row in entry.get().split(';')])
                    matrices.append(matrix)

                if matrix_count == 2:
                    A, B = matrices[0], matrices[1]
                    if A.shape[0] != A.shape[1]:
                        messagebox.showerror("Error", "Matrix A must be square.")
                        return
                    if A.shape[0] != B.shape[0]:
                        messagebox.showerror("Error", "Matrix B must have the same number of rows as Matrix A.")
                        return
                elif matrix_count == 3:
                    Q, P, R = matrices[0], matrices[1], matrices[2]
                    if A is not None and (Q.shape != A.shape or P.shape != A.shape):
                        messagebox.showerror("Error", "Matrices Q and P must have the same dimensions as Matrix A.")
                        return
                    if R.shape[0] != R.shape[1] or (B is not None and R.shape[0] != B.shape[1]):
                        messagebox.showerror("Error", "Matrix R must be square and match the number of columns of Matrix B.")
                        return

                messagebox.showinfo("Success", "Matrices saved successfully!")
                matrix_window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Invalid input. Please ensure the format is correct.")

        ttk.Button(matrix_window, text="Submit", command=submit_matrices).pack(pady=20)
    
    def on_state_checked(self):
        """Eveniment declanșat când checkbox-ul de stare este activat."""
        if self.state_checked.get():
            messagebox.showinfo("Info", "Primal-Dual Formulation Selected")
            
    def generate_data(self):
        """Funcție pentru generarea datelor în funcție de opțiuni."""
        global horizon_length, state_vectors

        # Deschide o fereastră pentru introducerea parametrilor
        input_window = tk.Toplevel(self.root)
        input_window.title("Generate Data Parameters")
        input_window.geometry("300x200")

        ttk.Label(input_window, text="Enter the number of data points:").pack(pady=5)
        entry_data_points = ttk.Entry(input_window)
        entry_data_points.pack(pady=5)
        entry_data_points.insert(0, "500")  # Valoare implicită

        ttk.Label(input_window, text="Enter the horizon length (N):").pack(pady=5)
        entry_horizon_length = ttk.Entry(input_window)
        entry_horizon_length.pack(pady=5)
        entry_horizon_length.insert(0, str(horizon_length))  # Valoare implicită

        def submit_data_generation():
            try:
                # Validare: toate câmpurile trebuie completate
                if not entry_data_points.get() or not entry_horizon_length.get():
                    messagebox.showerror("Error", "Please complete all fields before generating data.")
                    return

                # Preluarea valorilor introduse de utilizator
                num_points = int(entry_data_points.get())
                horizon_length = int(entry_horizon_length.get())

                # Validare: toate matricile și limitele trebuie definite
                if state_vectors is None or A is None or B is None or Q is None or P is None or R is None:
                    messagebox.showerror("Error", "Please ensure all matrices and bounds are defined.")
                    return

                # Generare de stări inițiale (z0) între limitele specificate
                z0 = np.random.uniform(
                    low=state_vectors['z_lb'], 
                    high=state_vectors['z_ub'], 
                    size=(num_points, A.shape[0])
                    )

                # Listă pentru a stoca legile de control generate
                all_u = []

                # Calculul legilor de control pentru fiecare stare inițială
                for z_init in z0:
                    if self.state_checked.get():
                        u = fom_agpd(A, B, Q, P, R, z_init, horizon_length)  # Exemplu pentru fom_agpd
                    else:
                        u = pd(A, B, Q, P, R, z_init, horizon_length)  # Exemplu pentru pd

                    all_u.append(u)

                # Salvare date în folderul "data"
                output_folder = "data"
                os.makedirs(output_folder, exist_ok=True)

                # Creare structură pentru salvare
                all_data = {
                    "Initial States (z0)": z0.tolist(),
                    "Control Laws (u)": all_u
                    }

                # Salvare ca CSV
                df = pd.DataFrame({
                    "z0": [list(z) for z in z0],
                    "u": [list(ui) for ui in all_u]
                    })
                df.to_csv(os.path.join(output_folder, "generated_data.csv"), index=False)

                messagebox.showinfo("Success", f"Data generated and saved to {output_folder}/generated_data.csv")
                input_window.destroy()

            except ValueError:
                messagebox.showerror("Error", "Please enter valid integer values for data points and horizon length.")
            except Exception as e:
                messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

        ttk.Button(input_window, text="Generate", command=submit_data_generation).pack(pady=20)