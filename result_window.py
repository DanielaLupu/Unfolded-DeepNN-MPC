import numpy as np
import torch
from tkinter import messagebox
from tkinter import ttk

class ShowWindow:
    def __init__(self, root, name, model, A, B):
        
        self.root = root
        self.model = model
        self.A, self.B = A, B

        self.root.geometry("300x300")

        # Label for setting the simulation time
        ttk.Label(self.root, text="Set the simulation time (Tsim):").pack(pady=10)

        # Entry for simulation time
        self.Tsim_entry = ttk.Entry(self.root)
        self.Tsim_entry.pack(pady=5)

        # Label for setting the initial state vector
        ttk.Label(self.root, text="Set the initial state (z_init):").pack(pady=10)

        # Entry for initial state vector
        self.z_init_entry = ttk.Entry(self.root)
        self.z_init_entry.pack(pady=5)

        # Button to start simulation
        start_simulation_button = ttk.Button(self.root, text="Start Simulation", command=self.start_simulation)
        start_simulation_button.pack(pady=20)

    def start_simulation(self):
        """Starts the simulation process using the initial state and simulation time."""
        try:
            # Get simulation time (Tsim) and initial state vector (z_init)
            Tsim = int(self.Tsim_entry.get())  # Simulation time
            z_init = np.fromstring(self.z_init_entry.get()[1:-1], sep=',')  # Parse state vector from string
            z_init = z_init.reshape(-1, 1)  # Reshape into column vector

            # Convert to the appropriate data type
            u_dim = self.B.shape[1]
            z_init_nn = np.double(z_init)
            u_nn = np.empty((u_dim, 1))  # Initialize control input storage
            z_nn = z_init_nn  # Initialize state storage

            self.model.eval()  # Set model to evaluation mode

            # Simulation loop
            for i in range(Tsim):
                # Forward pass through the model to get control input (cmd_nn)
                cmd_nn = self.model(torch.tensor(z_init_nn.T))  # Forward pass using model
                cmd_nn = cmd_nn.detach().numpy()  # Convert to numpy array

                # Update state using system dynamics (A and B matrices must be predefined)
                z_init_nn = np.dot(self.A, z_init_nn) + np.dot(self.B, cmd_nn[:u_dim,])  # Update state

                # Concatenate the results to store state and control input over time
                u_nn = np.concatenate((u_nn, cmd_nn[:u_dim,]), axis=1)
                z_nn = np.concatenate((z_nn, z_init_nn), axis=1)

            # After the simulation, you can visualize or save the results (e.g., u_nn and z_nn)
            print("Simulation complete.")
            print("State trajectory (z_nn):")
            print(z_nn)
            print("Control input trajectory (u_nn):")
            print(u_nn)

            # Optionally visualize or save the results (e.g., using matplotlib)
            # plot_state_trajectory(z_nn)
            # plot_control_input_trajectory(u_nn)

            messagebox.showinfo("Simulation Complete", "Simulation completed successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during simulation: {str(e)}")
