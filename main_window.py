import os, sys
import tkinter as tk
from tkinter import ttk

from PIL import ImageTk, Image

sys.path.append(os.path.abspath("methods"))

from data_window import DatasubWindow
from train_window import TrainWindow
from result_window import ShowWindow


# Variabile globale pentru stocarea datelor
A = None
B = None
Q = None
P = None
R = None
u_ub = None
u_lb = None
z_ub = None
z_lb = None
state_vectors = None
horizon_length = 5  # Default horizon length
model= None


class MainWindow:
    def __init__(self, root, name):
        self.root = root
        self.root.title(name)
        
        # Set theme
        self.__set_theme()
        
        # Set geometry
        self.root.geometry("350x300")
        #self.root.iconbitmap(os.path.join("Resources", 'elo-hyp_logo.ico'))  # Commented out

        # Load images
        #self.__load_images()
        
        # Initialize GUI components
        self.__create_widgets()
        
        # Start main loop
        self.root.mainloop()

    def __set_theme(self):
        """Set the theme for the GUI"""
        self.root.tk.call("source", os.path.join("Resources", "UI", "sun-valley.tcl"))
        self.root.tk.call("set_theme", "light")

  
    def __create_widgets(self):
        """Create and place widgets in the main window"""
        # Display main logo (commented out)
        # tk.Label(self.root, image=self.logo_image).pack(pady=(10, 5))
        
        # Generate Data Button
        ttk.Button(self.root, text="Generate Data", command=self.__get_data_window, width=30).pack(pady=(10, 5))
        
        # Train Network Button
        ttk.Button(self.root, text="Train Network", command=self.__get_train_net, width=30).pack(pady=5)
        
        # Classification Buttons
        ttk.Button(self.root, text="Show results", command=self.__get_show_result, width=30).pack(pady=5)

        # Information label
        ttk.Label(
            self.root,
            text="\n***The research leading to this application has received \nfunding from the UEFISCDI, "
                 "PN-III-P4-PCE-2021-0720, under \nproject L2O-MOC, nr. 70/2022."
        ).pack(pady=(10, 10))

    def __get_data_window(self):
        """Open the data window"""
        self.data_window = DatasubWindow(tk.Toplevel(), "Input Data")
        
    def __get_train_net(self):
        """Open the train window"""
        self.train_window = TrainWindow(tk.Toplevel(), "Train Network", u_ub, u_lb, z_ub, z_lb, state_vectors, horizon_length)
    
    def __get_show_result(self):
        """Open the train window"""
        self.result_window = ShowWindow(tk.Toplevel(), "Show results", model, A, B)
    
    

    

