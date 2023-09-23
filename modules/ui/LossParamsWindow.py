import customtkinter as ctk
from modules.util.ui import components
from modules.util.ui.UIState import UIState
from modules.util.args.TrainArgs import TrainArgs

class LossParamsWindow(ctk.CTkToplevel):
    def __init__(self, parent, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.train_args = TrainArgs.default_values()
        self.ui_state = UIState(self, self.train_args)
       
        self.title("Loss Function Parameters")
        self.geometry("800x400")
        self.resizable(True, True)
        self.wait_visibility()
        self.grab_set()
        self.focus_set()

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)

        self.frame = ctk.CTkFrame(self, width=600, height=300)
        self.frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        components.button(self, 1, 0, "ok", self.__ok)
        
    def __ok(self):
        self.destroy()
