import customtkinter as ctk
from modules.util.ui import components
from modules.util.args.TrainArgs import TrainArgs
from modules.util.enum.LearningRateScheduler import LearningRateScheduler
from modules.util.lr_scheduler_util import *
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class SchedulerParamsWindow(ctk.CTkToplevel):
    def __init__(self, parent, ui_state, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.train_args = TrainArgs.default_values()
        self.ui_state = ui_state

        self.title("Optimizer Settings")
        self.geometry("1100x800")
        self.resizable(True, True)
        self.wait_visibility()
        self.grab_set()
        self.focus_set()
        
        self.MODEL_MAP = {
            'SD': ['unet', 'text_encoder'],
            'STABLE_DIFFUSION_XL_10_BASE': ['unet', 'text_encoder', 'text_encoder_2'],
        }   

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        self.filler_frame = ctk.CTkFrame(self, height=40)
        self.filler_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)

        self.frame = ctk.CTkFrame(self)
        self.frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.frame.grid_columnconfigure(0, weight=0)
        self.frame.grid_columnconfigure(1, weight=1)
        self.frame.grid_columnconfigure(2, minsize=20)
        self.frame.grid_columnconfigure(3, weight=0)
        self.frame.grid_columnconfigure(4, weight=1)
        self.frame.grid_columnconfigure(5, minsize=20)
        self.frame.grid_columnconfigure(6, weight=0)
        self.frame.grid_columnconfigure(7, weight=1)
               
        components.button(self.filler_frame, 0, 0, "ok", self.__ok)
        self.button = None
        self.main_frame(self.frame)  
        self.draw_graph()           
    
    def __ok(self):
        self.destroy()
        
    def draw_graph(self):
        model = self.ui_state.vars['model_type'].get()
        points_of_accuracy = 1000
        epochs = int(self.ui_state.vars['epochs'].get())
        subitems = self.MODEL_MAP[model]

        # Window and canvas sizing ✨
        window_width = self.winfo_width()
        window_height = self.winfo_height()
        canvas_width = int(window_width - 20)
        canvas_height = int(window_height - 500)
        advanced = self.ui_state.vars['schedulers_advanced'].get()

        fig, ax = plt.subplots()
        
        # Fabulous coloring 💄
        fig.patch.set_facecolor('#121212')
        ax.set_facecolor('#121212')
        ax.grid(color='white', linestyle='-', linewidth=0.5)
        ax.tick_params(axis='both', colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        
        all_y_values = []
        if advanced:
            # Go through each subitem, check if its train switch is enabled, and plot if so 🌈
            for subitem in subitems:
                train_switch = self.ui_state.vars[f'{subitem}_train_switch'].get()
                if train_switch:
                    y_values = self.plot_graph(ax, epochs, points_of_accuracy, subitem, self.ui_state)
                    all_y_values.extend(y_values)
        else:
            # In basic mode, we just plot using the "global" prefix 🌍
            y_values = self.plot_graph(ax, epochs, points_of_accuracy, "global", self.ui_state)
            all_y_values.extend(y_values)
            
        y_min = min(all_y_values)
        y_max = max(all_y_values)
        
        ax.set_yticks(np.linspace(y_min, y_max, 5))
        ax.set_yticklabels([f"{tick:.0e}" for tick in np.linspace(y_min, y_max, 5)])

        ax.legend()

        # Display it all! 🖼️
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.configure(width=canvas_width, height=canvas_height)
        self.canvas_widget.grid(row=1, column=0)
        self.canvas.draw()

    
    def plot_graph(self, ax, epochs, points_of_accuracy, prefix, ui_state):
        color_map = {
            'global': 'green',
            'unet': 'magenta',
            'text_encoder': 'cyan',
            'text_encoder_2': 'orange'
        }
        scheduler_name = ui_state.vars[f'{prefix}_learning_rate_scheduler'].get()
        num_cycles = int(ui_state.vars[f'{prefix}_num_cycles'].get())
        learning_rate = float(ui_state.vars[f'{prefix}_learning_rate'].get())  # Assuming this is a float
        min_learning_rate = float(ui_state.vars[f'{prefix}_min_learning_rate'].get())  # Assuming this is a float
        reverse = False

        scheduler = self.get_scheduler(scheduler_name, points_of_accuracy, num_cycles, min_learning_rate, reverse)
        x = np.linspace(0, points_of_accuracy-1, points_of_accuracy-1)
        y = [scheduler(int(step)) * learning_rate for step in x]
        x = (x / points_of_accuracy) * epochs

        ax.plot(x, y, label=f"{prefix.replace('text_encoder','te')}", color=color_map.get(prefix, 'black'))
        
        return y

        
    def get_scheduler(self, scheduler_name, points_of_accuracy, num_cycles, MINLR, reverse):
        schedulers = {
            'CONSTANT': lr_lambda_constant(),
            'COSINE': lr_lambda_cosine(points_of_accuracy, num_cycles, MINLR, reverse=False),
            'COSINE_WITH_HARD_RESTARTS': lr_lambda_cosine_with_hard_restarts(points_of_accuracy, num_cycles, MINLR, reverse=False),
            'REX': lr_rex(points_of_accuracy, num_cycles, MINLR, reverse=False),
            'TRIANGLE': lr_triangle(points_of_accuracy, num_cycles, MINLR, reverse=False),
            'RAMP': lr_ramp(points_of_accuracy, num_cycles, MINLR, reverse=False),
            'SAWTOOTH': lr_sawtooth(points_of_accuracy, num_cycles, MINLR, reverse=False),
            'SQUARE': lr_square(points_of_accuracy, num_cycles, MINLR, reverse=False),
            'PULSE': lr_pulse(points_of_accuracy, num_cycles, MINLR, reverse=False),
            'ROUNDED_PULSE': lr_rounded_pulse(points_of_accuracy, num_cycles, MINLR, reverse=False),
            'TRIANGLE_PULSE': lr_triangle_pulse(points_of_accuracy, num_cycles, MINLR, reverse=False),
            'RAMP_PULSE': lr_ramp_pulse(points_of_accuracy, num_cycles, MINLR, reverse=False),
            'SAWTOOTH_PULSE': lr_sawtooth_pulse(points_of_accuracy, num_cycles, MINLR, reverse=False),
            'SINE_CUBED': lr_sine_cubed(points_of_accuracy, num_cycles, MINLR, reverse=False),
            'FLAME': lr_flame(points_of_accuracy, num_cycles, MINLR, reverse=False),
            'SEMICIRCLE': lr_semicircle(points_of_accuracy, num_cycles, MINLR, reverse=False)
        }
        return schedulers.get(scheduler_name)
        
    def create_ui_component(self, master, row, col, subitem, arg_info, ui_state, LearningRateScheduler):
        thisArg = f"{subitem}_{arg_info['arg-suffix']}"
        title = ""
        if arg_info['title'] == 'Train Switch':
            title = f"Train {subitem.replace('_', ' ').title()}"
        else:
            title = f"{subitem.replace('_', ' ').title().replace('Text Encoder', 'TE')} {arg_info['title']}"

        tooltip = arg_info['tooltip']
        type = arg_info['type']
        components.label(master, row, col, title, tooltip=tooltip)

        if arg_info['arg-suffix'] == "learning_rate_scheduler":
            components.options(master, row, col+1, [str(x) for x in list(LearningRateScheduler)], ui_state, thisArg)
        elif type != 'boolean':
            components.entry(master, row, col+1, ui_state, thisArg)
        else:
            components.switch(master, row, col+1, ui_state, thisArg)


    def create_dynamic_ui(self, master, components, ui_state):
        advanced = self.ui_state.vars['schedulers_advanced'].get()
        model = self.ui_state.vars['model_type'].get()
        print(model)
        
        PART_MAP = {
            'unet': ['train_switch', 'scheduler', 'LR', 'minLR', 'number_of_cycles', 'warmup_steps', 'maximum_train_epochs'],
            'text_encoder': ['train_switch', 'scheduler', 'LR', 'minLR', 'number_of_cycles', 'warmup_steps', 'maximum_train_epochs'],
            'text_encoder_2': ['train_switch', 'scheduler', 'LR', 'minLR', 'number_of_cycles', 'warmup_steps', 'maximum_train_epochs'],
            'global': ['scheduler', 'LR', 'minLR', 'number_of_cycles', 'warmup_steps', 'maximum_train_epochs'],
        }
        
        KEY_DETAIL_MAP = {
            'train_switch': {
                'title': 'Train Switch',
                'tooltip': 'Enable or disable training for this component',
                'type': 'boolean',
                'arg-suffix': 'train_switch',
            },
            'LR': {
                'title': 'Learning Rate',
                'tooltip': 'Initial learning rate for the optimizer',
                'type': 'float',
                'arg-suffix': 'learning_rate',
            },
            'minLR': {
                'title': 'Minimum Learning Rate %',
                'tooltip': 'The lowest learning rate to use during training. This is a percentage of LR',
                'type': 'float',
                'arg-suffix': 'min_learning_rate',
            },
            'number_of_cycles': {
                'title': 'Number of Cycles',
                'tooltip': 'The number of training cycles',
                'type': 'integer',
                'arg-suffix': 'num_cycles',
            },
            'warmup_steps': {
                'title': 'Warmup Steps',
                'tooltip': 'Number of warmup steps before training',
                'type': 'integer',
                'arg-suffix': 'warmup_steps',
            },
            'maximum_train_epochs': {
                'title': 'Max Training Epochs',
                'tooltip': 'The maximum number of training epochs',
                'type': 'integer',
                'arg-suffix': 'max_train_epochs',
            },
            'scheduler': {
                'title': 'Scheduler',
                'tooltip': 'Type of learning rate scheduler',
                'type': 'string',
                'arg-suffix': 'learning_rate_scheduler',
            },
        }



        if advanced:
            subitems = self.MODEL_MAP[model]  
            col = 0 
            for subitem in subitems:
                parts = PART_MAP[subitem]
                row = 1 
                for part in parts:
                    arg_info = KEY_DETAIL_MAP[part]
                    self.create_ui_component(master, row, col, subitem, arg_info, ui_state, LearningRateScheduler)
                    row += 1
                col += 3 
        else:
            col, row = 0, 1
            subitems = self.MODEL_MAP[model]  
            for subitem in subitems:
                parts = PART_MAP[subitem]
                part = PART_MAP[subitem][0]
                arg_info = KEY_DETAIL_MAP[part]
                self.create_ui_component(master, row, col, subitem, arg_info, ui_state, LearningRateScheduler)
                row += 1
            row = 1
            col = 3
            if "global" in PART_MAP:
                subitem = "global"
                parts = PART_MAP[subitem]
                for part in parts:
                    arg_info = KEY_DETAIL_MAP[part]
                    self.create_ui_component(master, row, col, subitem, arg_info, ui_state, LearningRateScheduler)
                    row += 1

        
    def main_frame(self, master):
  
        # Advance Switch
        components.label(master, 0, 0, "Advanced Mode", tooltip="The type of optimizer")
        components.switch(master, 0, 1, self.ui_state, "schedulers_advanced")

        
        self.ui_state.vars['schedulers_advanced'].trace_add('write', self.on_change)
        self.create_dynamic_ui(master, components, self.ui_state)
        
    def on_change(self, *args):
        self.clear_dynamic_ui(self.frame)
        self.create_dynamic_ui(self.frame, components, self.ui_state)
            
    def clear_dynamic_ui(self, master):
        try:
            for widget in master.winfo_children():
                grid_info = widget.grid_info()
                if int(grid_info["row"]) >= 1:
                    widget.destroy()
        except _tkinter.TclError as e:
            pass
            
