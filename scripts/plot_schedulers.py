import sys
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from modules.util.lr_scheduler_util import *
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('dark_background')

schedulers = {
    'Constant': lr_lambda_constant(),
    'Cosine_Down': lr_lambda_cosine_down(100, 1),
    'Cosine_Up': lr_lambda_cosine_up(100, 1),
    'Cosine_with_Hard_Restarts': lr_lambda_cosine_with_hard_restarts(100, 1),
    'Rex': lr_rex(100, 1),
    'Triangle': lr_triangle(100, 1),
    'Ramp': lr_ramp(100, 1),
    'Sawtooth': lr_sawtooth(100, 1),
    'Square': lr_square(100, 1),
    'Pulse': lr_pulse(100, 1),
    'Rounded_Pulse': lr_rounded_pulse(100, 1),
    'Triangle_Pulse': lr_triangle_pulse(100, 1),
    'Ramp_Pulse': lr_ramp_pulse(100, 1),
    'Sawtooth_Pulse': lr_sawtooth_pulse(100, 1),
    'Sine_Cubed': lr_sine_cubed(100, 1),
    'Flame': lr_flame(100, 1),
    'Semicircle': lr_semicircle(100, 1)
}


# Generate 4 repeats for the graph
scheduler_steps = 100
repeats = 3

# Loop through the schedulers
for name, scheduler in schedulers.items():
    plt.figure(figsize=(10, 3)) 
    x = np.linspace(0, scheduler_steps * repeats, scheduler_steps * repeats)
    y = [scheduler(int(step)) for step in x]

    # Plot the learning rate
    plt.plot(x, y, label=f"{name}", color='cyan')
    plt.title(f"{name} Learning Rate Scheduler", color='white')
    plt.xlabel("Steps", color='white')
    plt.ylabel("Learning Rate", color='white')
    plt.legend()
    
    # Save as PNG
    plt.savefig(f"{name}_LearningRate.png", facecolor='black')
    plt.close()

    print(f"Generated {name}_LearningRate.png 🌌💫")

print("And just like that, the spell is complete! 🌟")