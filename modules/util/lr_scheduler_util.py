import math
from typing import Callable
import numpy as np


# Warmup Steps
def lr_lambda_warmup(warmup_steps: int, lr_lambda: Callable[[int], float]):
    def warmup(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(warmup_steps)
        else:
            return lr_lambda(current_step - warmup_steps)

    return warmup

# Constant
def lr_lambda_constant():
    def lr_lambda(current_step: int):
        return 1

    return lr_lambda

# Cosine (Starting at max)
def lr_lambda_cosine_down(scheduler_steps: int, num_cycles: float, min_lr=0.1):
    def lr_lambda(current_step: int):
            progress = float(current_step) / float(scheduler_steps)
            schedule = math.cos(progress * 2.0 * math.pi * num_cycles)
            return min_lr + (1.0 - min_lr) * 0.5 * (1.0 + schedule)
    return lr_lambda

# Cosine (Starting at min)
def lr_lambda_cosine_up(scheduler_steps: int, num_cycles: float, min_lr=0.5):
    def lr_lambda(current_step: int):
        progress = float(current_step) / float(scheduler_steps)
        schedule = math.cos(progress * 2.0 * math.pi * num_cycles)
        return min_lr + (1.0 - min_lr) * 0.5 * (1.0 - schedule)
    return lr_lambda

# Cosine with Hard Restarts
def lr_lambda_cosine_with_hard_restarts(scheduler_steps: int, num_cycles: float):
    def lr_lambda(current_step: int):
        progress = float(current_step) / float(scheduler_steps)
        schedule = math.cos(((progress * num_cycles) % 1.0) * math.pi)
        return max(0.0, 0.5 * (1.0 + schedule))
    return lr_lambda

# Rex
def lr_rex(scheduler_steps: int, num_cycles: float, min_lr=0.1):
    steps_per_cycle = scheduler_steps / num_cycles   
    def lr_lambda(current_step: int):
        max_lr = 1
        d = 0.9
        cycle_idx = math.floor(current_step / steps_per_cycle)
        cycle_step = current_step - (cycle_idx * steps_per_cycle)
        progress = cycle_step / steps_per_cycle
        div = (1 - d) + (d * (1 - progress))
        return min_lr + (max_lr - min_lr) * ((1 - progress) / div)
    return lr_lambda
    
# Triangle
def lr_triangle(scheduler_steps, num_cycles, min_lr=0.1):
    def lr(current_step):
        progress = (current_step / scheduler_steps) % (1 / num_cycles)
        return min_lr + (1 - min_lr) * abs(2 * progress - 1)
    return lr

# Ramp
def lr_ramp(scheduler_steps, num_cycles, min_lr=0.1):
    def lr(current_step):
        progress = (current_step / scheduler_steps) % (1 / num_cycles)
        return min_lr + (1 - min_lr) * progress
    return lr

# Sawtooth
def lr_sawtooth(scheduler_steps, num_cycles, min_lr=0.1):
    def lr(current_step):
        progress = (current_step / scheduler_steps) % (1 / num_cycles)
        return min_lr + (1 - min_lr) * (1 - progress)
    return lr

# Square
def lr_square(scheduler_steps, num_cycles, min_lr=0.1):
    def lr(current_step):
        progress = (current_step / scheduler_steps) % (1 / num_cycles)
        return min_lr if progress < 0.5 else 1
    return lr

# Pulse
def lr_pulse(scheduler_steps, num_cycles, min_lr=0.1):
    def lr(current_step):
        progress = (current_step / scheduler_steps) % (1 / num_cycles)
        return min_lr if progress < 0.1 or progress > 0.9 else 1
    return lr

# Rounded Pulse
def lr_rounded_pulse(scheduler_steps, num_cycles, min_lr=0.1):
    def lr(current_step):
        progress = (current_step / scheduler_steps) % (1 / num_cycles)
        return min_lr + (1 - min_lr) * np.sin(progress * np.pi)**2
    return lr

# Triangle Pulse
def lr_triangle_pulse(scheduler_steps, num_cycles, min_lr=0.1):
    def lr(current_step):
        progress = (current_step / scheduler_steps) % (1 / num_cycles)
        return min_lr if progress < 0.1 or progress > 0.9 else abs(2 * progress - 1)
    return lr

# Ramp Pulse
def lr_ramp_pulse(scheduler_steps, num_cycles, min_lr=0.1):
    def lr(current_step):
        progress = (current_step / scheduler_steps) % (1 / num_cycles)
        return min_lr if progress < 0.1 or progress > 0.9 else progress
    return lr

# Sawtooth Pulse
def lr_sawtooth_pulse(scheduler_steps, num_cycles, min_lr=0.1):
    def lr(current_step):
        progress = (current_step / scheduler_steps) % (1 / num_cycles)
        return min_lr if progress < 0.1 or progress > 0.9 else 1 - progress
    return lr

# Sine Cubed
def lr_sine_cubed(scheduler_steps, num_cycles, min_lr=0.1):
    def lr(current_step):
        progress = (current_step / scheduler_steps) % (1 / num_cycles)
        return min_lr + (1 - min_lr) * np.sin(progress * np.pi)**3
    return lr

# Flame
def lr_flame(scheduler_steps, num_cycles, min_lr=0.1):
    def lr(current_step):
        progress = (current_step / scheduler_steps) % (1 / num_cycles)
        return min_lr + (1 - min_lr) * abs(math.sin(progress * 2 * np.pi * num_cycles))
    return lr

# Semicircle
def lr_semicircle(scheduler_steps, num_cycles, min_lr=0.1):
    def lr(current_step):
        progress = (current_step / scheduler_steps) % (1 / num_cycles)
        return min_lr + (1 - min_lr) * math.sqrt(1 - (2 * progress - 1)**2)
    return lr