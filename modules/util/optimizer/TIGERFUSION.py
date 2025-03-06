import torch
import math
from torch.optim.optimizer import Optimizer, ParamsT
from modules.util.bf16_stochastic_rounding import add_stochastic_
from pytorch_optimizer.base.exception import NoSparseGradientError

class TigerFusion(Optimizer):
    r"""TigerFusion Optimizer

    TigerFusion integrates advanced adaptive techniques on a Tiger (sign-based)
    update base. It combines:
      - Hybrid variance control (merging Yogi, ADOPT, and Adan ideas)
      - Aida-style step suppression via iterative projection
      - Trust ratio scaling (à la LAMB)
      - Confidence-guided weight decay
      - Lookahead integration
      - AutoClipper and OrthoGrad stabilization
      - Stochastic rounding for BF16
    The final update is performed using a sign-based update (Tiger-style).
    
    The optimizer retains the step() → step_parameter() structure.

    Arguments:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate (default: 1e-3).
        eps (Tuple[float, float], optional): Regularization constants (default: (1e-30, 1e-3)).
        clip_threshold (float, optional): Threshold to clip the RMS of the update (default: 1.0).
        clip_updates (bool, optional): Whether to apply update clipping (default: True).
        decay_rate (float, optional): Coefficient for running averages of squared gradients (default: -0.8).
        betas (Tuple[float, float, float], optional): Coefficients for momentum, variance, and slow EMA 
            (default: (0.9, 0.999, 0.9999)).
        weight_decay (float, optional): Weight decay factor (default: 0).
        stochastic_rounding (bool, optional): Enable stochastic rounding for BF16 (default: False).
        alpha (float, optional): Scaling factor for combining fast and slow averages (default: 5).
        k (int, optional): Number of projections in Aida-style step suppression (default: 2).
        xi (float, optional): Term to avoid division by zero in projections (default: 1e-20).
        lookahead_k (int, optional): Frequency of Lookahead updates (default: 5).
        lookahead_alpha (float, optional): Interpolation factor for Lookahead (default: 0.5).
        autoclipper (bool, optional): Enable AutoClipper gradient clipping (default: True).
        clip_percentile (float, optional): Percentile for AutoClipper (default: 10).
        history_size (int, optional): History size for AutoClipper (default: 10000).
        ortho_grad (bool, optional): Enable gradient orthogonalization (default: True).
        adan_coef (float, optional): Scaling for the gradient difference term (default: 0.5).
    """
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        eps=(1e-30, 1e-3),
        clip_threshold: float = 1.0,
        clip_updates: bool = True,
        decay_rate: float = -0.8,
        betas=(0.9, 0.999, 0.9999),
        weight_decay: float = 0.0,
        stochastic_rounding: bool = False,
        alpha: float = 5,
        k: int = 2,
        xi: float = 1e-20,
        lookahead_k: int = 5,
        lookahead_alpha: float = 0.5,
        autoclipper: bool = True,
        clip_percentile: float = 10,
        history_size: int = 5000,
        ortho_grad: bool = True,
        adan_coef: float = 0.5
    ):
        defaults = {
            "lr": lr,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "clip_updates": clip_updates,
            "decay_rate": decay_rate,
            "betas": betas,
            "weight_decay": weight_decay,
            "alpha": alpha,
            "k": k,
            "xi": xi,
            "lookahead_k": lookahead_k,
            "lookahead_alpha": lookahead_alpha,
            "autoclipper": autoclipper,
            "clip_percentile": clip_percentile,
            "history_size": history_size,
            "adan_coef": adan_coef,
        }
        self.stochastic_rounding = stochastic_rounding
        self.autoclipper_enabled = autoclipper
        self.ortho_grad = ortho_grad

        if autoclipper:
            self.grad_history = torch.zeros(history_size, dtype=torch.float32)
            self.history_index = 0
            self.history_size = history_size
            self.clip_percentile = clip_percentile

        super().__init__(params, defaults)

    @staticmethod
    def _get_lr(param_group, param_state):
        return param_group["lr"] if param_group["lr"] is not None else 1.0

    @staticmethod
    def _get_options(param_group, param_shape):
        # Use factored update for tensors with 2 or more dimensions.
        return len(param_shape) >= 2

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    def _update_grad_history(self, total_norm):
        assign_idx = self.history_index % self.history_size
        self.grad_history[assign_idx] = total_norm
        self.history_index += 1

    def _compute_clip_value(self):
        history_size = min(self.history_index, self.history_size)
        return torch.quantile(self.grad_history[:history_size], self.clip_percentile / 100.0)

    def _apply_autoclipper(self, group):
        # Gather gradient norms and compute clipping value.
        grad_norms = []
        for p in group["params"]:
            if p.grad is not None:
                grad_norms.append(p.grad.norm().detach())
        if not grad_norms:
            return
        device = grad_norms[0].device
        grad_norms_tensor = torch.stack(grad_norms).to(device=device, dtype=torch.float32)
        total_norm = grad_norms_tensor.norm()
        self._update_grad_history(total_norm.item())
        clip_value = self._compute_clip_value().item()
        for p in group["params"]:
            if p.grad is not None:
                torch.nn.utils.clip_grad_norm_(p.grad, clip_value)

    @staticmethod
    def _orthogonalize_gradients(params):
        # Projects gradients to be orthogonal to the parameters.
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    w = p.view(-1)
                    g = p.grad.view(-1)
                    w_norm_sq = torch.dot(w, w) + 1e-30
                    proj = torch.dot(w, g) / w_norm_sq
                    g_orth = g - proj * w
                    g_norm = g.norm(2)
                    g_orth_norm = g_orth.norm(2) + 1e-30
                    g_orth_scaled = g_orth * (g_norm / g_orth_norm)
                    p.grad.copy_(g_orth_scaled.view_as(p.grad))

    def _ratio(self, new_p, param, pre):
        curr_norm = torch.norm(new_p - pre)
        prev_norm = torch.norm(param - pre)
        ratio = (curr_norm - prev_norm) / (curr_norm + 1e-9)
        return torch.nn.functional.hardtanh(ratio, 0.0, 1.0)

    @torch.no_grad()
    def step_parameter(self, p, group, i):
        if p.grad is None:
            return
        grad = p.grad
        if grad.is_sparse:
            raise NoSparseGradientError(f"TigerFusion does not support sparse gradients: {p}")

        state = self.state[p]
        factored = self._get_options(group, grad.shape)

        # --- State Initialization ---
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(grad, device=grad.device)
            state["RMS"] = torch.zeros(1, device=grad.device, dtype=grad.dtype)
            if factored:
                state["exp_avg_sq_row"] = torch.zeros(grad.shape[:-1], device=grad.device, dtype=grad.dtype)
                state["exp_avg_sq_col"] = torch.zeros(grad.shape[:-2] + grad.shape[-1:], device=grad.device, dtype=grad.dtype)
                state["exp_avg_slow_row"] = torch.zeros(grad.shape[:-1], device=grad.device, dtype=grad.dtype)
                state["exp_avg_slow_col"] = torch.zeros(grad.shape[:-2] + grad.shape[-1:], device=grad.device, dtype=grad.dtype)
                state["exp_avg_res_row"] = torch.zeros(grad.shape[:-1], device=grad.device, dtype=grad.dtype)
                state["exp_avg_res_col"] = torch.zeros(grad.shape[:-2] + grad.shape[-1:], device=grad.device, dtype=grad.dtype)
            else:
                state["exp_avg_sq"] = torch.zeros_like(grad, device=grad.device)
                state["exp_avg_slow"] = torch.zeros_like(grad, device=grad.device)
                state["exp_avg_res"] = torch.zeros_like(grad, device=grad.device)
            state["pre"] = p.detach().clone()
            state["prev_grad"] = None

        # --- Tiger-Style Weight Decay ---
        p.data.mul_(1 - group["lr"] * group["weight_decay"])

        p_data = p.detach()
        grad_data = grad.detach()
        state["step"] += 1
        state["RMS"] = self._rms(p_data)
        lr_ = self._get_lr(group, state)
        beta1, beta2, beta3 = group["betas"]
        beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])

        # --- Hybrid Variance Adjustment ---
        if state["prev_grad"] is not None:
            squared_grad_updat
