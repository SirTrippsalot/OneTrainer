import torch
import math
from modules.util.bf16_stochastic_rounding import add_stochastic_
from pytorch_optimizer.base.exception import NoSparseGradientError

class Adafusion(torch.optim.Optimizer):
    """Implements Adafusion: a comprehensive hybrid adaptive optimizer.

    Adafusion combines multiple advanced optimization techniques:
      - Yogi-style variance adjustment (with a hybrid update that fuses ADOPT/Adan ideas).
      - Aida-style step suppression.
      - Trust ratio scaling (à la LAMB) to adapt the effective learning rate based on parameter norms.
      - Stochastic rounding (for BF16/low-precision formats).
      - Confidence-guided weight decay.
      - Selective Projection Decay (SPD).
      - AdaEMAMix-inspired slow-moving averages.
      - Lookahead integration.
      - AutoClipper integration.
      - OrthoGrad-like gradient orthogonalization.

    Note: In this version, no relative-step is used. The effective update is self-adjusting via the trust ratio.

    Arguments:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): External learning rate. Defaults to 1.0 if not provided.
        eps (Tuple[float, float], optional): Regularization constants (default: (1e-30, 1e-3)).
        clip_threshold (float, optional): Threshold to clip the RMS of the final gradient update (default: 1.0).
        clip_updates (bool, optional): Whether to apply update clipping (default: True).
        decay_rate (float, optional): Coefficient for running averages of squared gradients (default: -0.8).
        betas (Tuple[float, float, float], optional): Coefficients for gradient, squared gradient, 
                                                      and slow EMA (default: (0.9, 0.999, 0.9999)).
        weight_decay (float, optional): Weight decay factor (default: 0).
        stochastic_rounding (bool, optional): If True, uses stochastic rounding for BF16 (default: False).
        alpha (float, optional): Scaling factor for combining fast and slow averages (default: 5).
        k (int, optional): Number of projections in Aida-style step suppression (default: 2).
        xi (float, optional): Term to avoid division by zero in projections (default: 1e-20).
        lookahead_k (int, optional): Steps between Lookahead updates (default: 5).
        lookahead_alpha (float, optional): Interpolation factor for Lookahead updates (default: 0.5).
        autoclipper (bool, optional): Enable AutoClipper gradient clipping (default: True).
        clip_percentile (float, optional): Percentile for AutoClipper (default: 10).
        history_size (int, optional): History size for AutoClipper (default: 10000).
        ortho_grad (bool, optional): If True, applies gradient orthogonalization (default: True).
        adan_coef (float, optional): Scaling factor for the gradient difference term (default: 0.5).
    """
    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        clip_updates=True,
        decay_rate=-0.8,
        betas=(0.9, 0.999, 0.9999),
        weight_decay=0.0,
        stochastic_rounding=False,
        alpha=5,
        k=2,
        xi=1e-20,
        lookahead_k=5,
        lookahead_alpha=0.5,
        autoclipper=True,
        clip_percentile=10,
        history_size=10000,
        ortho_grad=True,
        adan_coef=0.5
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
            "adan_coef": adan_coef
        }
        self.stochastic_rounding = stochastic_rounding
        self.autoclipper_enabled = autoclipper
        self.ortho_grad = ortho_grad

        if autoclipper:
            self.grad_history = torch.zeros(history_size, dtype=torch.float32)
            self.history_index = 0
            self.history_size = history_size
            self.clip_percentile = clip_percentile
            
        # Collect unique dtype values
        self.unique_dtypes = set()

        for param in params:
            if isinstance(param, dict):
                param_list = param["params"]
            else:
                param_list = [param]

            for p in param_list:
                if p is not None:
                    self.unique_dtypes.add(p.dtype)

        # Print unique dtypes
        if self.unique_dtypes:
            print(f"Adafusion initialized with unique dtypes: {self.unique_dtypes}")

        super().__init__(params, defaults)

    @staticmethod
    def _get_lr(param_group, param_state):
        # Simply use the provided lr (defaulting to 1.0 if none is given)
        return param_group["lr"] if param_group["lr"] is not None else 1.0

    @staticmethod
    def _get_options(param_group, param_shape):
        # Use factored update if parameter tensor has 2 or more dimensions.
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
        # --- AutoClipper: Gather gradient norms from all parameters in the group ---
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
        # --- Clip each parameter's gradient ---
        for p in group["params"]:
            if p.grad is not None:
                torch.nn.utils.clip_grad_norm_(p.grad, clip_value)

    @staticmethod
    def _orthogonalize_gradients(params):
        """
        Projects each gradient to be orthogonal to the current weights,
        then rescales to preserve the original gradient norm.
        """
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

    @torch.no_grad()
    def step_parameter(self, p, group, i):
        if p.grad is None:
            return

        grad = p.grad
        if grad.is_sparse:
            raise NoSparseGradientError(f"Adafusion does not support sparse gradients: {p}")

        state = self.state[p]
        grad_shape = grad.shape
        factored = self._get_options(group, grad_shape)

        # --- Initialize state ---
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(grad, device=grad.device)
            state["RMS"] = torch.zeros(1, device=grad.device, dtype=grad.dtype)
            if factored:
                state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1], device=grad.device, dtype=grad.dtype)
                state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:], device=grad.device, dtype=grad.dtype)
                state["exp_avg_slow_row"] = torch.zeros(grad_shape[:-1], device=grad.device, dtype=grad.dtype)
                state["exp_avg_slow_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:], device=grad.device, dtype=grad.dtype)
                state["exp_avg_res_row"] = torch.zeros(grad_shape[:-1], device=grad.device, dtype=grad.dtype)
                state["exp_avg_res_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:], device=grad.device, dtype=grad.dtype)
            else:
                state["exp_avg_sq"] = torch.zeros_like(grad, device=grad.device)
                state["exp_avg_slow"] = torch.zeros_like(grad, device=grad.device)
                state["exp_avg_res"] = torch.zeros_like(grad, device=grad.device)
            state["pre"] = p.detach().clone()
            # Will store the previous gradient for ADAN-style updates.
            state["prev_grad"] = None

        p_data = p.detach()
        grad_data = grad.detach()

        state["step"] += 1
        state["RMS"] = self._rms(p_data)
        lr = self._get_lr(group, state)

        beta1, beta2, beta3 = group["betas"]
        beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])

        # --- Hybrid Variance Adjustment ---
        # Use a hybrid squared gradient: average of the plain squared grad and
        # the squared difference (if a previous grad exists)
        if state["prev_grad"] is not None:
            squared_grad_update = 0.5 * (grad_data ** 2) + 0.5 * ((grad_data - state["prev_grad"]) ** 2) + group["eps"][0]
        else:
            squared_grad_update = (grad_data ** 2) + group["eps"][0]

        bias_correction1 = 1 - beta1 ** state["step"]
        bias_correction2 = 1 - beta2 ** state["step"]

        if factored:
            exp_avg_sq_row = state["exp_avg_sq_row"]
            exp_avg_sq_col = state["exp_avg_sq_col"]
            exp_avg_slow_row = state["exp_avg_slow_row"]
            exp_avg_slow_col = state["exp_avg_slow_col"]
            exp_avg_res_row = state["exp_avg_res_row"]
            exp_avg_res_col = state["exp_avg_res_col"]

            row_mean = squared_grad_update.mean(dim=-1)
            col_mean = squared_grad_update.mean(dim=-2)

            exp_avg_sq_row.mul_(beta2t).addcmul_((exp_avg_sq_row - row_mean).sign_(), row_mean, value=-(1.0 - beta2t))
            exp_avg_sq_col.mul_(beta2t).addcmul_((exp_avg_sq_col - col_mean).sign_(), col_mean, value=-(1.0 - beta2t))

            exp_avg_slow_row.mul_(beta3).addcmul_((exp_avg_slow_row - row_mean).sign_(), row_mean, value=-(1.0 - beta3))
            exp_avg_slow_col.mul_(beta3).addcmul_((exp_avg_slow_col - col_mean).sign_(), col_mean, value=-(1.0 - beta3))

            res = (squared_grad_update - exp_avg_sq_row.mean()) ** 2 + group["eps"][1]
            exp_avg_res_row.mul_(beta3).addcmul_((exp_avg_res_row - res.mean(dim=-1)).sign_(), res.mean(dim=-1), value=-(1.0 - beta3))
            exp_avg_res_col.mul_(beta3).addcmul_((exp_avg_res_col - res.mean(dim=-2)).sign_(), res.mean(dim=-2), value=-(1.0 - beta3))

            update_direction = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col) * grad_data
            update_direction = update_direction / math.sqrt(bias_correction2)
        else:
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg_slow = state["exp_avg_slow"]
            exp_avg_res = state["exp_avg_res"]

            exp_avg_sq.mul_(beta2t).addcmul_((exp_avg_sq - squared_grad_update).sign_(), squared_grad_update, value=-(1.0 - beta2t))
            exp_avg_slow.mul_(beta3).addcmul_((exp_avg_slow - squared_grad_update).sign_(), squared_grad_update, value=-(1.0 - beta3))
            res = (squared_grad_update - exp_avg_sq.mean()) ** 2 + group["eps"][1]
            exp_avg_res.mul_(beta3).addcmul_((exp_avg_res - res).sign_(), res, value=-(1.0 - beta3))
            update_direction = exp_avg_sq.rsqrt() * grad_data
            if state["step"] > 1:
                denom = exp_avg_sq.rsqrt().clamp(min=group["eps"][1])
                update_direction = update_direction / denom
            update_direction = update_direction / math.sqrt(bias_correction2)

        # --- Aida-Style Step Suppression ---
        proj_grad = grad_data.clone()
        proj_momentum = state["exp_avg"].clone()
        for _ in range(group["k"]):
            proj_sum_gm = torch.sum(proj_grad * proj_momentum)
            scalar_g = proj_sum_gm / (torch.sum(proj_grad ** 2) + group["xi"])
            scalar_m = proj_sum_gm / (torch.sum(proj_momentum ** 2) + group["xi"])
            proj_grad = proj_grad * scalar_g
            proj_momentum = proj_momentum * scalar_m

        grad_residual = proj_momentum - proj_grad
        if factored:
            exp_avg_sq_row.mul_(beta2).addcmul_(grad_residual.mean(dim=-1), grad_residual.mean(dim=-1), value=1.0 - beta2)
            exp_avg_sq_col.mul_(beta2).addcmul_(grad_residual.mean(dim=-2), grad_residual.mean(dim=-2), value=1.0 - beta2)
        else:
            exp_avg_sq.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1.0 - beta2)

        # --- Update Clipping ---
        if group["clip_updates"]:
            update_rms = self._rms(update_direction)
            clipping_factor = (update_rms / group["clip_threshold"]).clamp(min=1.0)
            update_direction = update_direction / clipping_factor

        # --- ADAN-Style Gradient Difference Addition ---
        if state["prev_grad"] is not None:
            grad_diff = grad_data - state["prev_grad"]
            update_direction.add_(grad_diff, alpha=group["adan_coef"])

        # --- Trust Ratio Scaling ---
        param_norm = p_data.norm(2)
        update_norm = update_direction.norm(2)
        trust_ratio = param_norm / (update_norm + 1e-9) if update_norm > 0 else 1.0
        update_direction.mul_(trust_ratio)

        update_direction.mul_(lr)

        # --- Update the first moment estimate ---
        exp_avg = state["exp_avg"]
        exp_avg.mul_(beta1).add_(update_direction, alpha=(1 - beta1))
        exp_avg_corrected = exp_avg / bias_correction1

        # --- Final Update: Combine fast and slow averages ---
        if factored:
            final_update = exp_avg_corrected + group["alpha"] * (state["exp_avg_slow_row"].mean() + state["exp_avg_slow_col"].mean())
        else:
            final_update = exp_avg_corrected + group["alpha"] * state["exp_avg_slow"].mean()

        # --- Confidence-Guided Adjustment and Weight Decay ---
        pre = state["pre"]
        condition = -torch.sum(grad_data * (p_data - pre))
        if condition < 0.0:
            ratio = self._ratio(p_data - final_update, p_data, pre)
            final_update = final_update - group["weight_decay"] * ratio * (p_data - pre)

        if group["weight_decay"] != 0:
            p_data.add_(p_data, alpha=(-group["weight_decay"] * lr))

        # --- Update the parameter ---
        p_data.add_(-final_update)
        p.copy_(p_data)
        state["pre"] = p_data.clone()

        # --- Stochastic Rounding for BF16 (if enabled) ---
        if p.dtype == torch.bfloat16 and self.stochastic_rounding:
            add_stochastic_(p, p_data)

        # --- Lookahead Integration ---
        if group['lookahead_k'] > 0 and state['step'] % group['lookahead_k'] == 0:
            lookahead_alpha = group['lookahead_alpha']
            if factored:
                fast_row = state["exp_avg"].mean(dim=-1)
                fast_col = state["exp_avg"].mean(dim=-2)
                state["exp_avg_slow_row"].mul_(lookahead_alpha).add_(fast_row, alpha=(1 - lookahead_alpha))
                state["exp_avg_slow_col"].mul_(lookahead_alpha).add_(fast_col, alpha=(1 - lookahead_alpha))
            else:
                state["exp_avg"].mul_(lookahead_alpha).add_(state["exp_avg_slow"], alpha=(1 - lookahead_alpha))
                state["exp_avg_slow"].copy_(state["exp_avg"])

        # --- Update previous gradient for the next iteration ---
        state["prev_grad"] = grad_data.clone()

    def _ratio(self, new_p, param, pre):
        curr_norm = torch.norm(new_p - pre)
        prev_norm = torch.norm(param - pre)
        ratio = (curr_norm - prev_norm) / (curr_norm + 1e-9)
        return torch.nn.functional.hardtanh(ratio, 0.0, 1.0)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # Optionally orthogonalize gradients for stability.
        for group in self.param_groups:
            if self.ortho_grad:
                self._orthogonalize_gradients(group['params'])
            # --- Apply AutoClipper once per parameter group ---
            if self.autoclipper_enabled:
                self._apply_autoclipper(group)
            for i, p in enumerate(group["params"]):
                self.step_parameter(p, group, i)
        return loss
