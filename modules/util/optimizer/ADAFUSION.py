import torch
import math
from modules.util.bf16_stochastic_rounding import add_stochastic_

class Adafusion(torch.optim.Optimizer):
    """Implements Adafusion: a hybrid adaptive optimizer combining Yogi-style variance control, Aida step suppression, stochastic rounding, confidence-guided strategies, unified Prodigy-style relative learning rate management, and dynamic adaptation inspired by Prodigy and Adabelief.

    Adafusion is designed to provide a comprehensive approach to training optimization by incorporating multiple techniques:
    - **Yogi-style variance adjustment**: Helps control the accumulation of second-order moments to ensure stability in long-term training.
    - **Aida-style step suppression**: Dynamically controls the size of parameter updates to handle sudden gradient changes.
    - **Stochastic rounding**: Enables rounding for BF16 precision, providing numerical stability in low-precision contexts.
    - **Confidence-guided strategy**: Estimates the instability of updates to ensure robust parameter adjustments.
    - **Selective Projection Decay (SPD)**: Controls the decay of updates by projecting them selectively, ensuring stability in optimization steps.
    - **D-Adaptation Inspired Learning Rate Adjustment**: Dynamically adapts the learning rate using gradient magnitude and consistency.
    - **Schedule-Free Updates and Warmup Growth Control**: Uses schedule-free relative learning rates with growth control during warmup phases.

    Arguments:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): External learning rate. Typically left as `None` if using relative or unified relative step approach.
        eps (Tuple[float, float], optional): Regularization constants for stabilizing the update computations. The first value regularizes the square gradient, and the second regularizes the parameter scaling (default: (1e-30, 0.001)).
        clip_threshold (float, optional): Threshold value to clip the root mean square of the final gradient update, which prevents overly large updates and ensures numerical stability (default: 1.0).
        decay_rate (float, optional): Coefficient for computing running averages of squared gradients. Controls the decay of past gradients in Yogi-style variance adjustment (default: -0.8).
        betas (Tuple[float, float, float], optional): Coefficients used for computing running averages of gradient, squared gradient, and slow exponential moving average (EMA) respectively. These coefficients control the balance between past and new gradient information (default: (0.9, 0.999, 0.9999)).
        weight_decay (float, optional): Weight decay (L2 penalty). Used to prevent overfitting by adding a penalty proportional to the size of the parameters (default: 0).
        scale_parameter (bool, optional): If `True`, the learning rate is scaled by the root mean square of the parameter values, allowing for adaptive scaling of updates based on parameter magnitude (default: True).
        warmup_init (bool, optional): Specifies if the warm-up initialization is enabled. This option is effective only if `relative_step` is used, ensuring smooth initialization of learning rates (default: False).
        stochastic_rounding (bool, optional): Whether to utilize stochastic rounding with BF16 precision, which improves numerical stability when using low-precision formats (default: False).
        alpha (float, optional): Scaling factor for combining different update components during optimization. Influences the mix between slow and fast-moving averages in the parameter update process (default: 5).
        relative_step (bool, optional): Enables relative step, allowing dynamic learning rate adjustment. (default: True).
        min_step (float, optional): Minimum step size for relative step. If `None`, the current behavior is maintained. (default: None).
        d0 (float, optional): Initial D estimate for D-adaptation, used to dynamically adjust the learning rate (default: 1e-6).
        growth_rate (float, optional): Growth rate during the warm-up phase, controlling how quickly the learning rate can adapt upwards (default: inf).
    """
    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        betas=(0.9, 0.999, 0.9999),
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,
        stochastic_rounding=False,
        alpha=5,
        min_step=0.03,
        d0=1e-6,
        growth_rate=float('inf')
    ):
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual `lr` and `relative_step=True` options")
        if warmup_init and not relative_step:
            raise ValueError("`warmup_init=True` requires `relative_step=True")

        defaults = {
            "lr": lr,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "decay_rate": decay_rate,
            "betas": betas,
            "weight_decay": weight_decay,
            "scale_parameter": scale_parameter,
            "relative_step": relative_step,
            "warmup_init": warmup_init,
            "alpha": alpha,
            "min_step": min_step,
            "d": d0,
            "d_max": d0,
            "growth_rate": growth_rate
        }
        super().__init__(params, defaults)
        self.stochastic_rounding = stochastic_rounding

    @staticmethod
    def _get_lr(param_group, param_state):
        d = param_group["d"]
        lr = param_group["lr"] if param_group["lr"] is not None else 1.0
        rel_step_sz = lr * d
        if param_group["relative_step"]:
            min_step = param_group["min_step"] if param_group["min_step"] is not None else (1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2)
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"])) * d

        # Apply parameter scaling if scale_parameter is enabled
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps"][1], param_state["RMS"])
        return param_scale * rel_step_sz

    @staticmethod
    def _get_options(param_group, param_shape):
        factored = len(param_shape) >= 2
        return factored

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    @torch.no_grad()
    def step_parameter(self, p, group, i):
        if p.grad is None:
            return
        grad = p.grad
        if grad.dtype in {torch.float16, torch.bfloat16}:
            grad = grad.float()
        if grad.is_sparse:
            raise RuntimeError("Adafusion does not support sparse gradients.")

        state = self.state[p]
        grad_shape = grad.shape

        factored = self._get_options(group, grad_shape)
        if len(state) == 0:
            state["step"] = 0

            # Always initialize exp_avg since confidence strategy requires it
            state["exp_avg"] = torch.zeros_like(grad)
            if factored:
                state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
                state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(grad)
                state["exp_avg_slow_row"] = torch.zeros(grad_shape[:-1]).to(grad)
                state["exp_avg_slow_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(grad)
                state["exp_avg_res_row"] = torch.zeros(grad_shape[:-1]).to(grad)
                state["exp_avg_res_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(grad)
                state["pre"] = torch.zeros_like(p)
            else:
                state["exp_avg_sq"] = torch.zeros_like(grad)
                state["exp_avg_slow"] = torch.zeros_like(grad)
                state["exp_avg_res"] = torch.zeros_like(grad)
                state["pre"] = torch.zeros_like(p)

            state["RMS"] = 0
        else:
            state["exp_avg"] = state["exp_avg"].to(grad)
            if factored:
                state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
                state["exp_avg_slow_row"] = state["exp_avg_slow_row"].to(grad)
                state["exp_avg_slow_col"] = state["exp_avg_slow_col"].to(grad)
                state["exp_avg_res_row"] = state["exp_avg_res_row"].to(grad)
                state["exp_avg_res_col"] = state["exp_avg_res_col"].to(grad)
            else:
                state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)
                state["exp_avg_slow"] = state["exp_avg_slow"].to(grad)
                state["exp_avg_res"] = state["exp_avg_res"].to(grad)

        p_data_fp32 = p
        if p.dtype in {torch.float16, torch.bfloat16}:
            p_data_fp32 = p_data_fp32.float()

        state["step"] += 1
        state["RMS"] = self._rms(p_data_fp32)
        lr = self._get_lr(group, state)

        beta1, beta2, beta3 = group["betas"]
        beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
        update = (grad ** 2) + group["eps"][0]
        if factored:
            exp_avg_sq_row = state["exp_avg_sq_row"]
            exp_avg_sq_col = state["exp_avg_sq_col"]
            exp_avg_slow_row = state["exp_avg_slow_row"]
            exp_avg_slow_col = state["exp_avg_slow_col"]
            exp_avg_res_row = state["exp_avg_res_row"]
            exp_avg_res_col = state["exp_avg_res_col"]

            exp_avg_sq_row.mul_(beta2t).addcmul_((exp_avg_sq_row - update.mean(dim=-1)).sign_(), update.mean(dim=-1), value=-(1.0 - beta2t))
            exp_avg_sq_col.mul_(beta2t).addcmul_((exp_avg_sq_col - update.mean(dim=-2)).sign_(), update.mean(dim=-2), value=-(1.0 - beta2t))

            # Apply Yogi-style variance adjustment to slow EMA
            exp_avg_slow_row.mul_(beta3).addcmul_((exp_avg_slow_row - update.mean(dim=-1)).sign_(), update.mean(dim=-1), value=-(1.0 - beta3))
            exp_avg_slow_col.mul_(beta3).addcmul_((exp_avg_slow_col - update.mean(dim=-2)).sign_(), update.mean(dim=-2), value=-(1.0 - beta3))

            # Confidence-guided strategy: Calculate instability
            res = (update - exp_avg_sq_row.mean()) ** 2 + group["eps"][1]
            exp_avg_res_row.mul_(beta3).addcmul_((exp_avg_res_row - res.mean(dim=-1)).sign_(), res.mean(dim=-1), value=-(1.0 - beta3))
            exp_avg_res_col.mul_(beta3).addcmul_((exp_avg_res_col - res.mean(dim=-2)).sign_(), res.mean(dim=-2), value=-(1.0 - beta3))

            # Approximation of exponential moving average of instability
            res_approx = self._approx_sq_grad(exp_avg_res_row, exp_avg_res_col)
            update = res_approx.mul_(grad)

            update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
            update.mul_(grad)
        else:
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg_slow = state["exp_avg_slow"]
            exp_avg_res = state["exp_avg_res"]

            exp_avg_sq.mul_(beta2t).addcmul_((exp_avg_sq - update).sign_(), update, value=-(1.0 - beta2t))
            exp_avg_slow.mul_(beta3).addcmul_((exp_avg_slow - update).sign_(), update, value=-(1.0 - beta3))
            
            # Confidence-guided strategy: Calculate instability
            res = (update - exp_avg_sq.mean()) ** 2 + group["eps"][1]
            exp_avg_res.mul_(beta3).addcmul_((exp_avg_res - res).sign_(), res, value=-(1.0 - beta3))
            update = exp_avg_sq.rsqrt().mul_(grad)

        update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
        update.mul_(lr)

        exp_avg = state["exp_avg"]
        exp_avg.mul_(beta1).add_(update, alpha=(1 - beta1))
        update = exp_avg + group["alpha"] * (exp_avg_slow_row.mean() + exp_avg_slow_col.mean())

        # Selective Projection Decay (SPD)
        pre = state["pre"]
        condition = - torch.sum(torch.mul(grad, p_data_fp32 - pre))
        if condition < 0.0:
            ratio = self._ratio(p_data_fp32 - update, p_data_fp32, pre)
            update = update - group["weight_decay"] * ratio * (p_data_fp32 - pre)

        if group["weight_decay"] != 0:
            p_data_fp32.add_(p_data_fp32, alpha=(-group["weight_decay"] * lr))

        p_data_fp32.add_(-update)
        state["pre"] = p_data_fp32.clone()

        if p.dtype == torch.bfloat16 and self.stochastic_rounding:
            add_stochastic_(p, p_data_fp32)
        if p.dtype in {torch.float16, torch.bfloat16}:
            p.copy_(p_data_fp32)

    def _ratio(self, new_p, param, pre):
        curr_norm, prev_norm = torch.norm(new_p - pre), torch.norm(param - pre)
        ratio = (curr_norm - prev_norm) / curr_norm
        return torch.nn.functional.hardtanh(ratio, 0.0, 1.0)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        d_numerator = 0.0
        d_denom = 0.0

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is not None:
                    grad = p.grad
                    d_numerator += torch.sum(grad ** 2).item()
                    d_denom += torch.sum(grad.abs()).item()
                self.step_parameter(p, group, i)

            if d_denom > 0:
                d_hat = group['d_coef'] * (d_numerator / d_denom)
                group['d'] = min(group['d_max'], d_hat * group['growth_rate'])

        return loss
