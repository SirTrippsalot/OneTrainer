import torch
import math
from modules.util.bf16_stochastic_rounding import add_stochastic_
from pytorch_optimizer.base.exception import NoSparseGradientError

class Adafusion(torch.optim.Optimizer):
    """Implements Adafusion: a comprehensive hybrid adaptive optimizer.

    Adafusion is a sophisticated optimizer combining multiple advanced optimization techniques, including Yogi-style variance control, Aida-style step suppression, stochastic rounding, confidence-guided strategies, Selective Projection Decay (SPD), dynamic adaptation inspired by AdaEMAMix, and the convergence properties inspired by ADOPT.

    Key features of Adafusion include:
    - **Yogi-style variance adjustment**: Controls the accumulation of second-order moments to maintain stability during long-term training.
    - **Aida-style step suppression**: Manages parameter update sizes dynamically to handle sudden changes in gradients.
    - **Stochastic rounding**: Utilizes stochastic rounding for BF16 precision, improving numerical stability in low-precision environments.
    - **CAME Confidence-guided strategy**: Estimates the instability of parameter updates to ensure robust adjustments.
    - **Selective Projection Decay (SPD)**: Controls the decay of parameter updates selectively, promoting stability in optimization.
    - **AdaEMAMix-inspired adaptation**: Incorporates techniques from AdaEMAMix to enhance the balance between faster adaptation and long-term stability.
    - **ADOPT-inspired variance reduction**: Modifies second moment estimates by removing the current gradient from the accumulation, achieving robust convergence across different conditions.
    - **Lookahead integration**: Optionally applies Lookahead-style weight updates every `lookahead_k` steps, combining fast and slow weights for increased stability. Set `lookahead_k` to 0 to disable this functionality.
    - **AutoClipper integration**: Optional gradient clipping based on percentile statistics to control gradient magnitudes dynamically during training.

    Arguments:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): External learning rate. Typically left as `None` if using relative or unified relative step approach.
        eps (Tuple[float, float], optional): Regularization constants for stabilizing the update computations. The first value regularizes the square gradient, and the second regularizes the parameter scaling (default: (1e-30, 0.001)).
        clip_threshold (float, optional): Threshold value to clip the root mean square of the final gradient update, which prevents overly large updates and ensures numerical stability (default: 1.0).
        clip_updates (bool, optional): Whether to apply **Update Clipping** based on `clip_threshold`. This is distinct from Gradient Clipping and is not recommended to use both simultaneously. **Update Clipping** ensures the magnitude of parameter updates is controlled (default: True).
        decay_rate (float, optional): Coefficient for computing running averages of squared gradients. Controls the decay of past gradients in Yogi-style variance adjustment (default: -0.8).
        betas (Tuple[float, float, float], optional): Coefficients used for computing running averages of gradient, squared gradient, and slow exponential moving average (EMA) respectively. These coefficients control the balance between past and new gradient information (default: (0.9, 0.999, 0.9999)).
        weight_decay (float, optional): Weight decay (L2 penalty). Used to prevent overfitting by adding a penalty proportional to the size of the parameters (default: 0).
        scale_parameter (bool, optional): If `True`, the learning rate is scaled by the root mean square of the parameter values, allowing for adaptive scaling of updates based on parameter magnitude (default: True).
        warmup_init (bool, optional): Specifies if the warm-up initialization is enabled. This option is effective only if `relative_step` is used, ensuring smooth initialization of learning rates (default: False).
        stochastic_rounding (bool, optional): Whether to utilize stochastic rounding with BF16 precision, which improves numerical stability when using low-precision formats (default: False).
        alpha (float, optional): Scaling factor for combining different update components during optimization. Influences the mix between slow and fast-moving averages in the parameter update process (default: 5).
        relative_step (bool, optional): Enables relative step, allowing dynamic learning rate adjustment. (default: True).
        min_step (float, optional): Minimum step size for relative step. If `None`, the current behavior is maintained. (default: None).
        k (int, optional): Number of vector projections per iteration in Aida-style step suppression (default: 2).
        xi (float, optional): Term used in vector projections to avoid division by zero in Aida-style step suppression (default: 1e-20).
        lookahead_k (int, optional): Number of steps before applying Lookahead update. Set to 0 to disable Lookahead functionality (default: 5).
        lookahead_alpha (float, optional): Interpolation factor for Lookahead updates between fast and slow weights. Controls the mixing of fast and slow parameters during the Lookahead update (default: 0.5).
        autoclipper (bool, optional): Enable AutoClipper gradient clipping based on percentile (default: False).
        clip_percentile (float, optional): Percentile for AutoClipper (only used if `autoclipper=True`) (default: 10).
        history_size (int, optional): Number of gradient norms to retain for percentile calculation (default: 10000).
    """
    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        clip_updates=False,
        decay_rate=-0.8,
        betas=(0.9, 0.999, 0.9999),
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,
        stochastic_rounding=False,
        alpha=5,
        min_step=None,
        k=2,
        xi=1e-20,
        lookahead_k=5,
        lookahead_alpha=0.5,
        autoclipper=True,
        clip_percentile=10,
        history_size=10000
    ):
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual `lr` and `relative_step=True` options")
        if warmup_init and not relative_step:
            raise ValueError("`warmup_init=True` requires `relative_step=True")

        defaults = {
            "lr": lr,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "clip_updates": clip_updates,
            "decay_rate": decay_rate,
            "betas": betas,
            "weight_decay": weight_decay,
            "scale_parameter": scale_parameter,
            "relative_step": relative_step,
            "warmup_init": warmup_init,
            "alpha": alpha,
            "min_step": min_step,
            "k": k,
            "xi": xi,
            "lookahead_k": lookahead_k,
            "lookahead_alpha": lookahead_alpha,
            "autoclipper": autoclipper,
            "clip_percentile": clip_percentile,
            "history_size": history_size
        }
        self.stochastic_rounding = stochastic_rounding
        self.autoclipper_enabled = autoclipper

        if autoclipper:
            self.grad_history = torch.zeros(history_size, dtype=torch.float32)
            self.history_index = 0
            self.history_size = history_size
            self.clip_percentile = clip_percentile

        super().__init__(params, defaults)

    @staticmethod
    def _get_lr(param_group, param_state):
        lr = param_group["lr"] if param_group["lr"] is not None else 1.0
        rel_step_sz = lr
        if param_group["relative_step"]:
            min_step = param_group["min_step"] if param_group["min_step"] is not None else (1e-6 * param_state.get("step", 1) if param_group["warmup_init"] else 1e-2)
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state.get("step", 1)))

        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps"][1], param_state.get("RMS", 1.0))
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

    def _update_grad_history(self, total_norm):
        assign_idx = self.history_index % self.history_size
        self.grad_history[assign_idx] = total_norm
        self.history_index += 1

    def _compute_clip_value(self):
        history_size = min(self.history_index, self.history_size)
        return torch.quantile(self.grad_history[:history_size], self.clip_percentile / 100.0)

    def _apply_autoclipper(self, group):
        grad_norms = []
        for p in group["params"]:
            if p.grad is not None:
                grad_norms.append(p.grad.norm().item())

        if not grad_norms:
            return

        total_norm = torch.norm(torch.tensor(grad_norms, dtype=torch.float32))
        self._update_grad_history(total_norm)

        clip_value = self._compute_clip_value()
        for p in group["params"]:
            if p.grad is not None:
                p.grad.data = torch.nn.utils.clip_grad_norm_(p.grad, clip_value)

    @torch.no_grad()
    def step_parameter(self, p, group, i):
        if p.grad is None:
            return
        if self.autoclipper_enabled:
            self._apply_autoclipper(group)

        grad = p.grad
        if grad.is_sparse:
            raise RuntimeError("Adafusion does not support sparse gradients.")

        state = self.state[p]
        grad_shape = grad.shape

        factored = self._get_options(group, grad_shape)

        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(grad, dtype=p.dtype)
            state["RMS"] = torch.zeros(1).to(grad).to(p.dtype)

            if factored:
                state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1], dtype=p.dtype).type_as(grad)
                state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:], dtype=p.dtype).type_as(grad)
                state["exp_avg_slow_row"] = torch.zeros(grad_shape[:-1], dtype=p.dtype).type_as(grad)
                state["exp_avg_slow_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:], dtype=p.dtype).type_as(grad)
                state["exp_avg_res_row"] = torch.zeros(grad_shape[:-1], dtype=p.dtype).type_as(grad)
                state["exp_avg_res_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:], dtype=p.dtype).type_as(grad)
                state["pre"] = torch.zeros_like(p, dtype=p.dtype)
            else:
                state["exp_avg_sq"] = torch.zeros_like(grad, dtype=p.dtype)
                state["exp_avg_slow"] = torch.zeros_like(grad, dtype=p.dtype)
                state["exp_avg_res"] = torch.zeros_like(grad, dtype=p.dtype)
                state["pre"] = torch.zeros_like(p, dtype=p.dtype)
        else:
            for key in state:
                if isinstance(state[key], torch.Tensor):
                    state[key] = state[key].to(grad.device)

        p_data_fp32 = p.to(torch.float32)
        grad_fp32 = grad.to(torch.float32)

        state["step"] += 1
        state["RMS"] = self._rms(p_data_fp32).to(p.dtype)
        lr = self._get_lr(group, state)

        beta1, beta2, beta3 = group["betas"]
        beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
        update = (grad_fp32 ** 2) + group["eps"][0]
        
        bias_correction1 = 1 - beta1 ** state["step"]
        bias_correction2 = 1 - beta2 ** state["step"]
        
        if factored:
            exp_avg_sq_row = state["exp_avg_sq_row"].to(torch.float32)
            exp_avg_sq_col = state["exp_avg_sq_col"].to(torch.float32)
            exp_avg_slow_row = state["exp_avg_slow_row"].to(torch.float32)
            exp_avg_slow_col = state["exp_avg_slow_col"].to(torch.float32)
            exp_avg_res_row = state["exp_avg_res_row"].to(torch.float32)
            exp_avg_res_col = state["exp_avg_res_col"].to(torch.float32)

            exp_avg_sq_row.mul_(beta2t).addcmul_((exp_avg_sq_row - update.mean(dim=-1)).sign_(), update.mean(dim=-1), value=-(1.0 - beta2t))
            exp_avg_sq_col.mul_(beta2t).addcmul_((exp_avg_sq_col - update.mean(dim=-2)).sign_(), update.mean(dim=-2), value=-(1.0 - beta2t))

            exp_avg_slow_row.mul_(beta3).addcmul_((exp_avg_slow_row - update.mean(dim=-1)).sign_(), update.mean(dim=-1), value=-(1.0 - beta3))
            exp_avg_slow_col.mul_(beta3).addcmul_((exp_avg_slow_col - update.mean(dim=-2)).sign_(), update.mean(dim=-2), value=-(1.0 - beta3))

            res = (update - exp_avg_sq_row.mean()) ** 2 + group["eps"][1]
            exp_avg_res_row.mul_(beta3).addcmul_((exp_avg_res_row - res.mean(dim=-1)).sign_(), res.mean(dim=-1), value=-(1.0 - beta3))
            exp_avg_res_col.mul_(beta3).addcmul_((exp_avg_res_col - res.mean(dim=-2)).sign_(), res.mean(dim=-2), value=-(1.0 - beta3))

            res_approx = self._approx_sq_grad(exp_avg_res_row, exp_avg_res_col)
            update = res_approx.mul_(grad_fp32)

            update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
            update.mul_(grad_fp32)
            update = update / math.sqrt(bias_correction2)
        else:
            exp_avg_sq = state["exp_avg_sq"].to(torch.float32)
            exp_avg_slow = state["exp_avg_slow"].to(torch.float32)
            exp_avg_res = state["exp_avg_res"].to(torch.float32)

            exp_avg_sq.mul_(beta2t).addcmul_((exp_avg_sq - update).sign_(), update, value=-(1.0 - beta2t))
            exp_avg_slow.mul_(beta3).addcmul_((exp_avg_slow - update).sign_(), update, value=-(1.0 - beta3))

            res = (update - exp_avg_sq.mean()) ** 2 + group["eps"][1]
            exp_avg_res.mul_(beta3).addcmul_((exp_avg_res - res).sign_(), res, value=-(1.0 - beta3))
            update = exp_avg_sq.rsqrt().mul_(grad_fp32)

            if state["step"] > 1:
                denom = exp_avg_sq.rsqrt().clamp(min=group["eps"][1])
                update.div_(denom)

            update = update / math.sqrt(bias_correction2)


        proj_g = grad_fp32.detach().clone().to(p.device)  
        proj_m = state["exp_avg"].to(torch.float32).detach().clone().to(p.device)  

        for _ in range(group["k"]):
            proj_sum_gm = torch.sum(torch.mul(proj_g, proj_m))

            scalar_g = proj_sum_gm / (torch.sum(torch.pow(proj_g, 2)).add(group["xi"]))
            scalar_m = proj_sum_gm / (torch.sum(torch.pow(proj_m, 2)).add(group["xi"]))

            proj_g = proj_g * scalar_g  
            proj_m = proj_m * scalar_m  

        grad_residual = proj_m - proj_g
        if factored:
            exp_avg_sq_row.mul_(beta2).addcmul_(grad_residual.mean(dim=-1), grad_residual.mean(dim=-1), value=1.0 - beta2)
            exp_avg_sq_col.mul_(beta2).addcmul_(grad_residual.mean(dim=-2), grad_residual.mean(dim=-2), value=1.0 - beta2)
        else:
            exp_avg_sq.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1.0 - beta2)

        if group["clip_updates"]:
            update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
        update.mul_(lr)

        exp_avg = state["exp_avg"].to(torch.float32)
        exp_avg.mul_(beta1).add_(update, alpha=(1 - beta1))
        exp_avg = exp_avg / bias_correction1
        
        if factored:
            update = exp_avg + group["alpha"] * (exp_avg_slow_row.mean() + exp_avg_slow_col.mean())
        else:
            update = exp_avg + group["alpha"] * exp_avg_slow.mean()
        
        pre = state["pre"].to(torch.float32)
        condition = - torch.sum(torch.mul(grad_fp32, p_data_fp32 - pre))
        if condition < 0.0:
            ratio = self._ratio(p_data_fp32 - update, p_data_fp32, pre)
            update = update - group["weight_decay"] * ratio * (p_data_fp32 - pre)

        if group["weight_decay"] != 0:
            p_data_fp32.add_(p_data_fp32, alpha=(-group["weight_decay"] * lr))

        p_data_fp32.add_(-update)
        p.copy_(p_data_fp32.to(p.dtype))
        state["pre"] = p_data_fp32.clone().to(p.dtype)

        if p.dtype == torch.bfloat16 and self.stochastic_rounding:
            add_stochastic_(p, p_data_fp32)
        

        if group['lookahead_k'] > 0:
            if state['step'] % group['lookahead_k'] == 0:
                if "exp_avg" in state and "exp_avg_slow" in state:
                    state["RMS"] = state["RMS"].detach().clone()
                    lookahead_alpha = group['lookahead_alpha']
                    state["exp_avg"].mul_(lookahead_alpha).add_(state["exp_avg_slow"].to(torch.float32), alpha=(1 - lookahead_alpha))
                    state["exp_avg_slow"].copy_(state["exp_avg"].to(p.dtype))

    def _ratio(self, new_p, param, pre):
        curr_norm, prev_norm = torch.norm(new_p - pre), torch.norm(param - pre)
        ratio = (curr_norm - prev_norm) / curr_norm
        return torch.nn.functional.hardtanh(ratio, 0.0, 1.0)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                self.step_parameter(p, group, i)

        return loss
