from typing import Iterable
import math
import torch


class EMAModuleWrapper:
    def __init__(
            self,
            parameters: Iterable[torch.nn.Parameter],
            decay: float = 0.9999,
            update_step_interval: int = 1,
            device: torch.device | None = None,
            ema_type: str = "CSMA",  # EMA, CSMA
    ):
        parameters = list(parameters)
        self.ema_parameters = [p.clone().detach().to(device) for p in parameters]

        self.temp_stored_parameters = None

        self.decay = decay
        self.update_step_interval = update_step_interval
        self.device = device
        self.ema_type = ema_type


        # TODO: add an automatic decay calculation based on this formula:
        # The impact of the last n steps can be calculated as:
        #     impact = 1-(decay^n)
        # The number of steps needed to reach a specific impact is:
        #     n = log_decay(1-impact)
        # The decay needed to reach a specific impact after n steps is:
        #     decay = (1-impact)^(1/n)
        
        # Possible solution for this below
        
    def auto_tune_decay(self, desired_impact: float, steps: int):
        # Find the optimal decay.
        optimal_decay = math.pow(1 - desired_impact, 1 / steps)
        self.decay = optimal_decay
        
        # Calculate steps to reach impact
        needed_steps = math.log(1 - desired_impact) / math.log(self.decay)
        return {"Optimal Decay": optimal_decay, "Needed Steps for Impact": needed_steps}


    def get_current_decay(self, optimization_step) -> float:
        return min(
            (1 + optimization_step) / (10 + optimization_step),
            self.decay
        )

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter], optimization_step):
        parameters = list(parameters)

        one_minus_decay = 1 - self.get_current_decay(optimization_step)
               
        if (optimization_step + 1) % self.update_step_interval == 0:
            for ema_parameter, parameter in zip(self.ema_parameters, parameters):
                if parameter.requires_grad:
                    #Traditional EMA
                    if self.ema_type == "EMA":
                        parameter_copy = parameter if ema_parameter.device == parameter.device else parameter.detach().to(ema_parameter.device)

                        ema_parameter.add_(
                            parameter_copy.sub_(ema_parameter).mul_(one_minus_decay * self.update_step_interval)
                        )
                        if ema_parameter.device != parameter.device: del parameter_copy
                    
                    # Cosine Similarity Moving Average        
                    elif self.ema_type == "CSMA":
                        sim = torch.nn.CosineSimilarity(dim=0)
                        sims = torch.tensor([], dtype=torch.float32, device=ema_parameter.device)

                        parameter_copy = parameter if ema_parameter.device == parameter.device else parameter.detach().to(ema_parameter.device)
                        
                        cosine_similarity = sim(ema_parameter.to(torch.float32), parameter_copy.to(torch.float32))
                        sims = torch.cat((sims, cosine_similarity.unsqueeze(0)), dim=0)
                        sims = sims[~torch.isnan(sims)]

                        if sims.numel() > 0:
                            min_val, max_val = sims.min(), sims.max()
                            if min_val != max_val:
                                k = (cosine_similarity - min_val) / (max_val - min_val)
                                k = k - (one_minus_decay * self.update_step_interval)
                                k = k.clamp(min=0.0, max=1.0)

                                ema_parameter.mul_(1 - k).add_(parameter_copy * k)
                        
                        if ema_parameter.device != parameter.device: del parameter_copy
    
    # Apply EMA parameters to the provided model by replacing its parameters.    
    def apply_ema_to_model(self, parameters: Iterable[torch.nn.Parameter]):
        for ema_parameter, model_parameter in zip(self.ema_parameters, parameters):
            model_parameter.data.copy_(ema_parameter.data)
            
    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> None:
        self.device = device
        self.ema_parameters = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.ema_parameters
        ]

    def copy_ema_to(self, parameters: Iterable[torch.nn.Parameter], store_temp: bool = True) -> None:
        if store_temp:
            self.temp_stored_parameters = [parameter.detach().cpu() for parameter in parameters]

        parameters = list(parameters)
        for ema_parameter, parameter in zip(self.ema_parameters, parameters):
            parameter.data.copy_(ema_parameter.to(parameter.device).data)

    def copy_temp_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        for temp_parameter, parameter in zip(self.temp_stored_parameters, parameters):
            parameter.data.copy_(temp_parameter.data)

        self.temp_stored_parameters = None

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = self.decay if self.decay else state_dict.get("decay", self.decay)
        self.ema_parameters = state_dict.get("ema_parameters", None)
        self.to(self.device)

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "ema_parameters": self.ema_parameters,
        }
