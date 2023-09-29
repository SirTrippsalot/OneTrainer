from typing import Iterable

import torch
from torch.nn import Parameter

from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelSetup.BaseStableDiffusionXLSetup import BaseStableDiffusionXLSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import create
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs


class StableDiffusionXLLoRASetup(BaseStableDiffusionXLSetup):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super(StableDiffusionXLLoRASetup, self).__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def create_parameters(
            self,
            model: StableDiffusionXLModel,
            args: TrainArgs,
    ) -> Iterable[Parameter]:
        params = list()

        if args.text_encoder_train_switch:
            params += list(model.text_encoder_1_lora.parameters())
            params += list(model.text_encoder_2_lora.parameters())

        if args.unet_train_switch:
            params += list(model.unet_lora.parameters())

        return params

    def create_parameters_for_optimizer(
            self,
            model: StableDiffusionXLModel,
            args: TrainArgs,
    ) -> Iterable[Parameter] | list[dict]:
        param_groups = list()

        if args.text_encoder_train_switch:
            lr = args.text_encoder_learning_rate if args.text_encoder_learning_rate is not None else args.global_learning_rate
            param_groups.append({
                'params': model.text_encoder_1_lora.parameters(),
                'lr': lr,
                'initial_lr': lr,
            })
            param_groups.append({
                'params': model.text_encoder_2_lora.parameters(),
                'lr': lr,
                'initial_lr': lr,
            })

        if args.unet_train_switch:
            lr = args.unet_learning_rate if args.unet_learning_rate is not None else args.global_learning_rate
            param_groups.append({
                'params': model.unet_lora.parameters(),
                'lr': lr,
                'initial_lr': lr,
            })

        return param_groups

    def setup_model(
            self,
            model: StableDiffusionXLModel,
            args: TrainArgs,
    ):
        if model.text_encoder_1_lora is None and args.text_encoder_train_switch:
            model.text_encoder_1_lora = LoRAModuleWrapper(
                model.text_encoder_1, args.lora_rank, "lora_te1", args.lora_alpha
            )

        if model.text_encoder_2_lora is None and args.text_encoder_train_switch:
            model.text_encoder_2_lora = LoRAModuleWrapper(
                model.text_encoder_2, args.lora_rank, "lora_te2", args.lora_alpha
            )

        if model.unet_lora is None and args.unet_train_switch:
            model.unet_lora = LoRAModuleWrapper(
                model.unet, args.lora_rank, "lora_unet", args.lora_alpha, ["attentions"]
            )

        model.text_encoder_1.requires_grad_(False)
        model.text_encoder_2.requires_grad_(False)
        model.unet.requires_grad_(False)
        model.vae.requires_grad_(False)

        text_encoder_train_switch = args.text_encoder_train_switch and (model.train_progress.epoch < args.text_encoder_max_train_epochs)
        if model.text_encoder_1_lora is not None:
            model.text_encoder_1_lora.requires_grad_(text_encoder_train_switch)
        if model.text_encoder_2_lora is not None:
            model.text_encoder_2_lora.requires_grad_(text_encoder_train_switch)

        unet_train_switch = args.unet_train_switch and (model.train_progress.epoch < args.unet_max_train_epochs)
        if model.unet_lora is not None:
            model.unet_lora.requires_grad_(unet_train_switch)

        if model.text_encoder_1_lora is not None:
            model.text_encoder_1_lora.hook_to_module()
            model.text_encoder_1_lora.to(dtype=args.lora_weight_dtype.torch_dtype())
        if model.text_encoder_2_lora is not None:
            model.text_encoder_2_lora.hook_to_module()
            model.text_encoder_2_lora.to(dtype=args.lora_weight_dtype.torch_dtype())
        if model.unet_lora is not None:
            model.unet_lora.hook_to_module()
            model.unet_lora.to(dtype=args.lora_weight_dtype.torch_dtype())

        if args.rescale_noise_scheduler_to_zero_terminal_snr:
            model.rescale_noise_scheduler_to_zero_terminal_snr()
            model.force_v_prediction()

        model.optimizer = create.create_optimizer(
            self.create_parameters_for_optimizer(model, args), model.optimizer_state_dict, args
        )
        del model.optimizer_state_dict

        model.ema = create.create_ema(
            self.create_parameters(model, args), model.ema_state_dict, args
        )
        del model.ema_state_dict

        self.setup_optimizations(model, args)

    def setup_eval_device(
            self,
            model: StableDiffusionXLModel
    ):
        model.text_encoder_1.to(self.train_device)
        model.text_encoder_2.to(self.train_device)
        model.vae.to(self.train_device)
        model.unet.to(self.train_device)

        if model.text_encoder_1_lora is not None:
            model.text_encoder_1_lora.to(self.train_device)

        if model.text_encoder_2_lora is not None:
            model.text_encoder_2_lora.to(self.train_device)

        if model.unet_lora is not None:
            model.unet_lora.to(self.train_device)

        model.text_encoder_1.eval()
        model.text_encoder_2.eval()
        model.vae.eval()
        model.unet.eval()

    def setup_train_device(
            self,
            model: StableDiffusionXLModel,
            args: TrainArgs,
    ):
        model.text_encoder_1.to(self.train_device if args.text_encoder_train_switch else self.temp_device)
        model.text_encoder_2.to(self.train_device if args.text_encoder_train_switch else self.temp_device)
        model.vae.to(self.temp_device)
        model.unet.to(self.train_device)

        if model.text_encoder_1_lora is not None and args.text_encoder_train_switch:
            model.text_encoder_1_lora.to(self.train_device)

        if model.text_encoder_2_lora is not None and args.text_encoder_train_switch:
            model.text_encoder_2_lora.to(self.train_device)

        if model.unet_lora is not None:
            model.unet_lora.to(self.train_device)

        if args.text_encoder_train_switch:
            model.text_encoder_1.train()
            model.text_encoder_2.train()
        else:
            model.text_encoder_1.eval()
            model.text_encoder_2.eval()
        model.vae.eval()
        if args.unet_train_switch:
            model.unet.train()
        else:
            model.unet.eval()

    def after_optimizer_step(
            self,
            model: StableDiffusionXLModel,
            args: TrainArgs,
            train_progress: TrainProgress
    ):
        text_encoder_train_switch = args.text_encoder_train_switch and (model.train_progress.epoch < args.text_encoder_max_train_epochs)
        if model.text_encoder_1_lora is not None:
            model.text_encoder_1_lora.requires_grad_(text_encoder_train_switch)
        if model.text_encoder_2_lora is not None:
            model.text_encoder_2_lora.requires_grad_(text_encoder_train_switch)

        unet_train_switch = args.unet_train_switch and (model.train_progress.epoch < args.unet_max_train_epochs)
        if model.unet_lora is not None:
            model.unet_lora.requires_grad_(unet_train_switch)
