from typing import Iterable

import torch
from torch.nn import Parameter

from modules.model.StableDiffusionXLRModel import StableDiffusionXLRModel
from modules.modelSetup.BaseStableDiffusionXLRSetup import BaseStableDiffusionXLRSetup
from modules.util import create
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs


class StableDiffusionXLRFineTuneSetup(BaseStableDiffusionXLRSetup):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super(StableDiffusionXLRFineTuneSetup, self).__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def create_parameters(
            self,
            model: StableDiffusionXLRModel,
            args: TrainArgs,
    ) -> Iterable[Parameter]:
        params = list()

        if args.text_encoder_train_switch:
            params += list(model.text_encoder_2.parameters())

        if args.unet_train_switch:
            params += list(model.unet.parameters())

        return params

    def create_parameters_for_optimizer(
            self,
            model: StableDiffusionXLRModel,
            args: TrainArgs,
    ) -> Iterable[Parameter] | list[dict]:
        param_groups = list()

        if args.text_encoder_train_switch:
            lr = args.text_encoder_learning_rate if args.text_encoder_learning_rate is not None else args.global_learning_rate
            param_groups.append({
                'params': model.text_encoder_2.parameters(),
                'lr': lr,
                'initial_lr': lr,
            })

        if args.unet_train_switch:
            lr = args.unet_learning_rate if args.unet_learning_rate is not None else args.global_learning_rate
            param_groups.append({
                'params': model.unet.parameters(),
                'lr': lr,
                'initial_lr': lr,
            })

        return param_groups

    def setup_model(
            self,
            model: StableDiffusionXLRModel,
            args: TrainArgs,
    ):

        text_encoder_train_switch = args.text_encoder_train_switch and (model.train_progress.epoch < args.text_encoder_max_train_epochs)
        model.text_encoder_2.requires_grad_(text_encoder_train_switch)

        unet_train_switch = args.unet_train_switch and (model.train_progress.epoch < args.unet_max_train_epochs)
        model.unet.requires_grad_(unet_train_switch)

        model.vae.requires_grad_(False)

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
            model: StableDiffusionXLRModel
    ):
        model.text_encoder_2.to(self.train_device)
        model.vae.to(self.train_device)
        model.unet.to(self.train_device)

        model.text_encoder_2.eval()
        model.vae.eval()
        model.unet.eval()

    def setup_train_device(
            self,
            model: StableDiffusionXLRModel,
            args: TrainArgs,
    ):
        model.text_encoder_2.to(self.train_device if args.text_encoder_train_switch else self.temp_device)
        model.vae.to(self.temp_device)
        model.unet.to(self.train_device)

        if args.text_encoder_train_switch:
            model.text_encoder_2.train()
        else:
            model.text_encoder_2.eval()

        model.vae.train()
        model.unet.train()

    def after_optimizer_step(
            self,
            model: StableDiffusionXLRModel,
            args: TrainArgs,
            train_progress: TrainProgress
    ):
        text_encoder_train_switch = args.text_encoder_train_switch and (model.train_progress.epoch < args.text_encoder_max_train_epochs)
        model.text_encoder_2.requires_grad_(text_encoder_train_switch)

        unet_train_switch = args.unet_train_switch and (model.train_progress.epoch < args.unet_max_train_epochs)
        model.unet.requires_grad_(unet_train_switch)
