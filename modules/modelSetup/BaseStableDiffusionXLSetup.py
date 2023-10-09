from abc import ABCMeta

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor, XFormersAttnProcessor, AttnProcessor2_0
from diffusers.utils import is_xformers_available
from torch import Tensor

from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelSetup.BaseDiffusionModelSetup import BaseDiffusionModelSetup
from modules.modelSetup.stableDiffusion.checkpointing_util import \
    enable_checkpointing_for_transformer_blocks, enable_checkpointing_for_clip_encoder_layers
from modules.util import loss_util
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs
from modules.util.enum.AttentionMechanism import AttentionMechanism


class BaseStableDiffusionXLSetup(BaseDiffusionModelSetup, metaclass=ABCMeta):

    def setup_optimizations(
            self,
            model: StableDiffusionXLModel,
            args: TrainArgs,
    ):
        if args.attention_mechanism == AttentionMechanism.DEFAULT:
            model.unet.set_attn_processor(AttnProcessor())
            pass
        elif args.attention_mechanism == AttentionMechanism.XFORMERS and is_xformers_available():
            try:
                model.unet.set_attn_processor(XFormersAttnProcessor())
                model.vae.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(
                    "Could not enable memory efficient attention. Make sure xformers is installed"
                    f" correctly and a GPU is available: {e}"
                )
        elif args.attention_mechanism == AttentionMechanism.SDP:
            model.unet.set_attn_processor(AttnProcessor2_0())

            if is_xformers_available():
                try:
                    model.vae.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    print(
                        "Could not enable memory efficient attention. Make sure xformers is installed"
                        f" correctly and a GPU is available: {e}"
                    )

        if args.gradient_checkpointing:
            model.unet.enable_gradient_checkpointing()
            enable_checkpointing_for_transformer_blocks(model.unet)
            enable_checkpointing_for_clip_encoder_layers(model.text_encoder_1)
            enable_checkpointing_for_clip_encoder_layers(model.text_encoder_2)

    def predict(
            self,
            model: StableDiffusionXLModel,
            batch: dict,
            args: TrainArgs,
            train_progress: TrainProgress
    ) -> dict:
        vae_scaling_factor = model.vae.config['scaling_factor']
        model.noise_scheduler.set_timesteps(model.noise_scheduler.config['num_train_timesteps'])

        latent_image = batch['latent_image']
        scaled_latent_image = latent_image * vae_scaling_factor

        scaled_latent_conditioning_image = None
        if args.model_type.has_conditioning_image_input():
            scaled_latent_conditioning_image = batch['latent_conditioning_image'] * vae_scaling_factor

        generator = torch.Generator(device=args.train_device)
        generator.manual_seed(train_progress.global_step)

        latent_noise = self.create_noise(scaled_latent_image, args, generator)

        timestep = torch.randint(
            low=0,
            high=int(model.noise_scheduler.config['num_train_timesteps'] * args.max_noising_strength),
            size=(scaled_latent_image.shape[0],),
            generator=generator,
            device=scaled_latent_image.device,
        ).long()

        scaled_noisy_latent_image = model.noise_scheduler.add_noise(
            original_samples=scaled_latent_image, noise=latent_noise, timesteps=timestep
        )

        text_encoder_1_output = model.text_encoder_1(
            batch['tokens_1'], output_hidden_states=True, return_dict=True
        )
        text_encoder_1_output = text_encoder_1_output.hidden_states[-2]

        text_encoder_2_output = model.text_encoder_2(
            batch['tokens_2'], output_hidden_states=True, return_dict=True
        )
        pooled_text_encoder_2_output = text_encoder_2_output.text_embeds
        text_encoder_2_output = text_encoder_2_output.hidden_states[-2]

        text_encoder_output = torch.concat([text_encoder_1_output, text_encoder_2_output], dim=-1)

        # original size of the image
        original_height = batch['original_resolution'][0]
        original_width = batch['original_resolution'][1]
        crops_coords_top = batch['crop_offset'][0]
        crops_coords_left = batch['crop_offset'][1]
        target_height = batch['crop_resolution'][0]
        target_width = batch['crop_resolution'][1]

        add_time_ids = torch.stack([
            original_height,
            original_width,
            crops_coords_top,
            crops_coords_left,
            target_height,
            target_width
        ], dim=1)

        add_time_ids = add_time_ids.to(
            dtype=scaled_noisy_latent_image.dtype,
            device=scaled_noisy_latent_image.device,
        )

        if args.model_type.has_mask_input() and args.model_type.has_conditioning_image_input():
            latent_input = torch.concat(
                [scaled_noisy_latent_image, batch['latent_mask'], scaled_latent_conditioning_image], 1
            )
        else:
            latent_input = scaled_noisy_latent_image

        added_cond_kwargs = {"text_embeds": pooled_text_encoder_2_output, "time_ids": add_time_ids}
        predicted_latent_noise = model.unet(
            sample=latent_input,
            timestep=timestep,
            encoder_hidden_states=text_encoder_output,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        model_output_data = {}

        if model.noise_scheduler.config.prediction_type == 'epsilon':
            model_output_data = {
                'predicted': predicted_latent_noise,
                'target': latent_noise,
                'timesteps': timestep,
                'latents': scaled_latent_image,
                'noisy_latents': latent_input,
            }
        elif model.noise_scheduler.config.prediction_type == 'v_prediction':
            target_velocity = model.noise_scheduler.get_velocity(scaled_latent_image, latent_noise, timestep)
            model_output_data = {
                'predicted': predicted_latent_noise,
                'target': target_velocity,
                'timesteps': timestep,
                'latents': scaled_latent_image,
                'noisy_latents': latent_input,
            }

        if args.debug_mode:
            with torch.no_grad():
                # noise
                self.save_image(
                    self.project_latent_to_image(latent_noise).clamp(-1, 1),
                    args.debug_dir + "/training_batches",
                    "1-noise",
                    train_progress.global_step
                )

                # predicted noise
                self.save_image(
                    self.project_latent_to_image(predicted_latent_noise).clamp(-1, 1),
                    args.debug_dir + "/training_batches",
                    "2-predicted_noise",
                    train_progress.global_step
                )

                # noisy image
                self.save_image(
                    self.project_latent_to_image(scaled_noisy_latent_image).clamp(-1, 1),
                    args.debug_dir + "/training_batches",
                    "3-noisy_image",
                    train_progress.global_step
                )

                # predicted image
                alphas_cumprod = model.noise_scheduler.alphas_cumprod.to(args.train_device)
                sqrt_alpha_prod = alphas_cumprod[timestep] ** 0.5
                sqrt_alpha_prod = sqrt_alpha_prod.flatten().reshape(-1, 1, 1, 1)

                sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timestep]) ** 0.5
                sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten().reshape(-1, 1, 1, 1)

                scaled_predicted_latent_image = \
                    (scaled_noisy_latent_image - predicted_latent_noise * sqrt_one_minus_alpha_prod) \
                    / sqrt_alpha_prod
                self.save_image(
                    self.project_latent_to_image(scaled_predicted_latent_image).clamp(-1, 1),
                    args.debug_dir + "/training_batches",
                    "4-predicted_image",
                    model.train_progress.global_step
                )

                # image
                self.save_image(
                    self.project_latent_to_image(scaled_latent_image).clamp(-1, 1),
                    args.debug_dir + "/training_batches",
                    "5-image",
                    model.train_progress.global_step
                )

        return model_output_data

    def apply_snr_weight(self, loss, noisy_latents, latents, gamma):
        gamma = gamma
        if gamma:
            sigma = torch.sub(noisy_latents, latents)
            zeros = torch.zeros_like(sigma) 
            alpha_mean_sq = torch.nn.functional.mse_loss(latents.float(), zeros.float(), reduction="none").mean([1, 2, 3])
            sigma_mean_sq = torch.nn.functional.mse_loss(sigma.float(), zeros.float(), reduction="none").mean([1, 2, 3])
            snr = torch.div(alpha_mean_sq, sigma_mean_sq)
            gamma_over_snr = torch.div(torch.ones_like(snr) * gamma, snr)
            snr_weight = torch.minimum(gamma_over_snr, torch.ones_like(gamma_over_snr)).float()
            loss = loss * snr_weight
        return loss
              
    def apply_snr_weight_to_loss(self, loss, timesteps, gamma):
        snr_values = torch.stack([self.noise_scheduler.all_snr[t] for t in timesteps])
        gamma_over_snr = gamma / snr_values
        snr_weight = torch.clamp(gamma_over_snr, max=1).to(loss.device)
        return loss * snr_weight

    def scale_loss_by_noise(self, loss, timesteps):
        scale = self.get_snr_scale(timesteps)
        return loss * scale

    def add_v_prediction_loss(self, loss, timesteps, v_pred_like_loss):
        scale = self.get_snr_scale(timesteps)
        adjusted_loss = loss + (loss / scale) * v_pred_like_loss
        return adjusted_loss


    def calculate_loss(
            self,
            model: StableDiffusionXLModel,
            batch: dict,
            data: dict,
            args: TrainArgs,
            sub_step: int = 0
    ) -> Tensor:
        predicted = data['predicted']
        target = data['target']
        timesteps = data['timesteps']
        latents = data['latents']
        noisy_latents = data['noisy_latents']
        batch_size = predicted.shape[0]
        
        #Defining Stregths to facilitate UI integration
        mse_strength = 0.5
        mae_strength = 0.5
        cosine_strength = 0.2
        mae_losses, mse_losses, cosine_sim_losses = 0, 0, 0

        # TODO: don't disable masked loss functions when has_conditioning_image_input is true.
        #  This breaks if only the VAE is trained, but was loaded from an inpainting checkpoint
        if args.masked_training and not args.model_type.has_conditioning_image_input():
            losses = loss_util.masked_loss(
                F.mse_loss,
                predicted,
                target,
                batch['latent_mask'],
                args.unmasked_weight,
                args.normalize_masked_area_loss
            ).mean([1, 2, 3])
        else:
        
            #MSE/L2 Loss
            if mse_strength != 0:
                mse_losses = F.mse_loss(
                    predicted,
                    target,
                    reduction='none'
                )
            
            #MAE/L1 Loss
            if mae_strength != 0:
                mae_losses = F.l1_loss(
                    predicted, 
                    target, 
                    reduction='none'
                )
            
            #Cosine similarity loss
            if cosine_strength != 0:
                predicted_normalized = F.normalize(predicted, p=2, dim=1)
                target_normalized = F.normalize(target, p=2, dim=1)
                cosine_sim = torch.nn.functional.cosine_similarity(predicted_normalized, target_normalized, dim=1)
                adjusted_cosine_sim = (1 + cosine_sim) / 2
                cosine_sim_losses = 1 - adjusted_cosine_sim
                cosine_sim_losses = cosine_sim_losses.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            
            #Set Losses Strength (Should probably sum to 1)
            losses = (
                mse_strength * mse_losses +          
                mae_strength * mae_losses
            ).mean([1, 2, 3])
            # losses = self.apply_snr_weight(losses, noisy_latents, latents, 5)
            losses = losses * batch_size
            # crazy scaled loss
            # losses = (
                # mse_strength * mse_losses +          
                # mae_strength * mae_losses
            # ) * (1+cosine_sim_losses) * batch_size
            
            # if model.noise_scheduler.config.prediction_type == 'v_prediction':            
                # if 1 == 0: # args.min_snr_gamma:
                    # losses = self.apply_snr_weight_to_loss(losses, timesteps, predicted, args.min_snr_gamma)
                # if 1 == 1: # args.scale_v_snr_loss:
                    # losses = self.scale_loss_by_noise(losses, timesteps, predicted)
                # if 1 == 1: # args.v_pred_like_loss
                    # losses = self.add_v_prediction_loss(losses, timesteps, predicted, 0.1) # args.v_pred_like_loss)            
            if args.normalize_masked_area_loss:
                clamped_mask = torch.clamp(batch['latent_mask'], args.unmasked_weight, 1)
                losses = losses / clamped_mask.mean(dim=(1, 2, 3))

        return losses.mean()
