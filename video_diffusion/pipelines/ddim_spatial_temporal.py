# code mostly taken from https://github.com/huggingface/diffusers
import inspect
from typing import Callable, List, Optional, Union
import PIL
import torch
import numpy as np
from einops import rearrange
from tqdm import tqdm
import os, time
import cv2
from PIL import Image
from diffusers.utils import deprecate, logging
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
import imageio.v2 as imageio  # 需有 ffmpeg，環境通常已內建；沒有就 pip install imageio-ffmpeg



from .stable_diffusion import SpatioTemporalStableDiffusionPipeline
from diffusers.models import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

import torch.nn.functional as F
from omegaconf import OmegaConf
from video_diffusion.prompt_attention.attention_register import register_attention_control
from video_diffusion.prompt_attention.attention_util import ST_Layout_Attn_Control,ST_Layout_Attn_ControlEdit,Attention_Record_Processor
from video_diffusion.prompt_attention import attention_util
from video_diffusion.prompt_attention.sd_study_utils import *
from video_diffusion.prompt_attention.attention_store import AttentionStore
from video_diffusion.common.image_util import save_gif_mp4_folder_type

from PIL import Image
from einops import rearrange
from ..models.controlnet3d import ControlNetModel
from ..models.unet_3d_condition import UNetPseudo3DConditionModel

from diffusers.schedulers import (
    DDIMScheduler,
    DDIMInverseScheduler,
)
import os
import nltk


nltk.download('punkt')

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class DDIMSpatioTemporalStableDiffusionPipeline(SpatioTemporalStableDiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using Spatio-Temporal Stable Diffusion.
    """
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNetPseudo3DConditionModel,
        controlnet: ControlNetModel,
        scheduler: DDIMScheduler,
        inverse_scheduler: DDIMInverseScheduler,
        disk_store: bool=False,
        logdir=None,
        ):
        super().__init__(vae, text_encoder, tokenizer, unet, controlnet, scheduler,inverse_scheduler)
        self.store_controller = attention_util.AttentionStore(disk_store=disk_store)
        self.logdir=logdir

    r"""
    Pipeline for text-to-video generation using Spatio-Temporal Stable Diffusion.
    """


    def check_inputs(self, prompt, height, width, callback_steps, strength=None):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        if strength is not None:
            if strength <= 0 or strength > 1:
                raise ValueError(f"The value of strength should in (0.0, 1.0] but is {strength}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )


    @torch.no_grad()
    def prepare_source_latents(self, image, batch_size, num_images_per_prompt, 
                                    #   dtype, device, 
                                      text_embeddings,
                                      generator=None): 
        
        # Not sure if image need to change device and type
        # image = image.to(device=device, dtype=dtype)
        print("generator is list:",isinstance(generator, list))
        batch_size = batch_size * num_images_per_prompt
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if isinstance(generator, list):
            init_latents = [
                self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            ## org is
            #init_latents = self.vae.encode(image).latent_dist.sample(generator)
            init_latents = self.vae.encode(image).latent_dist.mean
        init_latents = 0.18215 * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        # get latents
        init_latents_bcfhw = rearrange(init_latents, "(b f) c h w -> b c f h w", b=batch_size)
        return init_latents_bcfhw


    def prepare_latents_ddim_inverted(self, image, batch_size, 
                                      source_prompt,
                                      do_classifier_free_guidance,
                                      control = None,
                                      controlnet_conditioning_scale=None,
                                      use_pnp=None,
                                      cluster_inversion_feature = None,
                                      **kwargs,
                                      ): 
        weight_dtype = image.dtype
        device = self._execution_device
        print('device',device)
        timesteps = self.scheduler.timesteps
        saved_features0 = []
        saved_features1 = []
        saved_features2 = []
        saved_q4 = []
        saved_k4 = []
        saved_q5 = []
        saved_k5 = []
        saved_q6 = []
        saved_k6 = []
        saved_q7 = []
        saved_k7 = []
        saved_q8 = []
        saved_k8 = []
        saved_q9 = []
        saved_k9 = []
        #ddim inverse
        num_inverse_steps = 50
        self.inverse_scheduler.set_timesteps(num_inverse_steps, device=device)
        inverse_timesteps, num_inverse_steps = self.get_inverse_timesteps(num_inverse_steps, 1, device)
        num_warmup_steps = len(inverse_timesteps) - num_inverse_steps * self.inverse_scheduler.order

        #============ddim inversion==========*
        prompt_embeds = self._encode_prompt(
            source_prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=None,
        )

        latents = self.prepare_video_latents(image, batch_size, self.unet.dtype, device)

        bz, c, clip_length, downsample_height, downsample_width = latents.shape
        del self.store_controller
        self.store_controller = attention_util.AttentionStore()
        attention_maps_list = []
        self_attention_maps_list = []
        cond_embeddings_list = []
        
        editor = Attention_Record_Processor(additional_attention_store=self.store_controller)
        attention_util.register_attention_control(self, editor, prompt_embeds, clip_length,downsample_height,downsample_width,ddim_inversion=True)


        with self.progress_bar(total=num_inverse_steps-1) as progress_bar:
            for i, t in enumerate(inverse_timesteps[1:]):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.inverse_scheduler.scale_model_input(latent_model_input, t)


                down_block_res_samples, mid_block_res_sample = self.controlnet(latent_model_input, t, encoder_hidden_states=prompt_embeds,controlnet_cond=control,return_dict=False)
                down_block_res_samples = [
                    down_block_res_sample * controlnet_conditioning_scale
                    for down_block_res_sample in down_block_res_samples
                ]
                mid_block_res_sample *= controlnet_conditioning_scale
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    **kwargs,
                ).sample 
                if use_pnp and t.cpu() in timesteps:
                    saved_features0.append(self.unet.up_blocks[1].resnets[0].out_layers_features.cpu())
                    saved_features1.append(self.unet.up_blocks[1].resnets[1].out_layers_features.cpu())
                    saved_features2.append(self.unet.up_blocks[2].resnets[0].out_layers_features.cpu())
                    saved_q4.append(self.unet.up_blocks[1].attentions[1].transformer_blocks[0].attn1.q.cpu())
                    saved_k4.append(self.unet.up_blocks[1].attentions[1].transformer_blocks[0].attn1.k.cpu())
                    saved_q5.append(self.unet.up_blocks[1].attentions[2].transformer_blocks[0].attn1.q.cpu())
                    saved_k5.append(self.unet.up_blocks[1].attentions[2].transformer_blocks[0].attn1.k.cpu())
                    saved_q6.append(self.unet.up_blocks[2].attentions[0].transformer_blocks[0].attn1.q.cpu())
                    saved_k6.append(self.unet.up_blocks[2].attentions[0].transformer_blocks[0].attn1.k.cpu())
                    saved_q7.append(self.unet.up_blocks[2].attentions[1].transformer_blocks[0].attn1.q.cpu())
                    saved_k7.append(self.unet.up_blocks[2].attentions[1].transformer_blocks[0].attn1.k.cpu())
                    saved_q8.append(self.unet.up_blocks[2].attentions[2].transformer_blocks[0].attn1.q.cpu())
                    saved_k8.append(self.unet.up_blocks[2].attentions[2].transformer_blocks[0].attn1.k.cpu())
                    saved_q9.append(self.unet.up_blocks[3].attentions[0].transformer_blocks[0].attn1.q.cpu())
                    saved_k9.append(self.unet.up_blocks[3].attentions[0].transformer_blocks[0].attn1.k.cpu())


                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + 1 * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.inverse_scheduler.step(noise_pred, t, latents).prev_sample.to(dtype=weight_dtype)
                if i == len(inverse_timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.inverse_scheduler.order == 0):
                    progress_bar.update()
        if use_pnp:
            saved_features0.reverse()
            saved_features1.reverse()
            saved_features2.reverse()
            saved_q4.reverse()
            saved_k4.reverse()
            saved_q5.reverse()
            saved_k5.reverse()
            saved_q6.reverse()
            saved_k6.reverse()
            saved_q7.reverse()
            saved_k7.reverse()
            saved_q8.reverse()
            saved_k8.reverse()
            saved_q9.reverse()
            saved_k9.reverse()

            attn_inversion_dict = {
                'features0': saved_features0, 'features1': saved_features1, 'features2': saved_features2,
                'q4': saved_q4,'k4': saved_k4,'q5': saved_q5,'k5': saved_k5,'q6': saved_q6,'k6': saved_k6,
                'q7': saved_q7,'k7': saved_k7,'q8': saved_q8,'k8': saved_k8,'q9': saved_q9,'k9': saved_k9
            }
        else:
            attn_inversion_dict = None

        if cluster_inversion_feature:
            logger.info('cluster ddim inversion feature')
            inv_self_avg_dict={}
            inv_cross_avg_dict={}
            element_name = 'attn'
            attn_size = 32
            for element_name in ['attn']:
                inv_self_avg_dict[element_name]={}
                inv_cross_avg_dict[element_name]={}

            self_attn_avg = editor.aggregate_attention(from_where=("up", "down", "mid"), 
                                                                    res=attn_size,is_cross=False)

            cross_attn_avg = editor.aggregate_attention(from_where=("up", "down", "mid"), 
                                                                    res=attn_size,is_cross=True)   

            print('self_attn_avg',self_attn_avg.shape)
            print('cross_attn_avg', cross_attn_avg.shape)
            inv_self_avg_dict[element_name][attn_size]=self_attn_avg
            inv_cross_avg_dict[element_name][attn_size]=cross_attn_avg

            os.makedirs(os.path.join(self.logdir, "attn_inv"), exist_ok=True)
            os.makedirs(os.path.join(self.logdir, "sd_study"), exist_ok=True)
            with open(os.path.join(self.logdir, 
                    "attn_inv/inv_self_avg_dict.pkl"), 
                    'wb') as f:
                pkl.dump(inv_self_avg_dict, f)

            with open(os.path.join(self.logdir, 
                    "attn_inv/inv_cross_avg_dict.pkl"), 
                    'wb') as f:
                pkl.dump(inv_cross_avg_dict, f)

            num_segments=3
            draw_pca(inv_self_avg_dict, resolution=32, dict_key='attn', 
                    save_path=os.path.join(self.logdir, 'sd_study'),
                    special_name='inv_self')
        
            run_clusters(inv_self_avg_dict, resolution=32, dict_key='attn', 
                    save_path=os.path.join(self.logdir, 'sd_study'),
                    special_name='inv_self',num_segments=num_segments)

            cross_attn_visualization = attention_util.show_cross_attention_plus_org_img(self.tokenizer, source_prompt, 
                                        image, editor, 32, ["up", "down", "mid"], save_path= os.path.join(self.logdir,'sd_study'),attention_maps=cross_attn_avg)


            dict_key='attn'
            special_name='inv_self'
            resolution = 32
            threshold=0.1
            
            tokenized_prompt = nltk.word_tokenize(source_prompt)
            nouns = [(i, word) for (i, (word, pos)) in enumerate(nltk.pos_tag(tokenized_prompt)) if pos[:2] == 'NN']
            print(nouns)

            npy_name=f'cluster_{dict_key}_{resolution}_{special_name}.npy'
            save_path=os.path.join(self.logdir, 'sd_study')

            abs_filename=os.path.join(self.logdir, "attn_inv", f"inv_cross_avg_dict.pkl")
            inv_cross_avg_dict=read_pkl(abs_filename)

            video_cross_attention = inv_cross_avg_dict['attn'][32]
            video_clusters=np.load(os.path.join(save_path, npy_name))

            t = video_clusters.shape[0]
            for i in range(t):
                clusters = video_clusters[i]
                cross_attention = video_cross_attention[i]
                c2noun, c2mask = cluster2noun_(clusters, threshold, num_segments, nouns,cross_attention)
                print('c2noun',c2noun)
                merged_mask={}
                for index in range(len(c2noun)):    
                    # mask_ = merged_mask[class_name]
                    item=c2noun[index]
                    mask_ = c2mask[index]
                    mask_ = torch.from_numpy(mask_)
                    mask_ = F.interpolate(mask_.float().unsqueeze(0).unsqueeze(0), size=512, mode='nearest').round().bool().squeeze(0).squeeze(0)
                    
                    output_name = os.path.join(f"{save_path}",
                                                f"frame_{i}_{item}_{index}.png")

                    save_mask(mask_,  output_name)
        
        return latents, attn_inversion_dict

    
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start
    
    def get_inverse_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)

        # safety for t_start overflow to prevent empty timsteps slice
        if t_start == 0:
            return self.inverse_scheduler.timesteps, num_inference_steps
        timesteps = self.inverse_scheduler.timesteps[:-t_start]

        return timesteps, num_inference_steps - t_start

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        frames,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        print("self.vae_scale_factor",self.vae_scale_factor)
        shape = (
            batch_size,
            num_channels_latents,
            frames,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(
                    device
                )
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_video_latents(self, frames, batch_size, dtype, device, generator=None):
        if not isinstance(frames, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(frames)}"
            )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if isinstance(generator, list):
            latents = [
                self.vae.encode(frames[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
            ]
            latents = torch.cat(latents, dim=0)
        else:
            latents = self.vae.encode(frames).latent_dist.sample(generator)

        latents = self.vae.config.scaling_factor * latents

        latents = rearrange(latents, "(b f) c h w ->b c f h w", b=batch_size)

        return latents

    def clean_features(self):
        self.unet.up_blocks[1].resnets[0].out_layers_inject_features = None
        self.unet.up_blocks[1].resnets[1].out_layers_inject_features = None
        self.unet.up_blocks[2].resnets[0].out_layers_inject_features = None
        self.unet.up_blocks[1].attentions[1].transformer_blocks[0].attn1.inject_q = None
        self.unet.up_blocks[1].attentions[1].transformer_blocks[0].attn1.inject_k = None
        self.unet.up_blocks[1].attentions[2].transformer_blocks[0].attn1.inject_q = None
        self.unet.up_blocks[1].attentions[2].transformer_blocks[0].attn1.inject_k = None
        self.unet.up_blocks[2].attentions[0].transformer_blocks[0].attn1.inject_q = None
        self.unet.up_blocks[2].attentions[0].transformer_blocks[0].attn1.inject_k = None
        self.unet.up_blocks[2].attentions[1].transformer_blocks[0].attn1.inject_q = None
        self.unet.up_blocks[2].attentions[1].transformer_blocks[0].attn1.inject_k = None
        self.unet.up_blocks[2].attentions[2].transformer_blocks[0].attn1.inject_q = None
        self.unet.up_blocks[2].attentions[2].transformer_blocks[0].attn1.inject_k = None
        self.unet.up_blocks[3].attentions[0].transformer_blocks[0].attn1.inject_q = None
        self.unet.up_blocks[3].attentions[0].transformer_blocks[0].attn1.inject_k = None

    def _get_attention_type(self):
        sub_nets = self.unet.named_children()
        for net in sub_nets:
            if hasattr(net[1], 'children'):
                for net in net[1].named_children():
                    if hasattr(net[1], 'children'):
                        for net in net[1].named_children():
                            if net[1].__class__.__name__ == "SpatioTemporalTransformerModel":
                                for net in net[1].named_children():
                                    if hasattr(net[1], 'children'):
                                       for net in net[1].named_children():
                                            if net[1].__class__.__name__ == "SpatioTemporalTransformerBlock":
                                                for net in net[1].named_children():
                                                    if net[1].__class__.__name__ == "SparseCausalAttention":
                                                        attention_type = "SparseCausalAttention"
                                                    elif net[1].__class__.__name__ == "FullyFrameAttention":
                                                        attention_type = "FullyFrameAttention"
        #print("attention_type:",attention_type)
        return attention_type
    def _to_uint8(self, img01: torch.Tensor) -> np.ndarray:
        """
        img01: [H,W] in {0,1} or [H,W] float in [0,1]
        return uint8 [H,W] in {0,255}
        """
        img01 = img01.clamp(0, 1)
        return (img01 * 255.0).round().to(torch.uint8).cpu().numpy()
    
    def _save_png(self, arr_uint8: np.ndarray, path: str, palette=None):
        """
        Save uint8 2D array as PNG. If palette provided (list of 768 ints),
        will store as 'P' mode (indexed color).
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if palette is None:
            Image.fromarray(arr_uint8).save(path)
        else:
            im = Image.fromarray(arr_uint8, mode="P")
            im.putpalette(palette)
            im.save(path)
    
    def _make_simple_palette(self, num_classes: int) -> list:
        """
        Make a simple color palette for 'P' mode. Repeats if >8, but for你現在的2~3類足夠。
        """
        base = [
            (0,0,0),        # 0: bg - black
            (255,0,0),      # 1: red
            (0,255,0),      # 2: green
            (0,0,255),      # 3: blue
            (255,255,0),    # 4: yellow
            (255,0,255),    # 5: magenta
            (0,255,255),    # 6: cyan
            (255,128,0),    # 7: orange
        ]
        colors = []
        for i in range(num_classes):
            colors.append(base[i % len(base)])
        # palette needs length 256*3; pad with zeros
        flat = []
        for (r,g,b) in colors:
            flat += [r,g,b]
        flat += [0,0,0] * (256 - len(colors))
        return flat
    # [NEW] 找主提示中完整連續匹配的所有區段（回傳 [(j0,j1), ...]）
    def _find_all_spans(self, main_ids: torch.Tensor, sub_ids: torch.Tensor, max_len: int = 77):
        hits = []
        L = int(sub_ids.numel())
        if L <= 0 or L > max_len:
            return hits
        for j in range(max_len - L + 1):
            if (main_ids[j:j+L] == sub_ids).sum() == L:
                hits.append((j, j+L))
        return hits
    
    # [NEW] 互斥：m1' = m1 & ~m2，m2' = m2 & ~m1（簡版）
    def _disjoin_pair(self, m1: torch.Tensor, m2: torch.Tensor):
        return (m1 & (~m2)), (m2 & (~m1))
    def _ensure_dir(self, path: str):
        os.makedirs(path, exist_ok=True)
        return path
    
    def save_labels_memmap(self, labels: torch.Tensor, save_path: str):
        """
        labels: [N] int16/uint16 on CPU/GPU
        save_path: file path (e.g., ./sreg_labels/frame_000.npy or .dat)
        """
        labels_cpu = labels.detach().to(torch.int32).cpu().numpy().astype(np.uint16)
        self._ensure_dir(os.path.dirname(save_path))
        # 使用 memmap 建檔（w+ 會新建/覆蓋；也可用 np.save 但 memmap 更好做大批處理）
        mm = np.memmap(save_path, dtype=np.uint16, mode='w+', shape=(labels_cpu.shape[0],))
        mm[:] = labels_cpu[:]
        del mm  # 關閉映射（確保 flush）
    
    def load_labels_memmap(self, load_path: str) -> torch.Tensor:
        """
        回傳 CPU tensor（呼叫端自行 .to(device)）
        """
        mm = np.memmap(load_path, dtype=np.uint16, mode='r')
        # 這裡 copy() 是為了切斷對 memmap 的長期連結（也可不 copy，視你流程）
        return torch.from_numpy(np.array(mm, copy=True)).to(torch.int32)
    
    def apply_sreg_mask_logits_chunk(
        self,
        logits_chunk: torch.Tensor,   # [Bxh, Q, N]
        q_indices: torch.Tensor,      # [Q]
        key_labels: torch.Tensor,     # [N]
        big_neg: float = -1e3,
        bg_id: int = 0                # [# MOD-APPLY] 我們這版 0 當背景；若你用哨兵，改成 65535
    ):
        """
        僅允許「非背景」且「同類」互看；其他一律加大負偏置。
        """
        device = logits_chunk.device
        key_labels = key_labels.to(device)            # [N]
        q_labels = key_labels[q_indices]              # [Q]
    
        same_class = (q_labels.view(-1, 1) == key_labels.view(1, -1))  # [Q, N] bool
    
        # 不讓背景互看（query/key 若為背景，都視為不合法）
        not_bg_q   = (q_labels.view(-1, 1) != bg_id)
        not_bg_key = (key_labels.view(1, -1) != bg_id)
        valid = same_class & not_bg_q & not_bg_key
    
        mask = (~valid).to(logits_chunk.dtype) * big_neg
        return logits_chunk + mask
    def _downsample_two_instances_to_64x64(self, li_full: torch.Tensor, out_hw=(64, 64), thr: float = 0.5):
        """
        li_full: [2, H, W] 兩個 instance 的 mask（0/1 或機率）
        回傳: li_64: [2, 64, 64]，先 hard 化再 max-pool，保留小面積
        """
        assert li_full.dim() == 3 and li_full.shape[0] == 2, f"Expect [2,H,W], got {tuple(li_full.shape)}"
        # 先 hard 化
        hard = (li_full > thr).float()                    # [2,H,W]
        H, W = hard.shape[-2:]
        oh, ow = out_hw
        # 計算整數縮放比（若 H/W 不是 64 的整數倍，改用 interpolate 最近鄰）
        if H % oh == 0 and W % ow == 0:
            kh, kw = H // oh, W // ow
            li_64 = F.max_pool2d(hard.unsqueeze(0), kernel_size=(kh, kw), stride=(kh, kw)).squeeze(0)  # [2,64,64]
        else:
            # 後備：最近鄰下採樣（相對不如 max-pool 穩，但仍可用）
            li_64 = F.interpolate(hard.unsqueeze(0), size=(oh, ow), mode='nearest').squeeze(0)         # [2,64,64]
        return li_64
    # ==== [REPLACE FROM HERE] ===============================================
    def _excel_letters(self, n: int) -> str:
        """0 -> 'a', 1 -> 'b', ... 25 -> 'z', 26 -> 'aa', ... (lowercase)."""
        s = []
        n += 1
        while n:
            n, r = divmod(n - 1, 26)
            s.append(chr(ord('a') + r))
        return ''.join(reversed(s))
    
    def _gen_identity_names(self, seg_cls: int):
        base = ["S", "B", "BG"]  # fixed first three
        if seg_cls <= len(base):
            return base[:seg_cls]
        # add a, b, c, ... after BG (can go beyond 'z' -> 'aa', 'ab', ...)
        extras = [self._excel_letters(i) for i in range(seg_cls - len(base))]
        return base + extras

    def _prepare_attention_layout(self,bsz,height,width,layouts,prompts,clip_length,attention_type,device):
        ## current layouts  f s c h w
        ## org layouts s c h w

        #print("prompt:",prompts)
        # sp_sz =self.unet.sample_size
        print("============ prepare attention layout ===============")
        sp_sz = height*width
        frames, seg_cls, c, h ,w = layouts.shape
        text_input = self.tokenizer(prompts, padding="max_length", return_length=True, return_overflowing_tokens=False, 
                                    max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        cond_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]

        uncond_input = self.tokenizer([""]*bsz, padding="max_length", max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors="pt")
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]

        for i in range(1,len(prompts)):
            wlen = text_input['length'][i] - 2
            widx = text_input['input_ids'][i][1:1+wlen]
            for j in range(77):
                if (text_input['input_ids'][0][j:j+wlen] == widx).sum() == wlen:
                    break

        ###########################
        ###### prep for sreg ###### 
        ###########################
      
        sreg_maps = {}
        reg_sizes = {}
        reg_sizes_c = {}
        
        # === 🔵 新增：分 class 的 sreg 與尺寸比例 ===
        sreg_maps_class = {}   # key: Q (=h*w) ; val: [bsz, 1, Q, Q, C]  (C = seg_cls or seg_cls+1 with BG)
        reg_sizes_class = {}   # key: Q ; val: [bsz, 1, 1, 1, C]
        # >>> [ADDED]：part_layouts（S-head / B-head）初始化（不改回傳，掛在 self 上）
        # part_masks_by_res = {}   # 結構：self.part_masks_by_res[h*w] = {'S_head': [F,H,W], 'B_head': [F,H,W]}
        
        # >>> [ADDED]：part 的通道定義（確保 0= S_head, 1= B_head）
        # part_id_map = {'S_head': 0, 'B_head': 1}
        # # >>> [ADDED]：基本形狀檢查（允許 part_layouts 缺席）
        # has_part = (part_layouts is not None)
        # if has_part:
        #     try:
        #         frames_p, seg_part_cls, c_p, h_p, w_p = part_layouts.shape
        #         if frames_p != frames:
        #             print(f"[WARN] part_layouts frames({frames_p}) != layouts frames({frames})，後續仍以 layouts 的 frames 為準")
        #     except Exception as e:
        #         print(f"[WARN] part_layouts 形狀解析失敗：{e}")
        #         has_part = False     
                
        device = layouts.device

        frame_index_pre = torch.arange(frames)+(-1)
        frame_index_pre = frame_index_pre.clip(0, frames-1)
        

        id_masks_by_res = {}           
            
        for r in range(4):
            layouts_s_frames = []
            if attention_type == "SparseCausalAttention":
                layouts_s_sparse_attn = []

            h = int(height/np.power(2,r))
            w= int(width/np.power(2,r))
            Q = h*w

            #layouts torch.Size([70, 2, 1, 64, 64])
            # layouts_interpolate = F.interpolate(layouts.squeeze(2), (res, res), mode='nearest').unsqueeze(2)    
            layouts_interpolate_flat = F.interpolate(layouts.squeeze(2), (h, w), mode='nearest').unsqueeze(2)  
            layouts_interpolate = layouts_interpolate_flat.view(frames,seg_cls,1,-1)   ## frames,seg_cls,1,res^2
            # layouts_interp_hw = layouts_interpolate_flat


            # >>> [ADDED]：在「這個解析度」下，建立每幀、每身份的二值 mask
            #  取得 class channel 的 one-hot（因為你的 layouts 看起來已經是 0/1 mask，直接 >0 就好）
            #  shape: [F, S, H*W]
            class_mask_flat = (layouts_interpolate.squeeze(2) > 0).to(torch.float32)        # >>> [ADDED]
            # 展回 [F, S, H, W]
            class_mask_hw = class_mask_flat.view(frames, seg_cls, h, w)                     # >>> [ADDED]
    
             # ---- NEW: generate identity names by channel index order
            id_names = self._gen_identity_names(seg_cls)  # e.g., ['S','B','BG','a','b',...]
            id_dict = {}
        
            # Fill per-identity masks: each entry -> list of [H,W] per frame (no extra math)
            # If seg_cls < len(id_names) (shouldn't happen), missing ones default to zeros.
            for cidx, name in enumerate(id_names):
                if cidx < seg_cls:
                    stack = class_mask_hw[:, cidx]  # [F, h, w]
                else:
                    stack = torch.zeros((frames, h, w), device=class_mask_hw.device)
                id_dict[name] = [stack[t].contiguous() for t in range(frames)]
        
            # Store by resolution key Q
            id_masks_by_res[Q] = id_dict
            print("[DBG] id_masks_by_res keys:", list(id_masks_by_res.keys()))
            for k, v in id_masks_by_res.items():
                print(f"  {k}: type={type(v)}")
                if isinstance(v, dict):
                    print("    subkeys:", list(v.keys()))
                    for subk, subv in v.items():
                        if isinstance(subv, torch.Tensor):
                            print(f"      {subk}: shape={tuple(subv.shape)} dtype={subv.dtype}")
                        else:
                            print(f"      {subk}: type={type(subv)}")
            # if has_part:
            #     # 以與 layouts 相同的解析度 (h, w) 做最近鄰下採樣
            #     # part_layouts: [F, S_part, 1, H, W]  → squeeze(2)→ [F, S_part, H, W]
            #     part_interp_flat = F.interpolate(part_layouts.squeeze(2), (h, w), mode='nearest').unsqueeze(2)  # [F,S_part,1,h,w]
            #     # 二值化後展回 [F, S_part, h, w]
            #     part_mask_hw = (part_interp_flat.squeeze(2) > 0).to(torch.float32).view(-1, seg_part_cls, h, w)  # [F,S_part,h,w]
            
            #     # 取出 S-head / B-head；若通道不足則回退為 0
            #     if seg_part_cls > part_id_map['S_head']:
            #         S_head_stack = part_mask_hw[:, part_id_map['S_head']]     # [F,h,w]
            #     else:
            #         S_head_stack = torch.zeros((frames, h, w), device=part_mask_hw.device)
            
            #     if seg_part_cls > part_id_map['B_head']:
            #         B_head_stack = part_mask_hw[:, part_id_map['B_head']]     # [F,h,w]
            #     else:
            #         B_head_stack = torch.zeros((frames, h, w), device=part_mask_hw.device)
            
            #     # 存到 self.part_masks_by_res[h*w]
            #     part_masks_by_res[h * w] = {
            #         'S_head': [S_head_stack[t].contiguous() for t in range(frames)],
            #         'B_head': [B_head_stack[t].contiguous() for t in range(frames)],
            #     }
            # ===================== /part_layouts ======================

        
            ### implementation of sparse casual attn and fully frame attn
            for i in range(frames):
                #layouts_f = layouts[i]

                layouts_s = layouts_interpolate[i]

                if attention_type == "SparseCausalAttention":
                
                    ### prepare for SparseCausalAttention query, key/value
                    query= layouts_s
                    query = query.view(query.size(0),-1,1).to(device)  ### segcls,res^2,1  #[cls, 4096, 1]
        
                    ### key should be segcls,1,2xres^2
                    key = torch.cat((layouts_interpolate[0],layouts_interpolate[frame_index_pre[i]]),dim=-1).to(device)
                    #([cls, 1, 8192])

                    layouts_s_cross_frame_attn= (query * key).sum(0).unsqueeze(0).repeat(bsz,1,1)  ## 1,4096,8192

                    layouts_s_sparse_attn.append(layouts_s_cross_frame_attn)
                
                layouts_s = (layouts_s.view(layouts_s.size(0),1,-1)*layouts_s.view(layouts_s.size(0),-1,1)).sum(0).unsqueeze(0).repeat(bsz,1,1)

                layouts_s_frames.append(layouts_s)


            layouts_s_frames = torch.stack(layouts_s_frames,dim=0)
            if attention_type == "SparseCausalAttention":
                layouts_s_sparse_attn = torch.stack(layouts_s_sparse_attn,dim=0)
                sreg_maps[h*w] = layouts_s_sparse_attn
                reg_sizes[h*w] = 1-1.*layouts_s_frames.sum(-1, keepdim=True)/(np.power(clip_length, 2))
                reg_sizes_c[h*w]  = 1-1.*layouts_s_frames.sum(-1, keepdim=True)/(np.power(clip_length, 2))
            #### code for check error#####
            # num_nonzero = torch.count_nonzero(layouts_s_frames)
            # print("num_nonzero",num_nonzero)
            # print("layouts_s_frames",layouts_s_frames.shape)
            # print("layouts_s_frames",layouts_s_frames)
            # print("reg_size final shape:", (1-1.*layouts_s_frames.sum(-1, keepdim=True)/(np.power(res, 2))).shape)
            # print("reg_size", (1-1.*layouts_s_frames.sum(-1, keepdim=True)/(np.power(res, 2))))
            #### code for check error#####


            #print("layouts_s",layouts_s.shape)

            #print("layouts_s.view(layouts_s.size(0),-1,1)",*layouts_s.view(layouts_s.size(0),-1,1).shape)

            if attention_type == "FullyFrameAttention":
                layouts_s= rearrange(layouts_interpolate,"f s c res -> s c (f res)")
                if r==0:
                    layout_s = None
                    reg_sizes[h*w] = None
                    sreg_maps[h*w] = None
                    reg_sizes_c[h*w] = None
                else:
                    layouts_s = (layouts_s*layouts_s.view(layouts_s.size(0),-1,1)).sum(0).unsqueeze(0).repeat(bsz,1,1).to(torch.float16)
                    sreg_maps[h*w] = layouts_s
                    reg_sizes[h*w] = 1-1.*layouts_s.sum(-1, keepdim=True)/((h*clip_length)*(w*clip_length))
                    reg_sizes_c[h*w]  =  1-1.*layouts_s_frames.sum(-1, keepdim=True)/(h*w)
                #print("layouts_s",layouts_s.shape)
                # if res == 64:
                #     reg_sizes[np.power(res, 2)] = None
                # else:
                #     reg_sizes[np.power(res, 2)] = 1-1.*layouts_s.sum(-1, keepdim=True)/(np.power(res*clip_length, 2))
                # #sreg_maps[np.power(res, 2)] = layouts_s_frames
                # sreg_maps[np.power(res, 2)] = layouts_s
                # reg_sizes_c[np.power(res, 2)]  =  1-1.*layouts_s_frames.sum(-1, keepdim=True)/(np.power(res, 2))
            # =====================================================================
            # === 🔵 新增：分 class 的 sreg（含 BG） -> sreg_maps_class / reg_sizes_class ===
            # 目標：得到 [bsz, 1, Q, Q, C] 以及 [bsz, 1, 1, 1, C]
            # 做法（省顯存）：
            #   1) 先在每幀、每類別維度上用 bool masks（[h_r, w_r]）
            #   2) 以「跨幀 union」得到每類別在該尺度的 union 像素集合 v_c ∈ {0,1}^Q
            #   3) 再計算外積 v_c v_c^T 形成 [Q,Q]（或以 (V @ V.T)>0 得一張合併遮罩）；最後多一個 BG 通道
            # =====================================================================
    
            # 2-1) 收集 per-class union 像素（跨幀 OR）
            #      V: [Q, seg_cls]，bool
            # V_list = []
            # for s in range(seg_cls):
            #     # union over frames: [F, h_r, w_r] -> [h_r, w_r] -> reshape(-1)==Q
            #     # layouts_interp_hw: [F, S, 1, h_r, w_r]
            #     cls_union_hw = (layouts_interp_hw[:, s, 0] > 0)  # bool, [F, h_r, w_r]
            #     cls_union_hw = cls_union_hw.any(dim=0)          # [h_r, w_r]
            #     V_list.append(cls_union_hw.reshape(-1))         # [Q]
    
            # V = torch.stack(V_list, dim=1)  # [Q, seg_cls], bool
    
            # # 2-2) 背景通道 BG = ~(A ∪ B ∪ ...)，bool
            # union_all = V.any(dim=1)               # [Q]
            # bg_vec = ~union_all                    # [Q]
            # # 拼成 [Q, C]（C = seg_cls + 1）
            # VC = torch.cat([V, bg_vec.unsqueeze(1)], dim=1)  # [Q, C]
            # C = VC.shape[1]
    
            # # 2-3) reg_sizes_class：各通道像素占比（[bsz,1,1,1,C]）
            # reg_sizes_vec = VC.float().mean(dim=0)  # [C]
            # reg_sizes_class[Q] = reg_sizes_vec.view(1, 1, 1, 1, C).expand(bsz, 1, 1, 1, C).contiguous()
    
            # # 2-4) sreg_maps_class：把每個通道展成外積（必要時才 materialize）
            # #     這裡給出完整 [bsz,1,Q,Q,C]，dtype=bool 以省顯存（後面用時可轉 float）
            # sreg_qkc_list = []
            # VC_bool = VC  # [Q, C], bool
            # for cidx in range(C):
            #     v = VC_bool[:, cidx]              # [Q], bool
            #     outer_c = (v[:, None] & v[None, :])  # [Q, Q], bool
            #     sreg_qkc_list.append(outer_c)
    
            # sreg_qkc = torch.stack(sreg_qkc_list, dim=-1)  # [Q, Q, C], bool
            # sreg_maps_class[Q] = sreg_qkc.unsqueeze(0).unsqueeze(0).expand(bsz, 1, Q, Q, C).contiguous()  # [bsz,1,Q,Q,C], bool
    
            # === 🔵 新增結束 ===          
            
        ###########################
        ###### prep for creg ######
        ###########################
        pww_maps = torch.zeros(frames, 1, 77, height, width).to(device)
        for i in range(1,len(prompts)):
            wlen = text_input['length'][i] - 2
            widx = text_input['input_ids'][i][1:1+wlen]
            for j in range(77):
                if (text_input['input_ids'][0][j:j+wlen] == widx).sum() == wlen:
                    for f in range(frames):
                        pww_maps[f,:,j:j+wlen,:,:] = layouts[f,i-1:i]    # frames, seg_cls, c, h ,w = layouts.shape
                    cond_embeddings[0][j:j+wlen] = cond_embeddings[i][1:1+wlen]
                    print(prompts[i], i, '-th segment is handled.')
                    break
        
        # print("cond_embeddings",cond_embeddings)
        creg_maps = {}
        for r in range(4):
            pww_maps_frames = []
            h = int(height/np.power(2,r))
            w = int(width/np.power(2,r))
            for i in range(frames):
                pww_map_frame = pww_maps[i]
                pww_map_frame.view(1,77,height,width)
                pww_map_frame = F.interpolate(pww_map_frame, (h, w), mode='nearest')
                pww_map_frame = pww_map_frame.view(1, 77, -1).permute(0, 2, 1).repeat(bsz,1,1)  # 重新调整形状
                pww_maps_frames.append(pww_map_frame)
            # 使用 torch.cat 连接处理后的所有帧
            layout_c = torch.stack(pww_maps_frames, dim=0)
            # print("layout_c",layout_c)
            creg_maps[h*w] = layout_c

        ###########################    
        #### prep for text_emb ####
        ###########################
        text_cond = torch.cat([uncond_embeddings, cond_embeddings[:1].repeat(bsz,1,1)])

        return text_cond, sreg_maps, creg_maps, reg_sizes, reg_sizes_c, sreg_maps_class, reg_sizes_class, id_masks_by_res


### original + 64*64
#     def _prepare_attention_layout(self,bsz,height,width,layouts,prompts,clip_length,attention_type,device):
#         print("start prepare attention layout")
#         ## current layouts  f s c h w
#         ## org layouts s c h w
            
#         #print("prompt:",prompts)
#         # sp_sz =self.unet.sample_size
#         sp_sz = height*width
#         frames, seg_cls, c, h ,w = layouts.shape
#         text_input = self.tokenizer(prompts, padding="max_length", return_length=True, return_overflowing_tokens=False, 
#                                     max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
#         cond_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]

#         uncond_input = self.tokenizer([""]*bsz, padding="max_length", max_length=self.tokenizer.model_max_length,
#                                     truncation=True, return_tensors="pt")
#         uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]

#         for i in range(1,len(prompts)):
#             wlen = text_input['length'][i] - 2
#             widx = text_input['input_ids'][i][1:1+wlen]
#             for j in range(77):
#                 if (text_input['input_ids'][0][j:j+wlen] == widx).sum() == wlen:
#                     break

#         ###########################
#         ###### prep for sreg ###### 
#         ###########################
      
#         sreg_maps = {}
#         reg_sizes = {}
#         reg_sizes_c = {}

#         device = layouts.device

#         frame_index_pre = torch.arange(frames)+(-1)
#         frame_index_pre = frame_index_pre.clip(0, frames-1)

#          # === [NEW] 初始化 id_maps：存 per-scale 的 id_flat ===
#         id_maps = {}       


        
#         # [MOD-LABELS-64] 新增：labels 檔案存放資料夾（可依需求改）
#         # ----------------------------------------------------------
#         labels_root = self._ensure_dir("./sreg_labels_Roger")  # <--- [MOD-LABELS-64] 新增
#         debug_dir = self._ensure_dir("./debug_masks")


#         # ========================= 視覺化：原始輸入 layouts（可選） =========================
#         # [VIS-RAW-START] 直接從輸入 layouts 存圖（H=W=原尺寸），幫你確認「第二維通道」逐幀變化
#         debug_vis = True
#         if debug_vis:
#             raw_out_dir = os.path.join(debug_dir, "r_raw_input_{}x{}".format(h, w))
#             pal = self._make_simple_palette(seg_cls + 1)  # 用於語意彩圖
#             with torch.no_grad():
#                 # binary masks by class
#                 for f in range(frames):
#                     # layouts[f]: [seg_cls, c, h, w]，通常 c==1
#                     li = layouts[f, :, 0]  # [seg_cls, h, w]
#                     # 存各 class 的二值圖
#                     for s in range(seg_cls):
#                         m = (li[s] > 0.5).float()     # 二值化
#                         self._save_png(self._to_uint8(m), os.path.join(raw_out_dir, f"frame_{f:03d}_cls{s}.png"))

#                     # 再存 argmax 語意彩色圖
#                     # 先把 [seg_cls,H,W] 轉到 [H,W] 的類別ID（0=背景）
#                     # 這裡以最大值類別當成ID，沒有重疊時可以快速看翻轉
#                     score = li  # 假設 li 已是 [0,1] mask；如有 logits 可改 soft
#                     arg = score.argmax(dim=0) + 1  # 1..seg_cls；0保留給背景
#                     bg = (score.max(dim=0).values <= 0.5)
#                     arg[bg] = 0
#                     self._save_png(arg.to(torch.uint8).cpu().numpy(), os.path.join(raw_out_dir, f"frame_{f:03d}_colormap.png"), palette=pal)
#         # ----------------------------------------------------------
#         for r in range(4):
#             layouts_s_frames = []
#             if attention_type == "SparseCausalAttention":
#                 layouts_s_sparse_attn = []
#             import numpy as np
#             h = int(height/np.power(2,r))
#             w= int(width/np.power(2,r))
#             #layouts torch.Size([70, 2, 1, 64, 64])
#             # layouts_interpolate = F.interpolate(layouts.squeeze(2), (res, res), mode='nearest').unsqueeze(2)    
#             layouts_interpolate = F.interpolate(layouts.squeeze(2), (h, w), mode='nearest').unsqueeze(2)  
#             layouts_interpolate_2d = layouts_interpolate
#             layouts_interpolate = layouts_interpolate.view(frames,seg_cls,1,-1)   ## frames,seg_cls,1,res^2
          
#             # === [NEW] 建立 this-scale 的 id_flat: 0=bg, 1=spiderman, 2=bear ===
#             # 假設 layouts_interpolate[:,0,0] = Spiderman mask, layouts_interpolate[:,1,0] = PolarBear mask
#             sp_mask = (layouts_interpolate_2d [:, 0, 0] > 0)  # [F,h_r,w_r]
#             br_mask = (layouts_interpolate_2d [:, 1, 0] > 0)  # [F,h_r,w_r]
#             ids = torch.zeros((frames, h, w), dtype=torch.long, device=device)
#             ids[sp_mask] = 2
#             ids[br_mask] = 1
#             id_flat = ids.reshape(-1)  # 長度 = frames*h_r*w_r (frame-major 展平)
#             # 存兩種 key，避免混淆
#             id_maps[(h, w, frames)] = id_flat
#             id_maps[h * w] = id_flat

# ####################################################
#             h_r = h
#             w_r = w
#             N_r = h_r*w_r
#         # ========================= 64×64 分支 =========================
#             if h_r == 64 and w_r == 64:
#                 # [# MOD-CORE] 這裡把「2 個 instance 通道」轉成 3 類 labels：0 背景、1 左人、2 右人
#                 label_file_paths = []
#                 for i in range(frames):
#                     # li_i: [seg_cls, 1, N]  -> squeeze -> [seg_cls, N]
#                     li_i = layouts_interpolate[i].squeeze(1)   # [S, N]，S==2（左/右）
#                     S, N = li_i.shape
#                     assert S == 2, f"[64x64] Expect seg_cls==2 (two instances), got {S}"
    
#                     # ---- 把 li_i 還原回 [2,64,64] 做穩定下採樣與 hard 化 ----
#                     li_full = li_i.view(S, 64, 64)             # 已經是 64×64（若上層非整數倍，請改走 _downsample_two_instances_to_64x64）
#                     li_64   = self._downsample_two_instances_to_64x64(li_full, out_hw=(64, 64), thr=0.5)  # [2,64,64]
#                     li      = li_64.view(S, -1)                # [2,4096]
    
#                     # ---- 建立 3 類 labels ----
#                     thr = 0.5
#                     left  = (li[0] > thr)
#                     right = (li[1] > thr)
    
#                     labels = torch.zeros(N, dtype=torch.int32, device=li.device)   # 0=背景
#                     labels[left]  = 1
#                     labels[right] = 2
    
#                     # 解決下採樣造成的重疊：誰分數大就歸誰
#                     both = left & right
#                     if both.any():
#                         labels[both] = torch.where(
#                             li[0, both] >= li[1, both],
#                             torch.tensor(1, device=li.device, dtype=torch.int32),
#                             torch.tensor(2, device=li.device, dtype=torch.int32),
#                         )
    
#                     # ---- 存檔 ----
#                     save_path = os.path.join(labels_root, f"labels_f{i:03d}_64x64.dat")
#                     self.save_labels_memmap(labels, save_path)
#                     label_file_paths.append(save_path)
    
#                 # 記錄「labels 模式」的 sreg
#                 sreg_maps[N_r] = {
#                     "mode": "labels_memmap",
#                     "paths": label_file_paths,
#                     "H": h_r, "W": w_r, "N": N_r,
#                     "classes": 3,   # 0=bg,1=left,2=right（給下游參考）
#                 }
#                 reg_sizes[N_r] = None
#                 reg_sizes_c[N_r] = None
    
#                 # 64×64 不建 dense（避免 OOM）
#                 continue
#         # ======================= /64×64 分支 =======================
# ####################################################
#     ###################################################################################
#             # bg_idx = seg_cls - 1
#             # fg_idx = [i for i in range(seg_cls) if i != bg_idx]
#             # def _union_fg_with_bg(li_frame):
#             #     """
#             #     li_frame: (seg_cls, 1, h*w)  單幀、單解析度的扁平化 mask
            #     回傳:     (seg_cls, 1, h*w)  將每個前景通道加上背景（clamp 到 {0,1}）
            #     """
            #     # li_frame 是 0/1 mask；若是浮點，可直接相加再 clamp
            #     bg = li_frame[bg_idx:bg_idx+1]              # (1,1,N)
            #     li_frame[fg_idx] = torch.clamp(li_frame[fg_idx] + bg, max=1.0)
            
            #     # 可選：避免背景自己也形成 block（通常不希望背景彼此強連通）
            #     li_frame[bg_idx] = 0.0 * li_frame[bg_idx]
            #     return li_frame
    
    ###################################################################################
            ### implementation of sparse casual attn and fully frame attn
        #     for i in range(frames):
        #         #layouts_f = layouts[i]

        #         layouts_s = layouts_interpolate[i]

        #         if attention_type == "SparseCausalAttention":
                
        #             ### prepare for SparseCausalAttention query, key/value
        #             query= layouts_s
        #             query = query.view(query.size(0),-1,1).to(device)  ### segcls,res^2,1  #[cls, 4096, 1]
        
        #             ### key should be segcls,1,2xres^2
        #             key = torch.cat((layouts_interpolate[0],layouts_interpolate[frame_index_pre[i]]),dim=-1).to(device)
        #             #([cls, 1, 8192])

        #             layouts_s_cross_frame_attn= (query * key).sum(0).unsqueeze(0).repeat(bsz,1,1)  ## 1,4096,8192

        #             layouts_s_sparse_attn.append(layouts_s_cross_frame_attn)
                
        #         layouts_s = (layouts_s.view(layouts_s.size(0),1,-1)*layouts_s.view(layouts_s.size(0),-1,1)).sum(0).unsqueeze(0).repeat(bsz,1,1)

        #         layouts_s_frames.append(layouts_s)


        #     layouts_s_frames = torch.stack(layouts_s_frames,dim=0)
        #     if attention_type == "SparseCausalAttention":
        #         layouts_s_sparse_attn = torch.stack(layouts_s_sparse_attn,dim=0)
        #         sreg_maps[h*w] = layouts_s_sparse_attn
        #         reg_sizes[h*w] = 1-1.*layouts_s_frames.sum(-1, keepdim=True)/(np.power(clip_length, 2))
        #         reg_sizes_c[h*w]  = 1-1.*layouts_s_frames.sum(-1, keepdim=True)/(np.power(clip_length, 2))
        #     #### code for check error#####
        #     # num_nonzero = torch.count_nonzero(layouts_s_frames)
        #     # print("num_nonzero",num_nonzero)
        #     # print("layouts_s_frames",layouts_s_frames.shape)
        #     # print("layouts_s_frames",layouts_s_frames)
        #     # print("reg_size final shape:", (1-1.*layouts_s_frames.sum(-1, keepdim=True)/(np.power(res, 2))).shape)
        #     # print("reg_size", (1-1.*layouts_s_frames.sum(-1, keepdim=True)/(np.power(res, 2))))
        #     #### code for check error#####


        #     #print("layouts_s",layouts_s.shape)

        #     #print("layouts_s.view(layouts_s.size(0),-1,1)",*layouts_s.view(layouts_s.size(0),-1,1).shape)

        #     if attention_type == "FullyFrameAttention":
        #         layouts_s= rearrange(layouts_interpolate,"f s c res -> s c (f res)")  #layouts_s torch.Size([2, 1, 61440])
        #         ####################################################
        #         # layouts_s = _union_fg_with_bg(layouts_s)
        #         ####################################################
        #         if r==0:
        #             layout_s = None
        #             reg_sizes[h*w] = None
        #             sreg_maps[h*w] = None
        #             reg_sizes_c[h*w] = None
        #             print("layout mask order check:")
        #             print("mask[0][0] nonzero:", (layouts[0,0,0]>0).sum().item(), "  -> Spider candidate")
        #             print("mask[0][1] nonzero:", (layouts[0,1,0]>0).sum().item(), "  -> Bear candidate")
        #         else:
        #             layouts_s = (layouts_s*layouts_s.view(layouts_s.size(0),-1,1)).sum(0).unsqueeze(0).repeat(bsz,1,1).to(torch.float16) #layouts_s torch.Size([1, 15360, 15360]) for h,w=32
        #             sreg_maps[h*w] = layouts_s
        #             reg_sizes[h*w] = 1-1.*layouts_s.sum(-1, keepdim=True)/((h*clip_length)*(w*clip_length))
        #             reg_sizes_c[h*w]  =  1-1.*layouts_s_frames.sum(-1, keepdim=True)/(h*w)

        #         # layouts_s = (layouts_s*layouts_s.view(layouts_s.size(0),-1,1)).sum(0).unsqueeze(0).repeat(bsz,1,1).to(torch.float16) #layouts_s torch.Size([1, 15360, 15360])
        #         # sreg_maps[h*w] = layouts_s
        #         # reg_sizes[h*w] = 1-1.*layouts_s.sum(-1, keepdim=True)/((h*clip_length)*(w*clip_length))
        #         # reg_sizes_c[h*w]  =  1-1.*layouts_s_frames.sum(-1, keepdim=True)/(h*w)     
        #         ###################################################   
        #         # if res == 64:
        #         #     reg_sizes[np.power(res, 2)] = None
        #         # else:
        #         #     reg_sizes[np.power(res, 2)] = 1-1.*layouts_s.sum(-1, keepdim=True)/(np.power(res*clip_length, 2))
        #         # #sreg_maps[np.power(res, 2)] = layouts_s_frames
        #         # sreg_maps[np.power(res, 2)] = layouts_s
        #         # reg_sizes_c[np.power(res, 2)]  =  1-1.*layouts_s_frames.sum(-1, keepdim=True)/(np.power(res, 2))
            
            
        # ###########################
        # ###### prep for creg ######
        # ###########################
        # pww_maps = torch.zeros(frames, 1, 77, height, width).to(device)
        # for i in range(1,len(prompts)):
        #     wlen = text_input['length'][i] - 2
        #     widx = text_input['input_ids'][i][1:1+wlen]
        #     for j in range(77):
        #         if (text_input['input_ids'][0][j:j+wlen] == widx).sum() == wlen:
        #             for f in range(frames):
        #                 pww_maps[f,:,j:j+wlen,:,:] = layouts[f,i-1:i]    # frames, seg_cls, c, h ,w = layouts.shape
        #             cond_embeddings[0][j:j+wlen] = cond_embeddings[i][1:1+wlen]
        #             print(prompts[i], i, '-th segment is handled.')
        #             break
        
        # # print("cond_embeddings",cond_embeddings)
        # creg_maps = {}
        # for r in range(4):
        #     pww_maps_frames = []
        #     h = int(height/np.power(2,r))
        #     w = int(width/np.power(2,r))
        #     for i in range(frames):
        #         pww_map_frame = pww_maps[i]
        #         pww_map_frame.view(1,77,height,width)
        #         pww_map_frame = F.interpolate(pww_map_frame, (h, w), mode='nearest')
        #         pww_map_frame = pww_map_frame.view(1, 77, -1).permute(0, 2, 1).repeat(bsz,1,1)  # 重新调整形状
        #         pww_maps_frames.append(pww_map_frame)
        #     # 使用 torch.cat 连接处理后的所有帧
        #     layout_c = torch.stack(pww_maps_frames, dim=0)
        #     # print("layout_c",layout_c)
        #     creg_maps[h*w] = layout_c

        # ###########################    
        # #### prep for text_emb ####
        # ###########################
        # # text_cond = torch.cat([uncond_embeddings, cond_embeddings[:1].repeat(bsz,1,1)])


        # # # ========= 【新增】Mask→Prompt 標記 + 輸出影片 =========
        # # try:
        # #     import cv2
        # #     import numpy as np
        # #     from datetime import datetime

        # #     viz_out_dir = self._ensure_dir(os.path.join(debug_dir, "viz_mask_prompt"))
        # #     run_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
        # #     out_png_dir = self._ensure_dir(os.path.join(viz_out_dir, f"frames_{height}x{width}_{run_tag}"))
        # #     out_mp4_path = os.path.join(viz_out_dir, f"mask_prompt_{height}x{width}_{run_tag}.mp4")

        # #     # 1) 準備「通道 → 文字」的對應（按照你 pww_maps 的寫法：prompts[1:] 對應 layouts 的 seg_cls 通道 0..seg_cls-1）
        # #     #    若 seg_cls 比 prompts[1:] 多，超出的通道會用 "cls{idx}" 當作預設標籤。
        # #     ch2label = {}
        # #     max_bind = min(seg_cls, len(prompts) - 1)
        # #     for s in range(seg_cls):
        # #         if s < max_bind:
        # #             ch2label[s] = str(prompts[s + 1])
        # #         else:
        # #             ch2label[s] = f"cls{s}"

        # #     # 2) 一些顏色（BGR），夠用就好，不夠會輪流使用
        # #     color_table = [
        # #         (0, 160, 255),   # 橘
        # #         (0, 220, 0),     # 綠
        # #         (255, 64, 64),   # 藍->(BGR 這其實偏紅，下面再來個紫/青)
        # #         (200, 0, 200),   # 紫
        # #         (200, 200, 0),   # 青
        # #         (0, 100, 255),   # 深橘
        # #         (180, 180, 180), # 灰
        # #         (0, 255, 255),   # 黃
        # #     ]

        # #     # 3) 視訊輸出器
        # #     fps = max(4, min(30, clip_length if isinstance(clip_length, int) and clip_length > 0 else 8))
        # #     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # #     writer = cv2.VideoWriter(out_mp4_path, fourcc, fps, (width, height))

        # #     # 4) 逐幀畫圖
        # #     alpha = 0.45   # mask 疊加透明度
        # #     txt_scale = max(0.4, min(1.2, width / 1024.0))   # 依輸出尺寸調整字體大小
        # #     txt_thick = 1

        # #     with torch.no_grad():
        # #         for f in range(frames):
        # #             # 底圖
        # #             canvas = np.zeros((height, width, 3), dtype=np.uint8)
        # #             canvas[:] = (10, 10, 10)

        # #             # 角落資訊（縮小、不干擾中間）
        # #             info = f"frame {f:03d}/{frames-1:03d}"
        # #             cv2.putText(canvas, info, (12, 24),
        # #                         cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        # #             for s in range(seg_cls):
        # #                 m = layouts[f, s, 0]
        # #                 m_np = (m.detach().float().cpu().numpy() > 0.5).astype(np.uint8)
        # #                 if m_np.sum() == 0:
        # #                     continue

        # #                 # 顏色 & 疊加
        # #                 color = color_table[s % len(color_table)]
        # #                 overlay = canvas.copy()
        # #                 overlay[m_np.astype(bool)] = color
        # #                 canvas = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)

        # #                 # 找最大輪廓 → 外接框
        # #                 cnts, _ = cv2.findContours(m_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # #                 if len(cnts) == 0:
        # #                     continue
        # #                 cnt = max(cnts, key=cv2.contourArea)
        # #                 x, y, w, h = cv2.boundingRect(cnt)

        # #                 # 畫輪廓
        # #                 cv2.drawContours(canvas, [cnt], -1, color, 2)

        # #                 # 準備標籤
        # #                 label = ch2label.get(s, f"cls{s}")

        # #                 # 以 scale=1 估基準文字尺寸
        # #                 (tw0, th0), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        # #                 if tw0 == 0 or th0 == 0:
        # #                     continue

        # #                 # 自適應縮放：讓文字落在 bbox 的 70% 範圍內
        # #                 LABEL_SCALE_BOOST = 0.3   # 整體縮小字體，原本 1.0 → 改 0.7
        # #                 MIN_SCALE = 0.25          # 原本 0.35 → 改 0.25
        # #                 MAX_SCALE = 1.8           # 原本 2.0 → 改 1.8
        # #                 TARGET_RATIO = 0.60       # 原本 0.85 → 改 0.60，文字佔 bbox 面積更小
        # #                 target_ratio = TARGET_RATIO
        # #                 scale_w = (w * target_ratio) / max(1, tw0)
        # #                 scale_h = (h * target_ratio) / max(1, th0)
        # #                 scale = min(scale_w, scale_h) * LABEL_SCALE_BOOST
        # #                 scale = max(MIN_SCALE, min(MAX_SCALE, scale))
        # #                 thickness = max(1, int(round(2 * scale)))

        # #                 # 以縮放後尺寸重新取得寬高
        # #                 (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)

        # #                 # 預設：把文字放在 bbox 內置中（靠上些）
        # #                 pad = max(2, int(4 * scale))
        # #                 tx = int(x + (w - tw) / 2)
        # #                 ty = int(y + (h + th) / 2)  # baseline
        # #                 # 嘗試避免超框：若 bbox 太小或文字高度接近 bbox，改放到 bbox 上方並加指示線
        # #                 place_outside = False
        # #                 if tw > w * 0.98 or th > h * 0.90:
        # #                     place_outside = True

        # #                 if not place_outside:
        # #                     # 畫底框提高可讀性
        # #                     bx0, by0 = max(0, tx - pad), max(0, ty - th - pad)
        # #                     bx1, by1 = min(width - 1, tx + tw + pad), min(height - 1, ty + pad)
        # #                     cv2.rectangle(canvas, (bx0, by0), (bx1, by1), (0, 0, 0), -1)
        # #                     cv2.putText(canvas, label, (tx, ty),
        # #                                 cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        # #                 else:
        # #                     # 放到 bbox 上方中線，並拉一條線指回 bbox
        # #                     out_ty = max(10, y - 8)  # 文字 baseline
        # #                     out_tx = int(x + (w - tw) / 2)
        # #                     bx0, by0 = max(0, out_tx - pad), max(0, out_ty - th - pad)
        # #                     bx1, by1 = min(width - 1, out_tx + tw + pad), min(height - 1, out_ty + pad)
        # #                     cv2.rectangle(canvas, (bx0, by0), (bx1, by1), (0, 0, 0), -1)
        # #                     cv2.putText(canvas, label, (out_tx, out_ty),
        # #                                 cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # #                     # 指示線：從外框中心指到 bbox 重心
        # #                     M = cv2.moments(m_np)
        # #                     if M["m00"] > 1e-6:
        # #                         cx = int(M["m10"] / M["m00"])
        # #                         cy = int(M["m01"] / M["m00"])
        # #                     else:
        # #                         ys, xs = np.where(m_np > 0)
        # #                         cx, cy = (int(xs.mean()), int(ys.mean())) if xs.size > 0 else (x + w // 2, y + h // 2)
        # #                     line_y = int((by0 + by1) / 2)
        # #                     line_x = int((bx0 + bx1) / 2)
        # #                     cv2.line(canvas, (line_x, line_y), (cx, cy), (0, 0, 0), max(1, thickness - 1))
        # #                     cv2.line(canvas, (line_x, line_y), (cx, cy), (255, 255, 255), 1)

        # #             # 存檔與寫影格
        # #             png_path = os.path.join(out_png_dir, f"{f:04d}.png")
        # #             cv2.imwrite(png_path, canvas)
        # #             writer.write(canvas)

        # #     writer.release()
        # #     print(f"[Mask-Prompt 可視化] 單幀輸出：{out_png_dir}")
        # #     print(f"[Mask-Prompt 可視化] MP4 影片：{out_mp4_path}")

        # # except Exception as e:
        # #     print(f"[Mask-Prompt 可視化] 產生失敗：{e}")
        # # ========= 【新增】區塊結束 =========
        # # import sys
        # # sys.exit()
        # return text_cond, sreg_maps, creg_maps, reg_sizes, reg_sizes_c, id_maps



    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        latent_mask: Union[torch.FloatTensor, PIL.Image.Image] = None,
        layouts: Union[torch.FloatTensor, PIL.Image.Image] = None,
        dirty_A: list = None,
        dirty_B: list = None,
        clean_A: list = None,
        clean_B: list = None,
        latent_part_mask: Union[torch.FloatTensor, PIL.Image.Image] = None,
        part_layouts: Union[torch.FloatTensor, PIL.Image.Image] = None,
        blending_percentage: float=0.25,
        modulated_percentage: float=0.3,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = None,
        num_inference_steps: int = 50,
        clip_length: int = 8,
        guidance_scale: float = 7.5,
        source_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        control: Optional[torch.FloatTensor] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        logdir: str=None,
        controlnet_conditioning_scale: float = 1.0,
        use_pnp:  bool = False,
        cluster_inversion_feature: bool = False,
        vis_cross_attn: bool = False,
        attn_inversion_dict: dict=None,
        **kwargs,
    ):
        print("============= pipeline =================")
        # 0. Default height and width to unet
        t , c , height, width = image.shape
        prompt = OmegaConf.to_container(prompt, resolve=True)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps, strength)

        # 2. Define call parameters
        batch_size = 1
        weight_dtype = image.dtype
        device = self._execution_device

        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
    
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        
        if latents is None:
            cache_path = "./inversion_cache/inversion_latents.pt"
            
            if os.path.exists(cache_path):
                print(f"✅ Found cached inversion latents at {cache_path}")
                data = torch.load(cache_path, map_location="cuda")   # 自動放回 GPU
                latents = data["latents"]
                attn_inversion_dict = data["attn_inversion_dict"]
            else:
                os.makedirs(cache_path, exist_ok=True)
                latents, attn_inversion_dict = self.prepare_latents_ddim_inverted(
                    image, batch_size, source_prompt,
                    do_classifier_free_guidance, generator,
                    control, controlnet_conditioning_scale, use_pnp, cluster_inversion_feature
                )
                print("use inversion latents")
            
                torch.save({
                    "latents": latents.cpu(),
                    "attn_inversion_dict": attn_inversion_dict
                }, cache_path)
        ## prepare text embedding, self attention map, cross attention map
        _, _, _, downsample_height, downsample_width = latents.shape
        attention_type = self._get_attention_type()

        text_cond, sreg_maps, creg_maps, reg_sizes,reg_sizes_c,sreg_maps_class, reg_sizes_class, id_masks_by_res = self._prepare_attention_layout(batch_size,downsample_height,downsample_width,layouts, prompt,clip_length,attention_type,device)
                                                                                                
        # , sreg_maps_class, reg_sizes_class, id_masks_by_res
        time_steps = self.scheduler.timesteps

        #============do visualization for st-layout attn===============#
        self.store_controller = attention_util.AttentionStore()
        editor = ST_Layout_Attn_ControlEdit(text_cond=text_cond,sreg_maps=sreg_maps,creg_maps=creg_maps,reg_sizes=reg_sizes,reg_sizes_c=reg_sizes_c,
                                                time_steps=time_steps,clip_length=clip_length,attention_type=attention_type,
                                                additional_attention_store=self.store_controller,
                                                save_self_attention = True,
                                                disk_store = False,
                                                video = image,
                                                # id_maps = id_maps
                                                # sreg_maps_class = sreg_maps_class,
                                                # reg_sizes_class = reg_sizes_class,
                                                id_masks_by_res = id_masks_by_res
                                                )  
        print("======register_attention_control=======")
        if editor.sreg_maps is None:
            print("sreg_map is none!!!!!")
        else:
            print("sreg_map is not none")
        # attention_util.register_attention_control(self, editor, text_cond, clip_length, downsample_height,downsample_width,ddim_inversion=False, id_masks_by_res=id_masks_by_res)

        print()
        attention_util.register_attention_control(self, editor, text_cond, clip_length, downsample_height,downsample_width,ddim_inversion=False, id_masks_by_res=id_masks_by_res, dirty_A=dirty_A, dirty_B=dirty_B, clean_A=clean_A, clean_B=clean_B)
        #============do visualization for st-layout attn===============#

        # editor = ST_Layout_Attn_Control(text_cond=text_cond,sreg_maps=sreg_maps,creg_maps=creg_maps,reg_sizes=reg_sizes,reg_sizes_c=reg_sizes_c,
        #                                    time_steps=time_steps,clip_length=clip_length,attention_type=attention_type)  

        # register_attention_control(self, editor, text_cond, clip_length,downsample_height,downsample_width,ddim_inversion=False)

        # 3. Encode input prompt  
        prompt = prompt[:1]
        print("Prepare text embedding ...")
        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )
        print("Prepare source latent ...")
        source_latents = self.prepare_source_latents(
            image, batch_size, num_images_per_prompt, 
            # text_embeddings.dtype, device, 
            text_embeddings,
            generator,
        )

        # 7. Denoising loop
        # print("Start Denoise")
        num_warmup_steps = len(time_steps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps* (1-blending_percentage)) as progress_bar:
            for i, t in enumerate(time_steps[int(len(time_steps) * blending_percentage):]):
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # inject features
                    if use_pnp and i < kwargs["inject_step"]:
                        print("use pnp pnp pnp!")
                        self.unet.up_blocks[1].resnets[0].out_layers_inject_features = attn_inversion_dict['features0'][i].to(device)
                        self.unet.up_blocks[1].resnets[1].out_layers_inject_features = attn_inversion_dict['features1'][i].to(device)
                        self.unet.up_blocks[2].resnets[0].out_layers_inject_features = attn_inversion_dict['features2'][i].to(device)
                        self.unet.up_blocks[1].attentions[1].transformer_blocks[0].attn1.inject_q = attn_inversion_dict['q4'][i].to(device)
                        self.unet.up_blocks[1].attentions[1].transformer_blocks[0].attn1.inject_k =  attn_inversion_dict['k4'][i].to(device)
                        self.unet.up_blocks[1].attentions[2].transformer_blocks[0].attn1.inject_q =  attn_inversion_dict['q5'][i].to(device)
                        self.unet.up_blocks[1].attentions[2].transformer_blocks[0].attn1.inject_k =  attn_inversion_dict['k5'][i].to(device)
                        self.unet.up_blocks[2].attentions[0].transformer_blocks[0].attn1.inject_q =  attn_inversion_dict['q6'][i].to(device)
                        self.unet.up_blocks[2].attentions[0].transformer_blocks[0].attn1.inject_k =  attn_inversion_dict['k6'][i].to(device)
                        self.unet.up_blocks[2].attentions[1].transformer_blocks[0].attn1.inject_q = attn_inversion_dict['q7'][i].to(device)
                        self.unet.up_blocks[2].attentions[1].transformer_blocks[0].attn1.inject_k =  attn_inversion_dict['k7'][i].to(device)
                        self.unet.up_blocks[2].attentions[2].transformer_blocks[0].attn1.inject_q =  attn_inversion_dict['q8'][i].to(device)
                        self.unet.up_blocks[2].attentions[2].transformer_blocks[0].attn1.inject_k =  attn_inversion_dict['k8'][i].to(device)
                        self.unet.up_blocks[3].attentions[0].transformer_blocks[0].attn1.inject_q =  attn_inversion_dict['q9'][i].to(device)
                        self.unet.up_blocks[3].attentions[0].transformer_blocks[0].attn1.inject_k =  attn_inversion_dict['k9'][i].to(device)
                    else:
                        self.clean_features()

                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings,
                        controlnet_cond=control,
                        return_dict=False,
                    )
                    down_block_res_samples = [
                        down_block_res_sample * controlnet_conditioning_scale
                        for down_block_res_sample in down_block_res_samples
                    ]
                    mid_block_res_sample *= controlnet_conditioning_scale
                    # print("Start pred noise")
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        **kwargs,
                    ).sample.to(dtype=weight_dtype)


                    # perform guidance
                    if do_classifier_free_guidance:
                        # print("do_classifier_free_guidance")
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample


                    # Blending
                    noise_source_latents = self.scheduler.add_noise(
                        source_latents, torch.randn_like(latents), t
                    )

                    latents = latents * latent_mask + noise_source_latents * (1 - latent_mask)
##########################################################################################################
                                    # === 參數 ===
                    make_step_videos = False       # 是否每 5 步輸出影片
                    video_every = 5               # 每幾步輸出一次影片（50 步 → 每 5 步 1 支 → 共 10 支）
                    video_fps = 12                # 影片幀率
                    save_only_first_batch = True  # 只輸出 batch=0
                    
                    # === 輸出資料夾（用 timestamp 避免覆蓋）===
                    run_tag = time.strftime("%Y%m%d-%H%M%S")
                    video_dir = os.path.join(logdir, f"denoise_videos")
                    os.makedirs(video_dir, exist_ok=True)
                    
                    def _save_mp4(frames_np, out_path, fps=12):
                        """
                        frames_np: [T, H, W, C]，數值 0~1 或 0~255
                        會自動轉成 uint8 再寫入 H.264 MP4
                        """
                        assert frames_np.ndim == 4 and frames_np.shape[-1] in (1, 3), f"unexpected: {frames_np.shape}"
                        arr = frames_np
                        if arr.dtype != np.uint8:
                            arr = np.clip(arr, 0.0, 1.0) * 255.0
                            arr = arr.astype(np.uint8)
                        # 若是單通道，轉成 3 通道方便寫入
                        if arr.shape[-1] == 1:
                            arr = np.repeat(arr, 3, axis=-1)
                        writer = imageio.get_writer(out_path, fps=fps, codec='libx264', quality=8)
                        for f in range(arr.shape[0]):
                            writer.append_data(arr[f])
                        writer.close()
                        # 取 VAE/UNet 的 dtype / device（可在迴圈外先算好）
                    # 取 VAE 的 dtype/device（放到循環外更好）
                    vae_device = next(self.vae.parameters()).device
                    vae_dtype  = next(self.vae.parameters()).dtype  # 常見 torch.float16
                    
                    # 若你的 diffusers 版本支援，開啟 VAE slicing/tiling（放循環外執行一次）
                    try:
                        if hasattr(self.vae, "enable_slicing"): self.vae.enable_slicing()
                        if hasattr(self.vae, "enable_tiling"):  self.vae.enable_tiling()
                    except Exception:
                        pass
                    
                    # =====================  插入：每 5 步輸出 1 支影片（串流小塊解碼）  =====================
                    if make_step_videos and (video_every > 0) and ((i % video_every) == 0):
                        import imageio.v2 as imageio
                        import numpy as np
                        from torch.cuda.amp import autocast
                    
                        with torch.inference_mode():
                            latents_vis = latents[:1] if save_only_first_batch else latents     # [B,C,T,H,W]
                            B, C, T, H, W = latents_vis.shape
                            step_abs = int(len(time_steps) * blending_percentage) + i
                            out_path = os.path.join(video_dir, f"video_step_{step_abs:04d}_b0.mp4")
                    
                            # 串流寫 MP4，不把整段影片放在 GPU/CPU 記憶體
                            writer = imageio.get_writer(out_path, fps=video_fps, codec='libx264', quality=8)
                    
                            # 重要：按時間維分塊（例如每次解 4 幀），你可依 GPU 改 t_chunk=1/2/4/8
                            t_chunk = 4
                    
                            for t0 in range(0, T, t_chunk):
                                t1 = min(T, t0 + t_chunk)
                                # 取 [B,C,tt,H,W] → [B*tt,C,H,W]
                                x = latents_vis[:, :, t0:t1].contiguous()
                                Btt = x.shape[0] * x.shape[2]
                                x = x.permute(0, 2, 1, 3, 4).reshape(Btt, C, H, W)
                                x = x.detach().to(device=vae_device, dtype=vae_dtype)
                    
                                # 在 VAE 精度下跑（舊版 torch 用 autocast(dtype=...)）
                                try:
                                    ctx = autocast(dtype=(torch.float16 if vae_dtype == torch.float16 else torch.bfloat16))
                                except TypeError:
                                    # 更舊版 torch 沒有 dtype 參數，直接關掉 AMP
                                    class _Dummy:
                                        def __enter__(self): pass
                                        def __exit__(self, *a): pass
                                    ctx = _Dummy()
                    
                                with ctx:
                                    # 直接走 VAE.decode（不經你原本的 decode_latents 一次吃爆）
                                    # diffusers VAE 輸出區間通常 [-1,1]，後處理到 [0,1]
                                    dec = self.vae.decode(x).sample  # [B*tt,3,H,W]，dtype≈vae_dtype
                    
                                dec = dec.float()                    # 後處理用 fp32
                                dec = (dec / 2 + 0.5).clamp(0, 1)    # [-1,1] → [0,1]
                                dec = dec.permute(0, 2, 3, 1).cpu().numpy()  # [B*tt,H,W,C]
                    
                                # 只寫 batch=0 的幀（如果想全部 batch，就多一層 batch 邏輯）
                                # 這裡由於我們已經把 B 合併進 B*tt，因此只需按 tt 順序寫
                                for k in range(dec.shape[0]):
                                    frame = (dec[k] * 255).astype(np.uint8)
                                    if frame.shape[-1] == 1:
                                        frame = np.repeat(frame, 3, axis=-1)
                                    writer.append_data(frame)
                    
                                # 釋放 GPU 記憶體
                                del x, dec
                                torch.cuda.empty_cache()
                    
                            writer.close()
                            print(f"[video] saved {out_path}")
                    # =====================  插入結束 =====================








                    # call the callback, if provided
                    if i == len(time_steps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)

        ### vis cross attn
        # image shape fchw
        if vis_cross_attn:
            save_path = os.path.join(logdir,'visualization_denoise')
            os.makedirs(save_path, exist_ok=True)
            attention_output = attention_util.show_cross_attention_plus_org_img(self.tokenizer,prompt, image, editor, 32, ["up","down"],save_path=save_path)

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        has_nsfw_concept = None

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)
        torch.cuda.empty_cache()
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
