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



    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        latent_mask: Union[torch.FloatTensor, PIL.Image.Image] = None,
        layouts: Union[torch.FloatTensor, PIL.Image.Image] = None,
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
            # cache_path = "./inversion_cache/inversion_latents.pt"
            
            # if os.path.exists(cache_path):
            #     print(f"✅ Found cached inversion latents at {cache_path}")
            #     data = torch.load(cache_path, map_location="cuda")   # 自動放回 GPU
            #     latents = data["latents"]
            #     attn_inversion_dict = data["attn_inversion_dict"]
            # else:
            #     os.makedirs(cache_path, exist_ok=True)
            latents, attn_inversion_dict = self.prepare_latents_ddim_inverted(
                image, batch_size, source_prompt,
                do_classifier_free_guidance, generator,
                control, controlnet_conditioning_scale, use_pnp, cluster_inversion_feature
            )
                # print("use inversion latents")
            
                # torch.save({
                #     "latents": latents.cpu(),
                #     "attn_inversion_dict": attn_inversion_dict
                # }, cache_path)
        ## prepare text embedding, self attention map, cross attention map
        _, _, _, downsample_height, downsample_width = latents.shape
        attention_type = self._get_attention_type()

        text_cond, sreg_maps, creg_maps, reg_sizes,reg_sizes_c,sreg_maps_class, reg_sizes_class, id_masks_by_res = self._prepare_attention_layout(batch_size,downsample_height,downsample_width,layouts, prompt,clip_length,attention_type,device)
                                                                                                
        # , sreg_maps_class, reg_sizes_class, id_masks_by_res
        time_steps = self.scheduler.timesteps

        #============do visualization for st-layout attn===============#
        self.store_controller = attention_util.AttentionStore()
        print(f"====attn:{attention_type}")
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
        attention_util.register_attention_control(self, editor, text_cond, clip_length, downsample_height,downsample_width,ddim_inversion=False, id_masks_by_res=id_masks_by_res)
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
                    #                 # === 參數 ===
                    # make_step_videos = False       # 是否每 5 步輸出影片
                    # video_every = 5               # 每幾步輸出一次影片（50 步 → 每 5 步 1 支 → 共 10 支）
                    # video_fps = 12                # 影片幀率
                    # save_only_first_batch = True  # 只輸出 batch=0
                    
                    # # === 輸出資料夾（用 timestamp 避免覆蓋）===
                    # run_tag = time.strftime("%Y%m%d-%H%M%S")
                    # video_dir = os.path.join(logdir, f"denoise_videos")
                    # os.makedirs(video_dir, exist_ok=True)
                    
                    # def _save_mp4(frames_np, out_path, fps=12):
                    #     """
                    #     frames_np: [T, H, W, C]，數值 0~1 或 0~255
                    #     會自動轉成 uint8 再寫入 H.264 MP4
                    #     """
                    #     assert frames_np.ndim == 4 and frames_np.shape[-1] in (1, 3), f"unexpected: {frames_np.shape}"
                    #     arr = frames_np
                    #     if arr.dtype != np.uint8:
                    #         arr = np.clip(arr, 0.0, 1.0) * 255.0
                    #         arr = arr.astype(np.uint8)
                    #     # 若是單通道，轉成 3 通道方便寫入
                    #     if arr.shape[-1] == 1:
                    #         arr = np.repeat(arr, 3, axis=-1)
                    #     writer = imageio.get_writer(out_path, fps=fps, codec='libx264', quality=8)
                    #     for f in range(arr.shape[0]):
                    #         writer.append_data(arr[f])
                    #     writer.close()
                    #     # 取 VAE/UNet 的 dtype / device（可在迴圈外先算好）
                    # # 取 VAE 的 dtype/device（放到循環外更好）
                    # vae_device = next(self.vae.parameters()).device
                    # vae_dtype  = next(self.vae.parameters()).dtype  # 常見 torch.float16
                    
                    # # 若你的 diffusers 版本支援，開啟 VAE slicing/tiling（放循環外執行一次）
                    # try:
                    #     if hasattr(self.vae, "enable_slicing"): self.vae.enable_slicing()
                    #     if hasattr(self.vae, "enable_tiling"):  self.vae.enable_tiling()
                    # except Exception:
                    #     pass
                    
                    # # =====================  插入：每 5 步輸出 1 支影片（串流小塊解碼）  =====================
                    # if make_step_videos and (video_every > 0) and ((i % video_every) == 0):
                    #     import imageio.v2 as imageio
                    #     import numpy as np
                    #     from torch.cuda.amp import autocast
                    
                    #     with torch.inference_mode():
                    #         latents_vis = latents[:1] if save_only_first_batch else latents     # [B,C,T,H,W]
                    #         B, C, T, H, W = latents_vis.shape
                    #         step_abs = int(len(time_steps) * blending_percentage) + i
                    #         out_path = os.path.join(video_dir, f"video_step_{step_abs:04d}_b0.mp4")
                    
                    #         # 串流寫 MP4，不把整段影片放在 GPU/CPU 記憶體
                    #         writer = imageio.get_writer(out_path, fps=video_fps, codec='libx264', quality=8)
                    
                    #         # 重要：按時間維分塊（例如每次解 4 幀），你可依 GPU 改 t_chunk=1/2/4/8
                    #         t_chunk = 4
                    
                    #         for t0 in range(0, T, t_chunk):
                    #             t1 = min(T, t0 + t_chunk)
                    #             # 取 [B,C,tt,H,W] → [B*tt,C,H,W]
                    #             x = latents_vis[:, :, t0:t1].contiguous()
                    #             Btt = x.shape[0] * x.shape[2]
                    #             x = x.permute(0, 2, 1, 3, 4).reshape(Btt, C, H, W)
                    #             x = x.detach().to(device=vae_device, dtype=vae_dtype)
                    
                    #             # 在 VAE 精度下跑（舊版 torch 用 autocast(dtype=...)）
                    #             try:
                    #                 ctx = autocast(dtype=(torch.float16 if vae_dtype == torch.float16 else torch.bfloat16))
                    #             except TypeError:
                    #                 # 更舊版 torch 沒有 dtype 參數，直接關掉 AMP
                    #                 class _Dummy:
                    #                     def __enter__(self): pass
                    #                     def __exit__(self, *a): pass
                    #                 ctx = _Dummy()
                    
                    #             with ctx:
                    #                 # 直接走 VAE.decode（不經你原本的 decode_latents 一次吃爆）
                    #                 # diffusers VAE 輸出區間通常 [-1,1]，後處理到 [0,1]
                    #                 dec = self.vae.decode(x).sample  # [B*tt,3,H,W]，dtype≈vae_dtype
                    
                    #             dec = dec.float()                    # 後處理用 fp32
                    #             dec = (dec / 2 + 0.5).clamp(0, 1)    # [-1,1] → [0,1]
                    #             dec = dec.permute(0, 2, 3, 1).cpu().numpy()  # [B*tt,H,W,C]
                    
                    #             # 只寫 batch=0 的幀（如果想全部 batch，就多一層 batch 邏輯）
                    #             # 這裡由於我們已經把 B 合併進 B*tt，因此只需按 tt 順序寫
                    #             for k in range(dec.shape[0]):
                    #                 frame = (dec[k] * 255).astype(np.uint8)
                    #                 if frame.shape[-1] == 1:
                    #                     frame = np.repeat(frame, 3, axis=-1)
                    #                 writer.append_data(frame)
                    
                    #             # 釋放 GPU 記憶體
                    #             del x, dec
                    #             torch.cuda.empty_cache()
                    
                    #         writer.close()
                    #         print(f"[video] saved {out_path}")
                    # # =====================  插入結束 =====================

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
