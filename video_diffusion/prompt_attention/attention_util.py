"""
Collect all function in prompt_attention folder.
Provide a API `make_controller' to return an initialized AttentionControlEdit class object in the main validation loop.
"""

from typing import Optional, Union, Tuple, List, Dict
import abc
import numpy as np
import copy
from einops import rearrange

import torch
import torch.nn.functional as F

import video_diffusion.prompt_attention.ptp_utils as ptp_utils
from video_diffusion.prompt_attention.visualization import show_cross_attention,show_cross_attention_plus_org_img,show_self_attention_comp,aggregate_attention
from video_diffusion.prompt_attention.attention_store import AttentionStore, AttentionControl
from video_diffusion.prompt_attention.attention_register import register_attention_control
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


from PIL import Image
import os
from video_diffusion.common.image_util import save_gif_mp4_folder_type,make_grid
import cv2
import math

from PIL import Image, ImageDraw
import numpy as np
import math
import os

class EmptyControl:
    
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn


def apply_jet_colormap(weight):
    # 将权重规范化到0-255
    weight = 255*(weight - weight.min()) / (weight.max() - weight.min()+1e-6)
    weight = weight.astype(np.uint8)

    # 应用Jet颜色映射
    color_mapped_weight = cv2.applyColorMap(weight, cv2.COLORMAP_JET)
    return color_mapped_weight

def show_self_attention_comp(self_attention_map, video, h_index:int, w_index:int, res: int, frames:int, place_in_unet: List[str], step:int ):

    attention_maps = self_attention_map.reshape(frames, res, res, frames, res, res)
    weights =  attention_maps[0,h_index,w_index,:,:,:]
    attention_list = []
    video_frames = []
    #video f,c,h,w

    for i in range(frames):
        weight = weights[i].cpu().numpy()
        weight_colored = apply_jet_colormap(weight)
        weight_colored = weight_colored[:, :, ::-1]  # BGR到RGB的转换
        weight_colored = np.array(Image.fromarray(weight_colored).resize((256, 256)))
        attention_list.append(weight_colored)

        frame = video[i].permute(1,2,0).cpu().numpy()
        mean = np.array((0.48145466, 0.4578275, 0.40821073)).reshape((1, 1, 3))   # [h, w, c]
        varas = np.array((0.26862954, 0.26130258, 0.27577711)).reshape((1, 1, 3)) 
        frame = frame * varas + mean
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-6) * 255
        frame = frame.astype(np.uint8)
        video_frames.append(frame)

    alpha = 0.5
    overlay_frames = [] 

    for frame, attention in zip(video_frames, attention_list):

        attention_resized = cv2.resize(attention, (frame.shape[1], frame.shape[0]))
    
        overlay_frame = cv2.addWeighted(frame, alpha, attention_resized, 1 - alpha, 0)
        
        overlay_frames.append(overlay_frame)
    print('vis self attn')
    save_path = "with_st_layout_vis_self_attn/vis_self_attn"
    os.makedirs(save_path, exist_ok=True)
    video_save_path = f'{save_path}/self-attn-{place_in_unet}-{step}-query-frame0-h{h_index}-w{w_index}.gif'
    save_gif_mp4_folder_type(overlay_frames, video_save_path,save_gif=False)


def draw_grid_on_image(image, grid_size, line_color="gray"):
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for i in range(0, w, grid_size):
        draw.line([(i, 0), (i, h)], fill=line_color)
    for i in range(0, h, grid_size):
        draw.line([(0, i), (w, i)], fill=line_color)
    return image


def identify_self_attention_max_min(sim, video, h_index:int, w_index:int, res: int, frames:int, place_in_unet: str, step:int):
    attention_maps = sim.reshape(frames, res, res, frames, res, res)
    weights = attention_maps[0, h_index, w_index, :, :, :]

    flattened_weights = weights.reshape(-1)
    global_max_index = flattened_weights.argmax().cpu().numpy()
    global_min_index = flattened_weights.argmin().cpu().numpy()
    print('weights.shape',weights.shape)

    frame_max, h_max, w_max = np.unravel_index(global_max_index, weights.shape)
    frame_min, h_min, w_min = np.unravel_index(global_min_index, weights.shape)

    video_frames = []

    query_frame_index = 0
    query_h = h_index
    query_w = w_index

    for i in range(frames):
        frame = video[i].permute(1, 2, 0).cpu().numpy()
        mean = np.array((0.48145466, 0.4578275, 0.40821073)).reshape((1, 1, 3))
        varas = np.array((0.26862954, 0.26130258, 0.27577711)).reshape((1, 1, 3))
        frame = (frame * varas + mean) * 255
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame_img = Image.fromarray(frame)

        grid_size = 512 // res
        frame_img = draw_grid_on_image(frame_img, grid_size)

        draw = ImageDraw.Draw(frame_img)
        if i == frame_max:
            max_pixel_pos = (w_max * grid_size, h_max * grid_size)
            draw.rectangle([max_pixel_pos, (max_pixel_pos[0] + grid_size, max_pixel_pos[1] + grid_size)], outline="red", width=2)
        if i == frame_min:
            min_pixel_pos = (w_min * grid_size, h_min * grid_size)
            draw.rectangle([min_pixel_pos, (min_pixel_pos[0] + grid_size, min_pixel_pos[1] + grid_size)], outline="blue", width=2)

        if i == query_frame_index:
            query_pixel_pos = (query_w * grid_size, query_h * grid_size)
            draw.rectangle([query_pixel_pos, (query_pixel_pos[0] + grid_size, query_pixel_pos[1] + grid_size)], outline="yellow", width=2)

        video_frames.append(frame_img) 

    save_path = "/visualization/correspondence_with_query"
    os.makedirs(save_path, exist_ok=True)
    video_save_path = os.path.join(save_path, f'self-attn-{place_in_unet}-{step}-query-frame0-h{h_index}-w{w_index}.gif')

    save_gif_mp4_folder_type(video_frames, video_save_path, save_gif=False)  




class ST_Layout_Attn_Control(AttentionControl, abc.ABC):

    def __init__(self, end_step=15, total_steps=50, step_idx=None, text_cond=None, sreg_maps=None, creg_maps=None, reg_sizes=None,reg_sizes_c=None, time_steps=None,clip_length=None,attention_type=None):
        """
        Spatial-Temporal Layout-guided Attention (ST-Layout Attn) for Stable-Diffusion model
        note: without vis cross attention weight function.
        Args:
            end_step: the step to end st-layout attn control
            total_steps: the total number of steps
            step_idx: list the steps to apply mutual self-attention control
            text_cond: discrete text embedding for each region.
            sreg_maps: spatial-temporal self-attention qk condition maps.
            creg_maps: cross-attention qk condition maps
            reg_sizes/reg_sizes_c: size regularzation maps for each instance in self_attn/cross_attention
            clip_length: frames len of video
            attention_type: FullyFrameAttention_sliced_attn/FullyFrameAttention/SparseCausalAttention
        """
        super().__init__()
        self.total_steps = total_steps
        self.step_idx = list(range(0, end_step))
        self.total_infer_steps = 50
        self.text_cond = text_cond
        self.sreg_maps = sreg_maps
        self.creg_maps = creg_maps
        self.reg_sizes = reg_sizes
        self.reg_sizes_c = reg_sizes_c
        self.clip_length = clip_length
        self.attention_type = attention_type
        self.sreg = .3
        self.creg = 1.
        self.count = 0
        self.reg_part = .3
        self.time_steps = time_steps
        print("Modulated Ctrl at denoising steps: ", self.step_idx)

    def forward(self, sim, is_cross, place_in_unet, **kwargs):
        """
        Attention forward function
        """
        #print("self.cur_step",self.cur_step)

        if self.cur_step not in self.step_idx:
            return super().forward(sim, is_cross, place_in_unet, **kwargs)


        ### sim for  "SparseCausalAttention": (frames, heads=8,res, 2*res)
        ### sim for  "FullyFrameAttention" : 1, heads, frame*res,frane*res  [1, 8, 12288, 12288])
        num_heads = sim.shape[1]
        if num_heads == 1:
           self.attention_type == "FullyFrameAttention_sliced_attn"

        treg = torch.pow((self.time_steps[self.cur_step]-1)/1000, 5)        

        
        if not is_cross:
            min_value = sim.min(-1)[0].unsqueeze(-1)
            max_value = sim.max(-1)[0].unsqueeze(-1) 
            if self.attention_type == "SparseCausalAttention":
                mask = self.sreg_maps[sim.size(2)].repeat(1,num_heads,1,1)
                size_reg = self.reg_sizes[sim.size(2)].repeat(1,num_heads,1,1)
            elif self.attention_type ==  "FullyFrameAttention":
                mask = self.sreg_maps[sim.size(2)//self.clip_length].repeat(1,num_heads,1,1)
                size_reg = self.reg_sizes[sim.size(2)//self.clip_length].repeat(1,num_heads,1,1)
            elif self.attention_type ==  "FullyFrameAttention_sliced_attn":
                mask = self.sreg_maps[sim.size(2)//self.clip_length]
                size_reg = self.reg_sizes[sim.size(2)//self.clip_length]

            else:
                print("unknown attention type")
                exit()
            # if place_in_unet == "up" and res == 32:
            #     # h_index 11 w_index =15
            #     show_self_attention_comp(sim,video=self.video,h_index=11,w_index=15,res=32,frames=self.clip_length,place_in_unet="up",step=self.cur_step)
            #if place_in_unet == "up" and res == 8:
            #    identify_self_attention_max_min(sim,video=self.video,h_index=3,w_index=4,res=8,frames=self.clip_length,place_in_unet="up",step=self.cur_step)
            
            sim += (mask>0)*size_reg*self.sreg*treg*(max_value-sim)
            sim -= ~(mask>0)*size_reg*self.sreg*treg*(sim-min_value) 

        else:
            
            min_value = sim.min(-1)[0].unsqueeze(-1)
            max_value = sim.max(-1)[0].unsqueeze(-1)  
            mask = self.creg_maps[sim.size(2)].repeat(1,num_heads,1,1)
            size_reg = self.reg_sizes_c[sim.size(2)].repeat(1,num_heads,1,1)

            sim += (mask>0)*size_reg*self.creg*treg*(max_value-sim)
            sim -= ~(mask>0)*size_reg*self.creg*treg*(sim-min_value)
            
        self.count +=1 
        return  sim




class Attention_Record_Processor(AttentionStore, abc.ABC):
    """ record ddim inversion self attention and cross attention """

    def __init__(self, additional_attention_store: AttentionStore =None,save_self_attention: bool=True,disk_store=False):
        super(Attention_Record_Processor, self).__init__(
            save_self_attention=save_self_attention,
            disk_store=disk_store)
        self.additional_attention_store = additional_attention_store
        self.attention_position_counter_dict = {
            'down_cross': 0,
            'mid_cross': 0,
            'up_cross': 0,
            'down_self': 0,
            'mid_self': 0,
            'up_self': 0,
        }

        #print("Modulated Ctrl at denoising steps: ", self.step_idx)
    
    def update_attention_position_dict(self, current_attention_key):
        self.attention_position_counter_dict[current_attention_key] +=1


    def forward(self, sim, is_cross: bool, place_in_unet: str,**kwargs):
        super(Attention_Record_Processor, self).forward(sim, is_cross, place_in_unet,**kwargs)
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        
        self.update_attention_position_dict(key)
    
        return sim

    
    def between_steps(self):

        super().between_steps()
        self.step_store = self.get_empty_store()
        
        self.attention_position_counter_dict = {
            'down_cross': 0,
            'mid_cross': 0,
            'up_cross': 0,
            'down_self': 0,
            'mid_self': 0,
            'up_self': 0,
        }        
        return 



class ST_Layout_Attn_ControlEdit(AttentionStore, abc.ABC):
    def __init__(self, end_step=15, total_steps=50, step_idx=None, text_cond=None, sreg_maps=None, creg_maps=None, reg_sizes=None,reg_sizes_c=None, 
                 time_steps=None,
                 clip_length=None,attention_type=None,
                 additional_attention_store: AttentionStore =None,                 
                 save_self_attention: bool=True,
                 disk_store=False,
                 video = None,
                 ):
        """
        Spatial-Temporal Layout-guided Attention (ST-Layout Attn) for Stable-Diffusion model
        note: with vis cross attention weight function.
        Args:
            end_step: the step to end st-layout attn control
            total_steps: the total number of steps
            step_idx: list the steps to apply mutual self-attention control
            text_cond: discrete text embedding for each region.
            sreg_maps: spatial-temporal self-attention qk condition maps.
            creg_maps: cross-attention qk condition maps
            reg_sizes/reg_sizes_c: size regularzation maps for each instance in self_attn/cross_attention
            clip_length: frames len of video
            attention_type: FullyFrameAttention_sliced_attn/FullyFrameAttention/SparseCausalAttention
        """
        super(ST_Layout_Attn_ControlEdit, self).__init__(
            save_self_attention=save_self_attention,
            disk_store=disk_store)
        self.total_steps = total_steps
        self.step_idx = list(range(0, end_step))
        self.total_infer_steps = 50
        self.text_cond = text_cond
        self.sreg_maps = sreg_maps
        self.creg_maps = creg_maps
        self.reg_sizes = reg_sizes
        self.reg_sizes_c = reg_sizes_c
        self.clip_length = clip_length
        self.attention_type = attention_type
        self.sreg = .3
        self.creg = 1.
        self.count = 0
        self.reg_part = .3
        self.time_steps = time_steps
        self.additional_attention_store = additional_attention_store
        self.attention_position_counter_dict = {
            'down_cross': 0,
            'mid_cross': 0,
            'up_cross': 0,
            'down_self': 0,
            'mid_self': 0,
            'up_self': 0,
        }
        self.video = video
    
    def update_attention_position_dict(self, current_attention_key):
        self.attention_position_counter_dict[current_attention_key] +=1


    def forward(self, sim, is_cross: bool, place_in_unet: str,**kwargs):
        super(ST_Layout_Attn_ControlEdit, self).forward(sim, is_cross, place_in_unet,**kwargs)

        # print("self.cur_step",self.cur_step)
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        
        self.update_attention_position_dict(key)

        if self.cur_step not in self.step_idx:
            return sim
        

        num_heads = sim.shape[1]
        if num_heads == 1:
           self.attention_type == "FullyFrameAttention_sliced_attn"

        treg = torch.pow((self.time_steps[self.cur_step]-1)/1000, 5)        


        if not is_cross:
            ## Modulate self-attention
            min_value = sim.min(-1)[0].unsqueeze(-1)
            max_value = sim.max(-1)[0].unsqueeze(-1) 

            if self.attention_type == "SparseCausalAttention":
                mask = self.sreg_maps[sim.size(2)].repeat(1,num_heads,1,1)
                size_reg = self.reg_sizes[sim.size(2)].repeat(1,num_heads,1,1)
            elif self.attention_type ==  "FullyFrameAttention":
                mask = self.sreg_maps[sim.size(2)//self.clip_length].repeat(1,num_heads,1,1)
                size_reg = self.reg_sizes[sim.size(2)//self.clip_length].repeat(1,num_heads,1,1)
            elif self.attention_type ==  "FullyFrameAttention_sliced_attn":
                mask = self.sreg_maps[sim.size(2)//self.clip_length]
                size_reg = self.reg_sizes[sim.size(2)//self.clip_length]

            else:
                print("unknown attention type")
                exit()

            sim += (mask>0)*size_reg*self.sreg*treg*(max_value-sim)
            sim -= ~(mask>0)*size_reg*self.sreg*treg*(sim-min_value)  
            
        else:
            #Modulate cross-attention

            min_value = sim.min(-1)[0].unsqueeze(-1)
            max_value = sim.max(-1)[0].unsqueeze(-1) 
            mask = self.creg_maps[sim.size(2)].repeat(1,num_heads,1,1)
            size_reg = self.reg_sizes_c[sim.size(2)].repeat(1,num_heads,1,1)
            sim += (mask>0)*size_reg*self.creg*treg*(max_value-sim)
            sim -= ~(mask>0)*size_reg*self.creg*treg*(sim-min_value)
        self.count +=1 
        return  sim

    
    def between_steps(self):

        super().between_steps()
        self.step_store = self.get_empty_store()
        
        self.attention_position_counter_dict = {
            'down_cross': 0,
            'mid_cross': 0,
            'up_cross': 0,
            'down_self': 0,
            'mid_self': 0,
            'up_self': 0,
        }        
        return 
