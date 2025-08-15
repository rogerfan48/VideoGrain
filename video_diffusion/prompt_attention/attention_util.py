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

def tensor_to_pil(image_tensor):
    # 首先确保tensor在CPU上
    image_tensor = image_tensor.cpu()
    # 将C,H,W转换为H,W,C
    image_tensor = image_tensor.permute(1, 2, 0)
    # 正规化到[0,1]
    image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())
    # 转换为255范围的uint8
    image_array = np.uint8(255 * image_tensor)
    # 创建PIL图像
    image_pil = Image.fromarray(image_array)
    return image_pil

def show_image_relevance(image_relevance, image: Image.Image, relevnace_res=16):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam
    image = tensor_to_pil(image)
    image = image.resize((relevnace_res * 8, relevnace_res * 8))
    image = np.array(image)

    image_relevance = image_relevance.reshape(relevnace_res, relevnace_res)
    
    image_relevance = image_relevance.unsqueeze(0).unsqueeze(0)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=(relevnace_res * 8, relevnace_res * 8), mode='bilinear')
    image_relevance = image_relevance.squeeze()
    
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    
    image = (image - image.min()) / (image.max() - image.min()+1e-8)
    vis = show_cam_on_image(image, image_relevance.cpu().numpy())
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis

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
        # Always use additional_attention_store if available
        if self.additional_attention_store is not None:
            self.additional_attention_store.forward(sim, is_cross, place_in_unet)
        else:
            super(Attention_Record_Processor, self).forward(sim, is_cross, place_in_unet,**kwargs)
        
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        self.update_attention_position_dict(key)
    
        return sim

    
    def between_steps(self):
        # Always use additional_attention_store if available
        if self.additional_attention_store is not None:
            self.additional_attention_store.between_steps()
        else:
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

def show_image_relevance_per_head(image_relevance, image, res):
    """Process single head attention for visualization"""
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam
    
    image = tensor_to_pil(image)
    image = image.resize((res * 8, res * 8))
    image = np.array(image)

    # Ensure image_relevance is 2D
    if image_relevance.dim() == 1:
        image_relevance = image_relevance.reshape(res, res)
    
    # Upsample relevance map to match image size
    image_relevance = image_relevance.unsqueeze(0).unsqueeze(0)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=(res * 8, res * 8), mode='bilinear')
    image_relevance = image_relevance.squeeze()
    
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    
    image = (image - image.min()) / (image.max() - image.min()+1e-8)
    vis = show_cam_on_image(image, image_relevance.numpy())
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis

def show_all_cross_attention_maps(tokenizer, prompts, org_images, attention_store: AttentionStore,
                                  vis_frames: List[int], select: int = 0, save_path=None):
    if isinstance(prompts, str):
        prompts = [prompts]
    
    attention_maps_avg = attention_store.get_average_attention()
    print(f"Available attention keys: {list(attention_maps_avg.keys())}")
    
    for key, att_maps_list in attention_maps_avg.items():
        if 'cross' not in key:
            continue
        
        print(f"Processing key: {key}, maps list length: {len(att_maps_list)}")
        
        # Check if we have any attention maps for this key
        if len(att_maps_list) == 0:
            print(f"Warning: No attention maps found for key {key}, skipping")
            continue

        num_frames = org_images.shape[0]
        # Fix: Use tokenizer properly with padding and max_length
        tokens_encoded = tokenizer(prompts[select], padding="max_length", max_length=tokenizer.model_max_length, 
                                 truncation=True, return_tensors="pt")
        num_tokens = tokens_encoded.input_ids.shape[1]  # This should be 77 for CLIP
        print(f"Prompt: '{prompts[select]}', Expected tokens: {num_tokens}")
        
        # Process each attention map in this layer
        for map_idx, attention_map in enumerate(att_maps_list):
            print(f"Processing attention map {map_idx} for {key}, shape: {attention_map.shape}")
            
            # Check if this is cross-attention by verifying last dimension
            if attention_map.shape[-1] != num_tokens:
                print(f"Skipping map {map_idx} in {key}: Expected {num_tokens} tokens, got {attention_map.shape[-1]}")
                continue
                
            num_frames_in_attn, num_heads, seq_len, _ = attention_map.shape
            
            # The sequence length represents spatial resolution per frame
            # Common resolutions: 1024=32×32, 256=16×16, 64=8×8
            res = int(math.sqrt(seq_len))
            
            if res * res != seq_len:
                print(f"Skipping map {map_idx} in {key}: seq_len {seq_len} is not a perfect square")
                continue
            
            print(f"[PROCESSING] {key} map {map_idx}: resolution {res}×{res}, {num_heads} heads, {num_frames_in_attn} frames")
            
            # For video diffusion, we need to properly handle the temporal dimension
            # Reshape to (num_frames, heads, spatial_res, spatial_res, tokens)
            try:
                attention_map = attention_map.reshape(num_frames_in_attn, num_heads, res, res, num_tokens)
                print(f"[RESHAPE SUCCESS] {key} map {map_idx}: {attention_map.shape}")
            except RuntimeError as e:
                print(f"Skipping map {map_idx} in {key} due to reshape error: {e}")
                continue

            # Process each head separately
            for head_idx in range(num_heads):
                # Create directory for this layer, map, and head
                layer_head_path = os.path.join(save_path, f"{key}_map_{map_idx}_head_{head_idx}")
                os.makedirs(layer_head_path, exist_ok=True)
                print(f"[CREATED DIR] {layer_head_path}")

                # Use the same tokenization as before
                tokens_encoded = tokenizer(prompts[select], padding="max_length", max_length=tokenizer.model_max_length, 
                                         truncation=True, return_tensors="pt")
                tokens = tokens_encoded.input_ids[0]  # Get the actual token IDs
                decoder = tokenizer.decode

                # Only process non-padding tokens (until first [PAD] token)
                actual_tokens = []
                for token_id in tokens:
                    if token_id == tokenizer.pad_token_id:
                        break
                    actual_tokens.append(token_id)
                
                print(f"[TOKEN INFO] Processing {len(actual_tokens)} actual tokens out of {len(tokens)} total tokens")
                
                # For each frame in vis_frames, create visualization
                for frame_idx in vis_frames:
                    if frame_idx >= num_frames:
                        print(f"[WARNING] Frame {frame_idx} >= {num_frames}, skipping")
                        continue
                    
                    # Map frame_idx to attention frame index
                    # If we have fewer attention frames than video frames, we need to interpolate
                    attn_frame_idx = min(frame_idx, num_frames_in_attn - 1)
                    
                    print(f"[PROCESSING FRAME] video frame {frame_idx} -> attention frame {attn_frame_idx} for {key} map {map_idx} head {head_idx}")

                    # Get attention for the specific frame and head
                    head_attention = attention_map[attn_frame_idx, head_idx]  # (res, res, num_tokens)

                    images = []
                    for token_idx in range(min(len(actual_tokens), head_attention.shape[2])):
                        token_attention = head_attention[:, :, token_idx]  # (res, res)
                        
                        # Use the frame_idx to select the corresponding original image
                        orig_image = org_images[frame_idx]
                        
                        relevance_vis = show_image_relevance_per_head(token_attention, orig_image, res)
                        relevance_vis = relevance_vis.astype(np.uint8)
                        relevance_vis_pil = Image.fromarray(relevance_vis).resize((256, 256))
                        relevance_vis_np = np.array(relevance_vis_pil)
                        
                        token_text = decoder(int(actual_tokens[token_idx]))
                        final_image = ptp_utils.text_under_image(relevance_vis_np, token_text)
                        images.append(final_image)
                    
                    if images:  # Only save if we have images
                        frame_save_path = os.path.join(layer_head_path, f'frame_{frame_idx}_cross_attn.jpg')
                        ptp_utils.view_images(np.stack(images, axis=0), save_path=frame_save_path)
                        print(f"[SAVED] {frame_save_path}")
                    else:
                        print(f"[WARNING] No images generated for frame {frame_idx}")


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
        # Always store in additional_attention_store
        if self.additional_attention_store is not None:
            self.additional_attention_store.forward(sim, is_cross, place_in_unet)
        
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
        # Always use additional_attention_store if available
        if self.additional_attention_store is not None:
            self.additional_attention_store.between_steps()
        else:
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
