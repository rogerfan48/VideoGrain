"""
Code of attention storer AttentionStore, which is a base class for attention editor in attention_util.py

"""

import abc
import os
import copy
import torch
from video_diffusion.common.util import get_time_string
from einops import rearrange
from typing import Any, Callable, Dict, List, Optional, Union

class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        """I guess the diffusion of google has some unconditional attention layer
        No unconditional attention layer in Stable diffusion

        Returns:
            _type_: _description_
        """
        # return self.num_att_layers if config_dict['LOW_RESOURCE'] else 0
        return 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
        # raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            # For classifier-free guidance scale!=1
            #print("half forward")
            h = attn.shape[0]
            if h == 1:
                #print("sliced attn")
                attn = self.forward(attn, is_cross, place_in_unet)
                self.sliced_attn_head_count+=1
                if self.sliced_attn_head_count == 8:
                    self.cur_att_layer += 1
                    self.sliced_attn_head_count = 0
            else:
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
                self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers-10:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()      

        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self, 
                 ):
        self.LOW_RESOURCE = False # assume the edit have cfg
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.sliced_attn_head_count = 0



class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        
        if attn.shape[2] <= 32 ** 2:  # 1024
            # Initialize list if not exists
            if key not in self.step_store:
                self.step_store[key] = []
            
            # Store the full attention tensor (including all heads)
            append_tensor = attn.cpu().detach()
            self.step_store[key].append(copy.deepcopy(append_tensor))
            
            # Only print for cross-attention to reduce noise
            if is_cross:
                phase_msg = f"[{self.phase.upper()}]" if hasattr(self, 'phase') else "[INVERSION]"
                print(f"{phase_msg} {key}: map {len(self.step_store[key])}, shape: {attn.shape}, res: {int(attn.shape[2]**0.5)}")

        return attn

    def between_steps(self):
        if self.phase == "generation":
            self.generation_steps += 1
            print(f"[GENERATION] Accumulating attention at step {self.generation_steps}")
            
            # Always accumulate during generation
            if len(self.attention_store) == 0:
                self.attention_store = self.step_store
            else:
                for key in self.step_store:
                    for i in range(len(self.step_store[key])):
                        if key not in self.attention_store:
                            self.attention_store[key] = []
                        if i >= len(self.attention_store[key]):
                            self.attention_store[key].append(self.step_store[key][i])
                        else:
                            self.attention_store[key][i] += self.step_store[key][i]
        else:
            self.inversion_steps += 1
            print(f"[INVERSION] Step {self.inversion_steps} - not storing for visualization")
            
        self.step_store = {}
        self.cur_step += 1
    
    def set_phase(self, phase):
        """Manually set the current phase (inversion or generation)"""
        if phase in ["inversion", "generation"]:
            print(f"[PHASE] Manually switching to {phase} phase")
            self.phase = phase
            if phase == "generation":
                # Clear attention store for generation phase
                self.attention_store = {}
                self.generation_steps = 0
            elif phase == "inversion":
                self.inversion_steps = 0

    def get_average_attention(self):
        """divide the attention map value by the number of steps used for generation attention"""       
        # Use the actual number of generation steps collected
        actual_steps_collected = max(1, self.generation_steps)
            
        print(f"[SUMMARY] Using {actual_steps_collected} generation steps for averaging")
        print(f"[SUMMARY] Processing {len(self.attention_store)} attention keys")
        
        average_attention = {}
        for key in self.attention_store:
            average_attention[key] = []
            if 'cross' in key:  # Only show details for cross-attention
                print(f"[SUMMARY] {key}: {len(self.attention_store[key])} attention maps")
            for item in self.attention_store[key]:
                average_attention[key].append(item / actual_steps_collected)
        return average_attention

    def aggregate_attention(self, from_where: List[str], res: int, is_cross: bool, element_name='attn') -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        out = []
        num_pixels = res ** 2
        attention_maps = self.get_average_attention()
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                print('is cross',is_cross)
                print('item',item.shape)
                #cross (t,head,res^2,77)
                #self (head,t, res^2,res^2)
                if is_cross:
                    t, h, res_sq, token = item.shape
                    if item.shape[2] == num_pixels:
                        cross_maps = item.reshape(t, -1, res, res, item.shape[-1])
                        out.append(cross_maps)
                else:
                    h, t, res_sq, res_sq = item.shape
                    if item.shape[2] == num_pixels:
                        self_item = item.permute(1, 0, 2, 3) #(t,head,res^2,res^2)
                        self_maps = self_item.reshape(t, h, res, res, self_item.shape[-1])
                        out.append(self_maps)
        out = torch.cat(out, dim=-4)  #average head attention
        out = out.sum(-4) / out.shape[-4]
        return out

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_cross_store()
        self.attention_store_all_step = []
        self.attention_store = {}

    def __init__(self, save_self_attention:bool=True, disk_store=False):
        super(AttentionStore, self).__init__()
        self.disk_store = disk_store
        if self.disk_store:
            time_string = get_time_string()
            path = f'./trash/attention_cache_{time_string}'
            os.makedirs(path, exist_ok=True)
            self.store_dir = path
        else:
            self.store_dir =None
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.save_self_attention = save_self_attention
        self.latents_store = []
        self.attention_store_all_step = []
        
        # New: Track different phases  
        self.phase = "inversion"  # "inversion" or "generation"
        self.inversion_steps = 0
        self.generation_steps = 0
        self.generation_start_step = None
        self.use_last_n_steps = 30  # Only use last 30 steps of generation