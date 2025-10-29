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
import json
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
                print("SparseCausalAttention")
                mask = self.sreg_maps[sim.size(2)].repeat(1,num_heads,1,1)
                size_reg = self.reg_sizes[sim.size(2)].repeat(1,num_heads,1,1)
            elif self.attention_type ==  "FullyFrameAttention":
                print("FullyFrameAttention")
                mask = self.sreg_maps[sim.size(2)//self.clip_length].repeat(1,num_heads,1,1)
                size_reg = self.reg_sizes[sim.size(2)//self.clip_length].repeat(1,num_heads,1,1)
            elif self.attention_type ==  "FullyFrameAttention_sliced_attn":
                print("FullyFrameAttention_sliced_attn")
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
            # print("CrossAttention")
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
                 id_maps =None,
                 sreg_maps_class=None,
                 reg_sizes_class=None,
                 id_masks_by_res=None
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
        self.id_maps = id_maps
        self.sreg_maps_class = sreg_maps_class
        self.reg_sizes_class = reg_sizes_class
        self.id_masks_by_res = id_masks_by_res
        self._idmass_acc = {} 
    def update_attention_position_dict(self, current_attention_key):
        self.attention_position_counter_dict[current_attention_key] +=1

    # def forward(self, sim, is_cross: bool, place_in_unet: str, **kwargs):
    #     """
    #     改成嚴格語意過濾：
    #       - self-attn：只允許同語意區塊的 token 互相注意
    #       - cross-attn：只允許與對應語意的文字 token 互相注意
    #     其餘位置全部設為極小值，softmax 後即為 0。
    #     """
    #     import torch
    
    #     super(ST_Layout_Attn_ControlEdit, self).forward(sim, is_cross, place_in_unet, **kwargs)
    #     # print("ST_Layout_Attn_ControlEdit attention_util.py (strict same-semantics masking)")
    
    #     key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
    #     self.update_attention_position_dict(key)
    
    #     if self.cur_step not in self.step_idx:
    #         return sim
    
    #     B, num_heads, Q, K = sim.shape
    
    #     # 修正：原本是比較(==)，應改為指定(=)
    #     if num_heads == 1:
    #         self.attention_type = "FullyFrameAttention_sliced_attn"
    
    #     # === time factor (若之後需要再用；目前硬遮罩已不需要調制) ===
    #     # treg = torch.pow((self.time_steps[self.cur_step] - 1) / 1000, 5)
    
    #     # -------- helpers --------
    #     def _expand_to_heads(x, heads: int):
    #         """
    #         x 可能是 [B, Q, K] 或 [B, 1, Q, K] 或 [B, H, Q, K]
    #         統一回傳 [B, heads, Q, K]
    #         """
    #         if x.dim() == 3:
    #             x = x.unsqueeze(1)                         # [B,1,Q,K]
    #         if x.size(1) == 1 and heads > 1:
    #             x = x.repeat(1, heads, 1, 1)               # [B,H,Q,K]
    #         return x
    
    #     # -------- 取得語意遮罩：True=同語意可保留；False=不同語意要丟掉 --------
    #     if not is_cross:
    #         # Self-Attention：使用 sreg_maps
    #         if self.attention_type == "SparseCausalAttention":
    #             base_mask = self.sreg_maps[Q]                     # 期望形狀 ~ [B,?,Q,K]
    #         elif self.attention_type == "FullyFrameAttention":
    #             base_mask = self.sreg_maps[Q // self.clip_length]
    #         elif self.attention_type == "FullyFrameAttention_sliced_attn":
    #             base_mask = self.sreg_maps[Q // self.clip_length]
    #         else:
    #             print("[WARN] unknown attention_type, fallback to no mask for self-attn")
    #             base_mask = torch.ones((B, 1, Q, K), device=sim.device, dtype=sim.dtype)
    
    #         same_mask = _expand_to_heads((base_mask > 0), num_heads)     # [B,H,Q,K] (bool)
    
    #     else:
    #         # Cross-Attention：使用 creg_maps（query 語意對應到文字 token 的位置）
    #         # 注意：這裡假設 creg_maps[Q or K] 給的是 QxK 的語意對應遮罩
    #         # 若你的 creg_maps 是以 K 為 key，必要時替換索引（例如 creg_maps[K]）
    #         if K in self.creg_maps:
    #             base_mask = self.creg_maps[K]
    #         elif Q in self.creg_maps:
    #             base_mask = self.creg_maps[Q]
    #         else:
    #             print("[WARN] creg_maps 找不到對應尺寸，cross-attn 將不遮罩")
    #             base_mask = torch.ones((B, 1, Q, K), device=sim.device, dtype=sim.dtype)
    
    #         same_mask = _expand_to_heads((base_mask > 0), num_heads)     # [B,H,Q,K] (bool)
    
    #     # -------- 硬遮罩：不同語意一律不用（置為 -inf） --------
    #     very_small = torch.finfo(sim.dtype).min if sim.dtype.is_floating_point else -1e9
    #     very_small = torch.tensor(-1e9, device=sim.device, dtype=sim.dtype) if not sim.dtype.is_floating_point else torch.tensor(very_small, device=sim.device, dtype=sim.dtype)
    
    #     # 為避免整列都被遮到 -inf 造成 NaN：若某列完全沒 True，則對該列不套遮罩
    #     # row_any: [B,H,Q,1]
    #     row_any = same_mask.any(dim=-1, keepdim=True)
    #     safe_mask = torch.where(row_any, same_mask, torch.ones_like(same_mask, dtype=torch.bool))
    
    #     sim = sim.masked_fill(~safe_mask, very_small)
    
    #     # 若你同時還想保留原本的「拉大同語意、壓小不同語意」調制，可在硬遮罩後對同語意再做溫和推拉：
    #     # min_value = sim.min(-1)[0].unsqueeze(-1)
    #     # max_value = sim.max(-1)[0].unsqueeze(-1)
    #     # if not is_cross:
    #     #     size_reg = torch.ones_like(sim)  # 如需區域權重可接回 self.reg_sizes[...] 並 _expand_to_heads
    #     #     sim = torch.where(
    #     #         safe_mask,
    #     #         sim,  # 已保留
    #     #         sim   # 已經是 very_small，無需再處理
    #     #     )
    #     # else:
    #     #     sim = torch.where(
    #     #         safe_mask,
    #     #         sim,
    #     #         sim
    #     #     )
    
    #     self.count += 1
    #     return sim



    def _make_disjoint_masks(self, masks_raw):
        """
        將多個通道的 mask 做互斥化（one-hot），避免重疊競爭。
        masks_raw: [B, H, Q, K, C]，值為 0/1 或連續 [0,1]
        回傳相同 shape 的 one-hot mask
        """
        # 取每個 (B,H,Q,K) 在 C 維上的最大通道，做成 one-hot
        # 若允許背景，請把背景通道也放進來（例如 C=3: A, B, bg）
        with torch.no_grad():
            # 除非你確定必然互斥，不然保守起見做一次 argmax
            winner = masks_raw.argmax(dim=-1, keepdim=True)  # [B,H,Q,K,1]
            one_hot = torch.zeros_like(masks_raw)
            one_hot.scatter_(-1, winner, 1.0)
            # 同時把原本確定為 0 的位置維持 0（避免把全 0 的地方硬變成 one-hot）
            zero_mask = (masks_raw.sum(dim=-1, keepdim=True) <= 0)
            one_hot = torch.where(zero_mask, torch.zeros_like(one_hot), one_hot)
        return one_hot
    
    def _multi_region_mass_floor(self, sim, masks, taus, *, cap_delta_per_class=None, meanstd_alpha=2.0, iters=2):
        """
        sim:   [B,H,Q,K]  注意力 logits（未 softmax）
        masks: [B,H,Q,K,C] 互斥區域（先經 _make_disjoint_masks），如為單幀 K=HW，會自動對齊到全幀 K=F*HW
        taus:  List[float] 長度 C，各區域目標最小質量，sum(taus) <= 1
        cap_delta_per_class: Optional[List[float]] 長度 C，對「非本區域」位置加上限：max_c + delta
        meanstd_alpha: 區域缺席時（該 Q 上該區域 K 皆 0）對負域做 mean+alpha*std 上限
        iters: 小迭代次數（2 通常很穩）
        """
        import torch
    
        B, H, Q, K = sim.shape
        device, dtype = sim.device, sim.dtype
        C = masks.shape[-1]
    
        # ---- 1) 形狀對齊與廣播 ----
        def _ensure_nd(x, tgt_rank):
            while x.dim() < tgt_rank:
                x = x.unsqueeze(0)
            return x

        # 期望 [B,H,Q,K,C]
        masks = _ensure_nd(masks, 5)
        if masks.shape[0] != B and masks.shape[0] == 1:
            masks = masks.expand(B, *masks.shape[1:])
        if masks.shape[1] != H and masks.shape[1] == 1:
            masks = masks.expand(B, H, *masks.shape[2:])
        if masks.shape[2] != Q and masks.shape[2] == 1:
            masks = masks.expand(B, H, Q, *masks.shape[3:])

        # 若 Q 不相等，但可整除，視為單幀 mask，需要沿 Q 維重複到全幀
        Q_mask = masks.shape[2]
        if Q_mask != Q:
            if Q % Q_mask != 0:
                raise RuntimeError(f"masks Q={Q_mask} cannot be aligned to sim Q={Q} (not divisible).")
            Fq = Q // Q_mask
            masks = masks.repeat_interleave(Fq, dim=2)  # 每個 query 幀重複 Fq 次，讓 Q = F*HW

        # 若 K 不相等，但可整除，視為單幀 mask，需要沿 K 維重複到全幀
        K_mask = masks.shape[3]
        if K_mask != K:
            if K % K_mask != 0:
                raise RuntimeError(f"masks K={K_mask} cannot be aligned to sim K={K} (not divisible).")
            Fk = K // K_mask
            masks = masks.repeat_interleave(Fk, dim=3)  # 每個 key 幀重複 Fk 次，讓 K = F*HW

        # 型別與 clamp
        masks = masks.to(dtype=sim.dtype, device=sim.device).clamp(0, 1)
    
        # ---- 2) 小迭代：把每個區域的質量往 tau 推（省記憶體版；避免 [B,H,Q,K,C]）----
        taus_t = torch.tensor(taus, device=device, dtype=dtype).view(1, 1, 1, 1, C).clamp(0, 0.98)

        # 建議：K/Q 分塊大小（依顯存可調）
        K_chunk = 8192 if K >= 8192 else K

        def _logit(x):
            return torch.log(x) - torch.log1p(-x)

        for _ in range(iters):
            # 為穩定也為了省顯存，softmax 可在 fp32 做，最後轉回
            sim_max = sim.amax(dim=-1, keepdim=True)
            p = torch.softmax((sim - sim_max).to(torch.float32), dim=-1).to(dtype)  # [B,H,Q,K]

            # 1) 計算每個類別的質量 mass_c（避免 [B,H,Q,K,C]）
            # mass_list 會存 C 個 [B,H,Q,1] 張量
            mass_list = []
            for ci in range(C):
                Mi = masks[..., ci]  # [B,H,Q,K]（view/broadcast；不會真複製）
                # 逐 K-chunk 求和，降低峰值
                mass_ci = 0.0
                k0 = 0
                while k0 < K:
                    k1 = min(K, k0 + K_chunk)
                    # 局部 [B,H,Q,k] 相乘再 sum(-1) -> [B,H,Q,1]
                    mass_ci_part = (p[..., k0:k1] * Mi[..., k0:k1]).sum(dim=-1, keepdim=True)
                    mass_ci = mass_ci + mass_ci_part
                    k0 += K_chunk
                mass_ci = mass_ci.clamp(1e-6, 1 - 1e-6)
                mass_list.append(mass_ci)

            # 2) 取得每類別 delta（logit(tau) - logit(mass)）
            #    把 C 個 [B,H,Q,1] 堆成 [B,H,Q,1,C]（沒有 K，量很小）
            mass_c = torch.stack(mass_list, dim=-1)                 # [B,H,Q,1,C]
            delta_c = (_logit(taus_t) - _logit(mass_c)).clamp(-20, 20)  # [B,H,Q,1,C]

            # 3) 更新 sim：sim += sum_c delta_c * M_c
            #    逐類別、逐 K-chunk 加回，避免 [B,H,Q,K,C]
            for ci in range(C):
                Mi = masks[..., ci]            # [B,H,Q,K]
                dci = delta_c[..., ci]         # [B,H,Q,1]
                k0 = 0
                while k0 < K:
                    k1 = min(K, k0 + K_chunk)
                    # 廣播成 [B,H,Q,k] 再就地加回
                    sim[..., k0:k1] = sim[..., k0:k1] + dci * Mi[..., k0:k1]
                    k0 += K_chunk

            # 釋放暫存
            del mass_list, mass_c, delta_c, p
            # 視需要：
            # torch.cuda.empty_cache()

    
        # ---- 3) 可選：per-class cap（抑制互串；非該區域的位置 ≤ 該區域 max + δ）----
        # ---- 3) 省記憶體的 per-class cap（逐類別就地更新，避免 [B,H,Q,K,C] 中間張量）----
        if cap_delta_per_class is not None:
            deltas = torch.tensor(cap_delta_per_class, device=device, dtype=dtype).view(1, 1, 1, 1, C)

            # 用一個「很小」的負數代替 -inf，避免 fp16 下 inf 擴散；fp32 也較穩
            # 這段做最大值時只要小到不會被選到即可
            neg_big = torch.tensor(-1e30, device=device, dtype=dtype)

            # 建議：把 sim 的運算臨時切到 fp32，減少數值問題（尤其是 amax / where）
            orig_dtype = sim.dtype
            if sim.dtype == torch.float16:
                sim = sim.to(torch.float32)

            # optional：Q/K 切片尺寸（依顯存可調）
            K_chunk = 8192 if K >= 8192 else K
            Q_chunk = 8192 if Q >= 8192 else Q

            for ci in range(C):
                Mi_full = masks[..., ci]  # [B,H,Q,K]，注意：這裡是 view/broadcast，不是巨量拷貝

                # 逐 Q 分塊，降低峰值
                q0 = 0
                while q0 < Q:
                    q1 = min(Q, q0 + Q_chunk)

                    # 在 K 維做 max，但僅在 Mi_full==1 的位置考慮 sim 值；其他位置用很小的數
                    # 先按 Q 切片，避免整個 [B,H,Q,K] 同時常駐
                    sim_q = sim[:, :, q0:q1, :]             # [B,H,q,QK]
                    Mi_q  = Mi_full[:, :, q0:q1, :]         # [B,H,q,K]

                    # 再按 K 切片，降低峰值
                    # 我們要得到 max_c: [B,H,q,1] = max_K where Mi_q==1 的 sim
                    # 做法：用逐塊更新的 running max
                    max_c = None
                    k0 = 0
                    while k0 < K:
                        k1 = min(K, k0 + K_chunk)
                        sim_qk = sim_q[..., k0:k1]                        # [B,H,q,k]
                        Mi_qk  = Mi_q[..., k0:k1]                         # [B,H,q,k]
                        # 將非該區域位置置為很小數，避免被 amax 選到
                        masked = torch.where(Mi_qk > 0, sim_qk, neg_big)  # [B,H,q,k]
                        part_max = masked.amax(dim=-1, keepdim=True)      # [B,H,q,1]
                        max_c = part_max if (max_c is None) else torch.maximum(max_c, part_max)
                        k0 += K_chunk

                    # 計算該類別的 cap：max_c + delta_c
                    cap_i = max_c + deltas[..., ci]  # [B,H,q,1]

                    # 只對「非該區域」的位置施加上限（該區域內不壓）
                    # 分 K 塊就地寫回，避免巨量臨時
                    k0 = 0
                    while k0 < K:
                        k1 = min(K, k0 + K_chunk)
                        sim_qk = sim_q[..., k0:k1]            # [B,H,q,k]
                        Mi_qk  = Mi_q[..., k0:k1]             # [B,H,q,k]
                        over   = (Mi_qk <= 0) & (sim_qk > cap_i)  # 超過上限且不在該區域
                        # 就地 clamp
                        sim_qk = torch.where(over, cap_i, sim_qk)
                        sim_q[..., k0:k1] = sim_qk
                        k0 += K_chunk

                    # 回寫 Q 片
                    sim[:, :, q0:q1, :] = sim_q
                    q0 += Q_chunk

            # 如前面轉到 fp32，最後轉回原 dtype（fp16）
            if orig_dtype != sim.dtype:
                sim = sim.to(orig_dtype)
    
        # ---- 4) 兜底：對完全沒覆蓋區域且負域尖峰做 mean+alpha*std 上限 ----
        # union_mask: 是否有任一區域選到此 K 位置
        union_mask = (masks > 0).any(dim=-1).to(dtype)  # [B,H,Q,K]
        no_pos = (union_mask.sum(dim=-1, keepdim=True) <= 0)  # [B,H,Q,1]
        if no_pos.any():
            neg_mask = (1.0 - union_mask).to(dtype)
            neg_count = neg_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            neg_sum   = (sim * neg_mask).sum(dim=-1, keepdim=True)
            neg_mean  = neg_sum / neg_count
            neg_sq_sum = (sim * sim * neg_mask).sum(dim=-1, keepdim=True)
            neg_var  = (neg_sq_sum / neg_count - neg_mean * neg_mean).clamp_min(0)
            neg_std  = torch.sqrt(neg_var + 1e-12)
            cap_ms   = neg_mean + meanstd_alpha * neg_std
            # 只在「無正區域」的 Q 上、且位於負域、且超過 cap_ms 的位置壓上限
            sim = torch.where((no_pos & (neg_mask > 0) & (sim > cap_ms)), cap_ms, sim)
    
        return sim




    
    # def _mass_floor_and_suppress(self, sim, pos_mask, *, 
    #                              tau=0.60,                     # 正區域最小質量
    #                              neg_cap_delta=0.0,            # “負區域不得高於正區域最大值 + delta”
    #                              meanstd_alpha=2.0,            # 當正區域為空時，對負區域用 mean+alpha*std 的上限
    #                              is_cross=False):
    #     """
    #     sim: [B,H,Q,K] 的 logits（未 softmax）
    #     pos_mask: [B,H,Q,K] 的 0/1 mask，1 表示“正區域”（希望保留注意力）
    #     策略：
    #       (1) mass floor：在機率空間保障正區域總質量≥tau，等效於對正區域 logits 加一個常數偏置
    #       (2) no-quantile clamp：不用分位數，改成 margin/meanstd 兩階段的“安全上限”
    #           - 若正區域非空：負區域 clamp 到 (max_pos + neg_cap_delta)
    #           - 若正區域為空：用負區域的 mean + alpha*std 當上限
    #     全程不會呼叫 torch.quantile，顯存友善。
    #     """
    #     B, H, Q, K = sim.shape
    #     device = sim.device
    #     dtype  = sim.dtype
    
    #     pos_mask = (pos_mask > 0).to(dtype)               # [B,H,Q,K]
    #     neg_mask = (1.0 - pos_mask)
    
    #     # ---- (1) mass floor：正區域機率保底 τ ----
    #     # 轉到 softmax 機率空間（平移避免 overflow），再反解回 logits 偏置
    #     sim_max = sim.amax(dim=-1, keepdim=True)          # [B,H,Q,1]
    #     p = torch.softmax(sim - sim_max, dim=-1)          # [B,H,Q,K]
    #     pos_mass = (p * pos_mask).sum(dim=-1, keepdim=True).clamp_(1e-6, 1-1e-6)  # [B,H,Q,1]
    
    #     # s = exp( logit(tau) - logit(pos_mass) )；等價對正區域 logits 加 log(s)
    #     logit = lambda x: torch.log(x) - torch.log1p(-x)
    #     tau_t = torch.full_like(pos_mass, tau)
    #     log_s = (logit(tau_t) - logit(pos_mass)).clamp(-20, 20)  # 防極端溢位
    #     sim = sim + log_s * pos_mask                             # [B,H,Q,K]
    
    #     # ---- (2) no-quantile clamp：負區域上限 ----
    #     # 2a) 若正區域非空：負區域不得高於 (max_pos + delta)
    #     # 先取正區域的最大值；若正區域全空，max_pos = -inf
    #     sim_pos_only = torch.where(pos_mask > 0, sim, torch.tensor(-1e9, device=device, dtype=dtype))
    #     max_pos = sim_pos_only.amax(dim=-1, keepdim=True)         # [B,H,Q,1]
    #     cap_from_pos = max_pos + neg_cap_delta
    
    #     # 對有正區域的行，做 clamp
    #     has_pos = (pos_mask.sum(dim=-1, keepdim=True) > 0)        # [B,H,Q,1], bool
    #     sim = torch.where(has_pos & (neg_mask > 0) & (sim > cap_from_pos), cap_from_pos, sim)
    
    #     # 2b) 若正區域為空：用 mean+alpha*std 作為負區域上限（僅在該行無正區域時啟用）
    #     no_pos = ~has_pos
    #     if no_pos.any():
    #         # 負區域的均值與方差（masked）
    #         # 注意：避免把 -1e9 這類佔位參與運算，直接用 mask 加權的方式
    #         neg_count = neg_mask.sum(dim=-1, keepdim=True).clamp_(min=1.0)   # [B,H,Q,1]
    #         neg_sum   = (sim * neg_mask).sum(dim=-1, keepdim=True)
    #         neg_mean  = neg_sum / neg_count
    #         # var = E[x^2] - mean^2
    #         neg_sq_sum = (sim * sim * neg_mask).sum(dim=-1, keepdim=True)
    #         neg_var = (neg_sq_sum / neg_count - neg_mean * neg_mean).clamp_min(0.0)
    #         neg_std = torch.sqrt(neg_var + 1e-12)
    
    #         cap_from_meanstd = neg_mean + meanstd_alpha * neg_std  # [B,H,Q,1]
    
    #         sim = torch.where(no_pos & (neg_mask > 0) & (sim > cap_from_meanstd),
    #                           cap_from_meanstd, sim)
    
    #     return sim
    
    
    # def _maybe_crossing_boost(self, tau_base, percentile_base, margin_base, treg_scalar):
    #     """
    #     交錯幀加強：treg_scalar 可能是 tensor / int / float。
    #     輸入任何型別都安全，輸出三個標量（tau, neg_percentile, margin）。
    #     """
    #     try:
    #         import torch
    #         is_tensor = torch.is_tensor(treg_scalar)
    #     except Exception:
    #         is_tensor = False
    
    #     if is_tensor:
    #         # 轉成標量 float，並保證非負
    #         val = treg_scalar
    #         try:
    #             # 若不是 0-d tensor，取平均變成標量（避免形狀錯）
    #             if val.numel() > 1:
    #                 val = val.mean()
    #             val = val.detach().float().clamp_min(0).item()
    #         except Exception:
    #             # 萬一某些運算型別不支援，退回到 0
    #             val = 0.0
    #     else:
    #         # Python 數值或其他可轉 float 的型別
    #         try:
    #             val = float(treg_scalar)
    #         except Exception:
    #             val = 0.0
    #         if val < 0:
    #             val = 0.0
    
    #     # boost >= 1.0；val 越大代表越需要加強
    #     boost = val + 1.0
    
    #     # 根據 boost 放大/收斂三個超參
    #     tau = min(0.95, tau_base * boost)
    #     neg_p = min(0.995, 1.0 - (1.0 - percentile_base) / boost)
    #     margin = margin_base * boost
    #     return tau, neg_p, margin
    
    ### original
    def forward(self, sim, is_cross: bool, place_in_unet: str, height=None, width=None, clip_length=None, **kwargs):
        super(ST_Layout_Attn_ControlEdit, self).forward(sim, is_cross, place_in_unet,**kwargs)
        assert sim is not None, "❌ Error: sim is None!"
        # 如果 sim 是 Tensor，但內含 NaN / Inf，也一併檢查
        if torch.is_tensor(sim):
            assert not torch.isnan(sim).any(), "❌ Error: sim contains NaN!"
            assert torch.isfinite(sim).all(), "❌ Error: sim contains Inf or -Inf!"
        else:
            raise AssertionError(f"❌ Error: sim is not a Tensor, got {type(sim)}")
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
                print(self.attention_type)
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
    
   

    # def forward(self, sim, is_cross: bool, place_in_unet: str, height=None, width=None, clip_length=None, **kwargs):
    #     super(ST_Layout_Attn_ControlEdit, self).forward(sim, is_cross, place_in_unet, **kwargs)
    #     assert sim is not None, "❌ Error: sim is None!"
    #     # 如果 sim 是 Tensor，但內含 NaN / Inf，也一併檢查
    #     if torch.is_tensor(sim):
    #         assert not torch.isnan(sim).any(), "❌ Error: sim contains NaN!"
    #         assert torch.isfinite(sim).all(), "❌ Error: sim contains Inf or -Inf!"
    #     else:
    #         raise AssertionError(f"❌ Error: sim is not a Tensor, got {type(sim)}")   
    #     # ---------------------- 位置鍵與步驟過濾 ----------------------
    #     key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
    #     self.update_attention_position_dict(key)
    
    #     if self.cur_step not in self.step_idx:
    #         return sim
    
    #     # ---------------------- 基本參數 ----------------------
    #     num_heads = sim.shape[1]
    #     # [FIX] 原本是 '=='（比較），這裡需要「賦值」才會真正切到 sliced 邏輯
    #     if num_heads == 1:
    #         self.attention_type = "FullyFrameAttention_sliced_attn"
    
    #     treg = torch.pow((self.time_steps[self.cur_step] - 1) / 1000, 5)
    #     eps = 1e-8
    
    #     # 可調參數（若未在 __init__ 設定，採預設）
    #     edge_gamma   = getattr(self, "edge_gamma", 1.5)  # 邊緣強化倍率
    #     ff_pull_gain = getattr(self, "ff_pull",   1.0)   # 同類拉近強度
    #     ff_push_gain = getattr(self, "ff_push",   1.0)   # 異類壓制強度
    #     ff_margin    = getattr(self, "ff_margin", 0.35)  # 擴張正負類均值間距比例
    
    #     # ---------------------- Self-Attention 路徑 ----------------------
    #     if not is_cross:
    #         is_ff = (self.attention_type == "FullyFrameAttention") or (self.attention_type == "FullyFrameAttention_sliced_attn")
    
    #         # 先取 min/max（沿 K 維）；後面會重用
    #         min_value = sim.min(-1, keepdim=True)[0]
    #         max_value = sim.max(-1, keepdim=True)[0]
    
    #         # --- 取得區域/同類遮罩與區域尺度 ---
    #         if self.attention_type == "SparseCausalAttention":
    #             mask = self.sreg_maps[sim.size(2)].repeat(1, num_heads, 1, 1)
    #             size_reg = self.reg_sizes[sim.size(2)].repeat(1, num_heads, 1, 1)
    #         elif self.attention_type == "FullyFrameAttention":
    #             # Full-frame：token 數等於 單幀 token 數 * clip_length
    #             # 所以用 sim.size(2) // clip_length 回到單幀 token 對應的 map
    #             mask = self.sreg_maps[sim.size(2) // clip_length].repeat(1, num_heads, 1, 1)
    #             size_reg = self.reg_sizes[sim.size(2) // clip_length].repeat(1, num_heads, 1, 1)
    #         elif self.attention_type == "FullyFrameAttention_sliced_attn":
    #             # 你的 sliced 版本內部已對齊過，不需要 repeat
    #             mask = self.sreg_maps[sim.size(2) // clip_length]
    #             size_reg = self.reg_sizes[sim.size(2) // clip_length]
    #         else:
    #             print("unknown attention type")
    #             return sim
    
    #         # 轉 dtype 以避免隱式升精度；保持與 sim 相同 dtype（通常 fp16/bf16）
    #         same_mask = (mask > 0)
    #         same_mask_f = same_mask.to(sim.dtype)
    
    #         if is_ff:
    #             # ====================== Memory-lite（避免建 [B,H,Q,K] 級 edge_w） ======================
    #             dtype = sim.dtype
    
    #             # [MEM-2] per-row / per-col 同類比例（小張量，不會 OOM）
    #             same_ratio_q = same_mask_f.mean(-1, keepdim=True)  # [B,H,Q,1]
    #             same_ratio_k = same_mask_f.transpose(-2, -1).mean(-1, keepdim=True).transpose(-2, -1)  # [B,H,1,K]
    
    #             mean_q = same_ratio_q.mean(dim=-2, keepdim=True)   # [B,H,1,1]
    #             mean_k = same_ratio_k.mean(dim=-1, keepdim=True)   # [B,H,1,1]
    
    #             edge_q = torch.clamp((mean_q - same_ratio_q) / (mean_q + eps), min=0.0, max=1.0)  # [B,H,Q,1]
    #             edge_k = torch.clamp((mean_k - same_ratio_k) / (mean_k + eps), min=0.0, max=1.0)  # [B,H,1,K]
    
    #             # [MEM-3] margin 的小路徑（僅保留 [B,H,1,1]）
    #             pos_cnt = same_mask_f.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)              # [B,H,1,1]
    #             neg_cnt = (1.0 - same_mask_f).sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)      # [B,H,1,1]
    
    #             pos_sum = (sim * same_mask_f).sum(dim=(-2, -1), keepdim=True)                     # [B,H,1,1]
    #             neg_sum = (sim * (1.0 - same_mask_f)).sum(dim=(-2, -1), keepdim=True)             # [B,H,1,1]
    #             pos_mean = pos_sum / pos_cnt
    #             neg_mean = neg_sum / neg_cnt
    #             del pos_sum, neg_sum  # 釋放臨時小張量
    
    #             margin = ff_margin * (pos_mean - neg_mean)                                        # [B,H,1,1]
    #             del pos_mean, neg_mean
    
    #             # [MEM-4] 同類拉升：只保留一個大臨時 term_same，並用原地運算
    #             term_same = same_mask_f * (max_value - sim)                                       # [B,H,Q,K]
    #             sim.add_(ff_pull_gain * treg * term_same)                                         # 原地：基礎拉升
    
    #             if edge_gamma > 0:
    #                 # Rank-1 兩側校正：避免建立 edge_w = 1 + γ*(edge_q + edge_k)
    #                 sim.add_(ff_pull_gain * treg * edge_gamma * (edge_q * term_same))             # 原地
    #                 sim.add_(ff_pull_gain * treg * edge_gamma * (edge_k * term_same))             # 原地
    
    #             # margin：同類往上拉
    #             sim.add_(same_mask_f * margin)                                                    # 原地（broadcast）
    #             del term_same  # 釋放這步唯一大臨時
    
    #             # [MEM-6] 異類壓低：同理只保留一個大臨時 term_not
    #             not_same_f = (1.0 - same_mask_f)
    #             term_not = not_same_f * (sim - min_value)                                         # [B,H,Q,K]
    #             sim.sub_(ff_push_gain * treg * term_not)                                          # 原地：基礎下壓
    
    #             if edge_gamma > 0:
    #                 sim.sub_(ff_push_gain * treg * edge_gamma * (edge_q * term_not))              # 原地
    #                 sim.sub_(ff_push_gain * treg * edge_gamma * (edge_k * term_not))              # 原地
    
    #             # margin：異類往下推
    #             sim.sub_(not_same_f * margin)                                                     # 原地
    
    #             # 收尾釋放
    #             del term_not
    #             del same_ratio_q, same_ratio_k, mean_q, mean_k, edge_q, edge_k, margin, not_same_f
    #             # 可選：torch.cuda.empty_cache()（對峰值幫助有限，通常不必每步呼叫）
    
    #         else:
    #             # 非 Full-frame 路徑：保留你原有的簡單調制（不作邊緣/拉推）
    #             sim = sim + (same_mask) * size_reg * self.sreg * treg * (max_value - sim)
    #             sim = sim - (~same_mask) * size_reg * self.sreg * treg * (sim - min_value)
    
    #     # ---------------------- Cross-Attention 路徑 ----------------------
    #     else:
    #         min_value = sim.min(-1, keepdim=True)[0]
    #         max_value = sim.max(-1, keepdim=True)[0]
    
    #         mask = self.creg_maps[sim.size(2)].repeat(1, num_heads, 1, 1)
    #         size_reg = self.reg_sizes_c[sim.size(2)].repeat(1, num_heads, 1, 1)
    
    #         sim = sim + (mask > 0) * size_reg * self.creg * treg * (max_value - sim)
    #         sim = sim - (~(mask > 0)) * size_reg * self.creg * treg * (sim - min_value)
    
    #     self.count += 1
    #     return sim


    
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
