import os

import numpy as np
from PIL import Image
from einops import rearrange
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .transform import short_size_scale, random_crop, center_crop, offset_crop
from ..common.image_util import IMAGE_EXTENSION
import cv2

class ImageSequenceDataset(Dataset):
    def __init__(
        self,
        path: str,
        layout_mask_dir: str,
        layout_mask_order: list,
        layout_mask_item: list,
        # layout_part_mask: list = None,
        prompt_ids: torch.Tensor,
        prompt: str,
        log_dir: str,
        start_sample_frame: int=0,
        n_sample_frame: int = 8,
        sampling_rate: int = 1,
        stride: int = -1, # only used during tuning to sample a long video
        image_mode: str = "RGB",
        image_size: int = 512,
        crop: str = "center",
                
        class_data_root: str = None,
        class_prompt_ids: torch.Tensor = None,
        
        offset: dict = {
            "left": 0,
            "right": 0,
            "top": 0,
            "bottom": 0
        },
        **args
        
    ):
        self.path = path
        self.images = self.get_image_list(path)
        #
        self.layout_mask_dir = layout_mask_dir
        self.layout_mask_order = list(layout_mask_order)
        self.layout_mask_item = list(layout_mask_item)
        self.log_dir = log_dir
        ###part
        # self.layout_part_mask = list(layout_part_mask)
        
        layout_mask_dir0 = os.path.join(self.layout_mask_dir,self.layout_mask_order[0])
        self.masks_index = self.get_image_list(layout_mask_dir0)

        #
        self.n_images = len(self.images)
        self.offset = offset
        self.start_sample_frame = start_sample_frame
        if n_sample_frame < 0:
            n_sample_frame = len(self.images)        
        self.n_sample_frame = n_sample_frame
        # local sampling rate from the video
        self.sampling_rate = sampling_rate

        self.sequence_length = (n_sample_frame - 1) * sampling_rate + 1
        if self.n_images < self.sequence_length:
            raise ValueError(f"self.n_images  {self.n_images } < self.sequence_length {self.sequence_length}: Required number of frames {self.sequence_length} larger than total frames in the dataset {self.n_images }")
        
        # During tuning if video is too long, we sample the long video every self.stride globally
        self.stride = stride if stride > 0 else (self.n_images+1)
        self.video_len = (self.n_images - self.sequence_length) // self.stride + 1

        self.image_mode = image_mode
        self.image_size = image_size
        crop_methods = {
            "center": center_crop,
            "random": random_crop,
        }
        if crop not in crop_methods:
            raise ValueError
        self.crop = crop_methods[crop]

        self.prompt = prompt
        self.prompt_ids = prompt_ids
        # Negative prompt for regularization to avoid overfitting during one-shot tuning
        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_images_path = sorted(list(self.class_data_root.iterdir()))
            self.num_class_images = len(self.class_images_path)
            self.class_prompt_ids = class_prompt_ids
        
        
    def __len__(self):
        max_len = (self.n_images - self.sequence_length) // self.stride + 1
        
        if hasattr(self, 'num_class_images'):
            max_len = max(max_len, self.num_class_images)
        
        return max_len

    def __getitem__(self, index):
        return_batch = {}
        frame_indices = self.get_frame_indices(index%self.video_len)
   
        frames = [self.load_frame(i) for i in frame_indices]
        frames = self.transform(frames)
        
        layout_ = []
        print(self.layout_mask_order)
      
        for layout_name in self.layout_mask_order:
            frame_indices = self.get_frame_indices(index%self.video_len)
            layout_mask_dir = os.path.join(self.layout_mask_dir,layout_name)
            if layout_name in self.layout_mask_item:
                big_output_dir = os.path.join(self.log_dir, "masks")
                output_dir = os.path.join(big_output_dir, layout_name)
                os.makedirs(output_dir, exist_ok=True)
            mask = []
            for i in frame_indices:
                mask.append(self._read_mask(layout_mask_dir, i, output_dir, self.layout_mask_item, layout_name))
            masks = np.stack(mask)
            print(masks.shape)
            layout_.append(masks)
            
            
        layout_ = np.stack(layout_)
        merged_masks = []
        for i in range(int(self.n_sample_frame)):
            merged_mask_frame = np.sum(layout_[:,i,:,:,:], axis=0)
            merged_mask_frame = (merged_mask_frame > 0).astype(np.uint8)    
            merged_masks.append(merged_mask_frame)
        masks = rearrange(np.stack(merged_masks), "f c h w -> c f h w")
        masks = torch.from_numpy(masks).half()

        layouts = rearrange(layout_,"s f c h w -> f s c h w" )
        layouts = torch.from_numpy(layouts).half()


        return_batch.update(
            {
            "images": frames,
            "masks":masks,
            "layouts":layouts,
            "prompt_ids": self.prompt_ids,
            }
        )
    # def __getitem__(self, index):
    #     return_batch = {}
    
    #     # ---------- 影格與影像 ----------
    #     base_idx = index % self.video_len
    #     frame_indices = self.get_frame_indices(base_idx)     # ← 避免重複呼叫
    #     frames = [self.load_frame(i) for i in frame_indices]
    #     frames = self.transform(frames)
    
    #     # =========================================================
    #     # A) 原本的 layout_mask_order 流程（不動你的語意，僅移除重複取 indices）
    #     # =========================================================
    #     layout_ = []
    #     print(self.layout_mask_order)  # ← 如需就保留
    
    #     for layout_name in self.layout_mask_order:
    #         frame_indices = self.get_frame_indices(index%self.video_len)
    #         layout_mask_dir = os.path.join(self.layout_mask_dir, layout_name)
    #         mask_list = [self._read_mask(layout_mask_dir, i) for i in frame_indices]
    #         masks_np = np.stack(mask_list)                      # [F, C, H, W]
    #         layout_.append(masks_np)
    
    #     layout_ = np.stack(layout_)                              # [S, F, C, H, W]
    
    #     # 合併成一張每幀的二值蒙版（各類別 OR 起來）
    #     merged_masks = []
    #     for f in range(int(self.n_sample_frame)):
    #         merged_mask_frame = np.sum(layout_[:, f, :, :, :], axis=0)   # [C, H, W]
    #         merged_mask_frame = (merged_mask_frame > 0).astype(np.uint8)
    #         merged_masks.append(merged_mask_frame)
    
    #     masks = rearrange(np.stack(merged_masks), "f c h w -> c f h w")  # [C, F, H, W]
    #     masks = torch.from_numpy(masks).half()
    
    #     layouts = rearrange(layout_, "s f c h w -> f s c h w")           # [F, S, C, H, W]
    #     layouts = torch.from_numpy(layouts).half()
    
    #     # 先把既有項目放入回傳
    #     return_batch.update(
    #         {
    #             "images": frames,
    #             "masks": masks,
    #             "layouts": layouts,
    #             "prompt_ids": self.prompt_ids,
    #         }
    #     )
    
    #     # =========================================================
    #     # B) 新增：對 self.layout_pa 做一模一樣的處理
    #     #    - 支援兩種來源：
    #     #        1) self.layout_pa_dir + self.layout_pa（list/iterable of names）
    #     #        2) self.layout_pa 是 dict: {layout_name: dir_path}
    #     # =========================================================
    #     try:
    #         has_part = hasattr(self, "layout_part_mask") and (self.layout_part_mask is not None)
    #         if has_part:
    #             print("part")
    #             part_layout_names = list(self.layout_part_mask.keys()) if isinstance((self.layout_part_mask), dict) else list(self.layout_part_mask)

        
            
    #             pa_layout_stack = []
    #             for layout_name in part_layout_names:
    #                 frame_indices = self.get_frame_indices(index%self.video_len)
    #                 pa_dir = os.path.join(self.layout_mask_dir, layout_name)
      

    #                 pa_mask_list = [self._read_mask(pa_dir, i) for i in frame_indices]   # 每幀回傳 [C, H, W]
    #                 pa_masks_np = np.stack(pa_mask_list)                                  # [F, C, H, W]
    #                 pa_layout_stack.append(pa_masks_np)
    
    #             pa_layout_np = np.stack(pa_layout_stack)                                  # [S, F, C, H, W]
    
    #             # 做出合併後的二值 pa 蒙版
    #             pa_merged = []
    #             for f in range(int(self.n_sample_frame)):
    #                 pa_merged_frame = np.sum(pa_layout_np[:, f, :, :, :], axis=0)         # [C, H, W]
    #                 pa_merged_frame = (pa_merged_frame > 0).astype(np.uint8)
    #                 pa_merged.append(pa_merged_frame)
    
    #             pa_masks = rearrange(np.stack(pa_merged), "f c h w -> c f h w")           # [C, F, H, W]
    #             pa_masks = torch.from_numpy(pa_masks).half()
    
    #             pa_layouts = rearrange(pa_layout_np, "s f c h w -> f s c h w")            # [F, S, C, H, W]
    #             pa_layouts = torch.from_numpy(pa_layouts).half()
    #             print(f"part mask: {self.layout_part_mask}")
    #             # 放進回傳
    #             return_batch.update(
    #                 {
    #                     "part_masks": pa_masks,
    #                     "part_layouts": pa_layouts,
    #                 }
    #             )
    #     except Exception as e:
    #         # 不讓訓練中斷；如需嚴格行為，把這裡改成 raise
    #         print(f"[WARN] layout_pa 處理失敗：{e}")
    
        if hasattr(self, 'class_data_root'):
            class_index = index % (self.num_class_images - self.n_sample_frame)
            class_indices = self.get_class_indices(class_index)           
            frames = [self.load_class_frame(i) for i in class_indices]
            return_batch["class_images"] = self.tensorize_frames(frames)
            return_batch["class_prompt_ids"] = self.class_prompt_ids
        return return_batch
    
    def transform(self, frames):
        frames = self.tensorize_frames(frames)
        frames = offset_crop(frames, **self.offset)
        frames = short_size_scale(frames, size=self.image_size)
        frames = self.crop(frames, height=self.image_size, width=self.image_size)
        return frames

    @staticmethod
    def tensorize_frames(frames):
        frames = rearrange(np.stack(frames), "f h w c -> c f h w")
        return torch.from_numpy(frames).div(255) * 2 - 1

    def _read_mask(self, mask_path,index: int, output_dir: str, layout_mask_item: list, layout_name: str):
        ### read mask by pil
        
        
        mask_path = os.path.join(mask_path,f"{index:05d}.png")
        print(f"mask_path:{mask_path}")

        ### read mask by cv2
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if layout_name in layout_mask_item:
            output_path = os.path.join(output_dir,f"{index:05d}.png")
            cv2.imwrite(output_path, mask)
        mask = (mask > 0).astype(np.uint8)
        # Determine dynamic destination size
        height, width = mask.shape
        dest_size = (width // 8, height // 8)
        # Resize using nearest neighbor interpolation
        mask = cv2.resize(mask, dest_size, interpolation=cv2.INTER_NEAREST) #cv2.INTER_CUBIC
        mask = mask[np.newaxis, ...]

        return mask


    def load_frame(self, index):
        image_path = os.path.join(self.path, self.images[index])
        return Image.open(image_path).convert(self.image_mode)

    def load_class_frame(self, index):
        image_path = self.class_images_path[index]
        return Image.open(image_path).convert(self.image_mode)

    def get_frame_indices(self, index):
        if self.start_sample_frame is not None:
            frame_start = self.start_sample_frame + self.stride * index
        else:
            frame_start = self.stride * index
        return (frame_start + i * self.sampling_rate for i in range(self.n_sample_frame))

    def get_class_indices(self, index):
        frame_start = index
        return (frame_start + i  for i in range(self.n_sample_frame))

    @staticmethod
    def get_image_list(path):
        images = []
        for file in sorted(os.listdir(path)):
            if file.endswith(IMAGE_EXTENSION):
                images.append(file)
        return images
