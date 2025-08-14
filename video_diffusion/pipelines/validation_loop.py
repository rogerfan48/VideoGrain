import os
import numpy as  np
from typing import List, Union
import PIL


import torch
import torch.utils.data
import torch.utils.checkpoint

from diffusers.pipeline_utils import DiffusionPipeline
from tqdm.auto import tqdm
from video_diffusion.common.image_util import make_grid, annotate_image
from video_diffusion.common.image_util import save_gif_mp4_folder_type
import cv2

class SampleLogger:
    def __init__(
        self,
        editing_prompts: List[str],
        clip_length: int,
        logdir: str,
        subdir: str = "sample",
        num_samples_per_prompt: int = 1,
        sample_seeds: List[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7,
        strength: float = None,
        annotate: bool = False,
        annotate_size: int = 15,
        make_grid: bool = True,
        grid_column_size: int = 2,
        layout_mask_dir: str = None,  # New parameter for the layout mask directory
        layouts_masks_orders: List[str]=None,
        stride: int = 1,
        n_sample_frame: int = 8,
        start_sample_frame: int = None,
        sampling_rate: int = 1,
        **args
        
    ) -> None:
        self.editing_prompts = editing_prompts
        self.clip_length = clip_length
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.strength = strength
        
        if sample_seeds is None:
            max_num_samples_per_prompt = int(1e5)
            if num_samples_per_prompt > max_num_samples_per_prompt:
                raise ValueError
            sample_seeds = torch.randint(0, max_num_samples_per_prompt, (num_samples_per_prompt,))
            sample_seeds = sorted(sample_seeds.numpy().tolist())
        self.sample_seeds = sample_seeds

        self.logdir = os.path.join(logdir, subdir)
        os.makedirs(self.logdir, exist_ok=True)

        self.annotate = annotate
        self.annotate_size = annotate_size
        self.make_grid = make_grid
        self.grid_column_size = grid_column_size


        self.layout_mask_dir = layout_mask_dir  # Initialize layout_mask_dir
        self.layout_mask_orders = layouts_masks_orders
        self.stride = stride
        self.n_sample_frame = n_sample_frame
        self.start_sample_frame = start_sample_frame
        self.sampling_rate = sampling_rate


    def _read_mask(self, mask_path, index: int, dest_size=(64, 64)):
        mask_path = os.path.join(mask_path, f"{index:05d}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)
        mask = cv2.resize(mask, dest_size, interpolation=cv2.INTER_NEAREST)
        mask = mask[np.newaxis, ...]
        return mask

    def get_frame_indices(self, index):
        if self.start_sample_frame is not None:
            frame_start = self.start_sample_frame + self.stride * index
        else:
            frame_start = self.stride * index
        return (frame_start + i * self.sampling_rate for i in range(self.n_sample_frame))

    def read_layout_and_merge_masks(self, index):

        layouts_all, masks_all = [],[]
        for idx,layout_mask_order_per in enumerate(self.layout_mask_orders):
            layout_ = []
            for layout_name in layout_mask_order_per:  # Loop over prompts
                frame_indices = self.get_frame_indices(index % self.clip_length)
                layout_mask_dir = os.path.join(self.layout_mask_dir, layout_name)
                mask = [self._read_mask(layout_mask_dir, i) for i in frame_indices]
                masks = np.stack(mask)
                layout_.append(masks)
            layout_ = np.stack(layout_)
            
            merged_masks = []
            for i in range(int(self.n_sample_frame)):
                merged_mask_frame = np.sum(layout_[:, i, :, :], axis=0)
                merged_mask_frame = (merged_mask_frame > 0).astype(np.uint8)    
                merged_masks.append(merged_mask_frame)

            masks = rearrange(np.stack(merged_masks), "f c h w -> c f h w")
            masks = torch.from_numpy(masks).half()

            layouts = rearrange(layout_, "s f c h w -> f s c h w")
            layouts = torch.from_numpy(layouts).half()
            layouts_all.append(layouts)
            masks_all.append(mask)
        return masks_all, layouts_all

    def log_sample_images(
        self, pipeline: DiffusionPipeline,
        device: torch.device, step: int,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        masks: Union[torch.FloatTensor, PIL.Image.Image] = None,
        layouts : Union[torch.FloatTensor, PIL.Image.Image] = None,
        latents: torch.FloatTensor = None,
        control: torch.FloatTensor = None,
        controlnet_conditioning_scale = None,
        negative_prompt: Union[str, List[str]] = None,
        blending_percentage = None,
        trajs = None,
        flatten_res = None,
        source_prompt = None,
        inject_step = None,
        old_qk = None,
        use_pnp = None,
        cluster_inversion_feature = None,
        vis_cross_attn = None,
        attn_inversion_dict = None,
    ):
        torch.cuda.empty_cache()
        samples_all = []
        attention_all = []
        # handle input image
        if image is not None:
            input_pil_images = pipeline.numpy_to_pil(tensor_to_numpy(image))[0]
            samples_all.append(input_pil_images)
            # samples_all.append([
            #                 annotate_image(image, "input sequence", font_size=self.annotate_size) for image in input_pil_images
            #             ])
        #masks_all, layouts_all = self.read_layout_and_merge_masks()
        #for idx, (prompt, masks, layouts) in enumerate(tqdm(zip(self.editing_prompts, masks_all, layouts_all), desc="Generating sample images")):
        for idx, prompt in enumerate(tqdm(self.editing_prompts, desc="Generating sample images")):
            for seed in self.sample_seeds:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)
                sequence_return = pipeline(
                    prompt=prompt,
                    image=image, # torch.Size([8, 3, 512, 512])
                    latent_mask=masks,
                    layouts = layouts,
                    strength=self.strength,
                    generator=generator,
                    num_inference_steps=self.num_inference_steps,
                    clip_length=self.clip_length,
                    guidance_scale=self.guidance_scale,
                    num_images_per_prompt=1,
                    # used in null inversion
                    control = control,
                    controlnet_conditioning_scale = controlnet_conditioning_scale,
                    latents = latents,
                    #uncond_embeddings_list = uncond_embeddings_list,
                    blending_percentage =  blending_percentage,
                    logdir = self.logdir,
                    trajs = trajs,
                    flatten_res = flatten_res,
                    negative_prompt=negative_prompt,
                    source_prompt=source_prompt,
                    inject_step=inject_step,
                    old_qk=old_qk,
                    use_pnp=use_pnp,
                    cluster_inversion_feature= cluster_inversion_feature,
                    vis_cross_attn = vis_cross_attn,
                    attn_inversion_dict=attn_inversion_dict,
                )

                sequence = sequence_return.images[0]
                torch.cuda.empty_cache()

                if self.annotate:
                    images = [
                        annotate_image(image, prompt, font_size=self.annotate_size) for image in sequence
                    ]
                else:
                    images = sequence

                if self.make_grid:
                    samples_all.append(images)
                save_path = os.path.join(self.logdir, f"step_{step}_{idx}_{seed}.gif")
                save_gif_mp4_folder_type(images, save_path)
        
        if self.make_grid:
            samples_all = [make_grid(images, cols=int(np.ceil(np.sqrt(len(samples_all))))) for images in zip(*samples_all)]
            save_path = os.path.join(self.logdir, f"step_{step}.gif")
            save_gif_mp4_folder_type(samples_all, save_path)
        return samples_all


from einops import rearrange

def tensor_to_numpy(image, b=1):
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16

    image = image.cpu().float().numpy()
    image = rearrange(image, "(b f) c h w -> b f h w c", b=b)
    return image