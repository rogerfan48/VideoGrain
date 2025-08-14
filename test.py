import os
from glob import glob
import copy
from typing import Optional,Dict
from tqdm.auto import tqdm
from omegaconf import OmegaConf
import click

import torch
import torch.utils.data
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDIMInverseScheduler,
)
from diffusers.utils.import_utils import is_xformers_available
from transformers import AutoTokenizer, CLIPTextModel
from einops import rearrange

from video_diffusion.models.unet_3d_condition import UNetPseudo3DConditionModel
#from video_diffusion.models.unet import UNet3DConditionModel
from video_diffusion.data.dataset import ImageSequenceDataset
from video_diffusion.common.util import get_time_string, get_function_args
from video_diffusion.common.logger import get_logger_config_path
from video_diffusion.common.image_util import log_train_samples,log_infer_samples,save_tensor_images_and_video,visualize_check_downsample_keypoints,sample_trajectories,save_videos_grid,sample_trajectories_new
from video_diffusion.common.instantiate_from_config import instantiate_from_config
from video_diffusion.pipelines.validation_loop import SampleLogger

# logger = get_logger(__name__)

from video_diffusion.models.controlnet3d import ControlNetModel
from annotator.util import get_control, HWC3
import numpy as np
import imageio
import torchvision
import cv2
from torchvision import transforms
from PIL import Image
import time

def collate_fn(examples):
    """Concat a batch of sampled image in dataloader
    """
    batch = {
        "prompt_ids": torch.cat([example["prompt_ids"] for example in examples], dim=0),
        "images": torch.stack([example["images"] for example in examples]),
        "masks": torch.cat([example["masks"] for example in examples]),
        "layouts": torch.cat([example["layouts"] for example in examples]),
    }
    return batch

def test(
    config: str,
    pretrained_model_path: str,
    dataset_config: Dict,
    logdir: str = None,
    editing_config: Optional[Dict] = None,
    control_config: Optional[Dict] = None,
    test_pipeline_config: Optional[Dict] = None,
    gradient_accumulation_steps: int = 1,
    seed: Optional[int] = None,
    mixed_precision: Optional[str] = "fp16",
    batch_size: int = 1,
    model_config: dict={},
    cluster_inversion_feature: bool=False,
    **kwargs

):
    args = get_function_args()

    time_string = get_time_string()
    if logdir is None:
        logdir = config.replace('config', 'result').replace('.yml', '').replace('.yaml', '')
    # logdir += f"_{time_string}"

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    if accelerator.is_main_process:
        os.makedirs(logdir, exist_ok=True)
        OmegaConf.save(args, os.path.join(logdir, "config.yml"))
    logger = get_logger_config_path(logdir)

    if seed is not None:
        set_seed(seed)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_path,
        subfolder="tokenizer",
        use_fast=False,
    )

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_path,
        subfolder="text_encoder",
    )

    vae = AutoencoderKL.from_pretrained(
        pretrained_model_path,
        subfolder="vae",
    )

    # unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")
    unet = UNetPseudo3DConditionModel.from_2d_model(
        os.path.join(pretrained_model_path, "unet"), model_config=model_config
    )

    pretrained_controlnet_path = control_config['pretrained_controlnet_path']
    controlnet= ControlNetModel.from_pretrained_2d(pretrained_controlnet_path)


    if 'target' not in test_pipeline_config:
        test_pipeline_config['target'] = 'video_diffusion.pipelines.stable_diffusion.SpatioTemporalStableDiffusionPipeline'

    pipeline = instantiate_from_config(
        test_pipeline_config,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        scheduler=DDIMScheduler.from_pretrained(
            pretrained_model_path,
            subfolder="scheduler",
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        ),
        inverse_scheduler=DDIMInverseScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler"),
        logdir=logdir,
    )
    pipeline.scheduler.set_timesteps(editing_config['num_inference_steps'])
    # pipeline.set_progress_bar_config(disable=True)
    #pipeline.print_pipeline(logger)

    if is_xformers_available():
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.requires_grad_(False)

    print("org prompt input",dataset_config["prompt"])
    print("edit prompt input",editing_config["editing_prompts"])

    prompt_ids = tokenizer(
        dataset_config["prompt"],
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids
    video_dataset = ImageSequenceDataset(**dataset_config, prompt_ids=prompt_ids)

    train_dataloader = torch.utils.data.DataLoader(
        video_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )
    train_sample_save_path = os.path.join(logdir, "infer_samples")
    log_infer_samples(save_path=train_sample_save_path, infer_dataloader=train_dataloader)

    unet, controlnet, train_dataloader  = accelerator.prepare(
        unet, controlnet, train_dataloader)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        print('use fp16')
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # These models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("video")  # , config=vars(args))
    logger.info("***** wait to fix the logger path *****")

    if editing_config is not None and accelerator.is_main_process:
        # validation_sample_logger = P2pSampleLogger(**editing_config, logdir=logdir, source_prompt=dataset_config['prompt'])
        validation_sample_logger = SampleLogger(**editing_config, logdir=logdir)

    def make_data_yielder(dataloader):
        while True:
            for batch in dataloader:
                yield batch
            accelerator.wait_for_everyone()

    train_data_yielder = make_data_yielder(train_dataloader)

    
    batch = next(train_data_yielder)
    # if editing_config.get('use_invertion_latents', False):
        # Precompute the latents for this video to align the initial latents in training and test
    assert batch["images"].shape[0] == 1, "Only support, overfiting on a single video"
    # we only inference for latents, no training




    ######precompute control condition##########

    images =  batch["images"]   # b c f h w, b=1
    b, c, f, height ,width = images.shape
    images = (images+1.0)*127.5  # norm back 
    
    ## save source video

    save_videos_grid(batch["images"].cpu(),os.path.join(logdir,"source_video.mp4"),rescale=True)

    images = rearrange(images.to(dtype=torch.float32), "b c f h w -> (b f) h w c")

    control_type = control_config['control_type']
    apply_control = get_control(control_type)

    control = []
    for i in images:
        img = i.cpu().numpy()
        i = img.astype(np.uint8)

        if control_type == 'canny':
            detected_map = apply_control(i, control_config['low_threshold'], control_config['high_threshold'])
        elif control_type == 'openpose':
            detected_map = apply_control(i, hand=control_config['hand'], face=control_config['face'])
            # keypoint.append(candidate_canvas_dict['candidate']) 
        elif control_type == 'dwpose':
            detected_map = apply_control(i, hand=control_config['hand'], face=control_config['face'])
        elif control_type == 'depth_zoe':
            detected_map = apply_control(i)
        elif control_type == 'depth':
            detected_map,_ = apply_control(i)
        elif control_type == 'hed' or control_type == 'seg':
            detected_map = apply_control(i)
        elif control_type == 'scribble':
            i = i
            detected_map = np.zeros_like(i, dtype=np.uint8)
            detected_map[np.min(i, axis=2) < control_config.value] = 255
        elif control_type == 'normal':
            _, detected_map = apply_control(i, bg_th=control_config['bg_threshold'])
        elif control_type == 'mlsd':
            detected_map = apply_control(i, control_config['value_threshold'], control_config['distance_threshold'])
        else:
            raise ValueError(control_type)
        control.append(HWC3(detected_map))

    control = np.stack(control)
    control = np.array(control).astype(np.float32) / 255.0
    control = torch.from_numpy(control).to(accelerator.device)
    control = control.unsqueeze(0) #[f h w c] -> [b f h w c ]
    control = rearrange(control, "b f h w c -> b c f h w")
    control = control.to(weight_dtype)
    batch['control'] = control

    control_save = control.cpu().float()
    
    print("save control")

    control_save_dir = os.path.join(logdir, "control")

    save_tensor_images_and_video(control_save, control_save_dir) 

    # compute optical flows and sample trajectories
    trajectories = sample_trajectories_new(os.path.join(logdir, "source_video.mp4"),accelerator.device,height,width)

    torch.cuda.empty_cache()

    for k in trajectories.keys():
        trajectories[k] = trajectories[k].to(accelerator.device)

    downsample_height, downsample_width = height//8, width//8
    # The externally specified flatten_res
    flatten_res = editing_config['flatten_res']  # This could be [1] or [1, 2], etc.

    # Generate the corresponding resolutions
    flatten_resolutions = [
        (downsample_height // factor, downsample_width // factor) 
        for factor in flatten_res
    ]

    # Update the editing_config dictionary
    editing_config['flatten_res'] = flatten_resolutions
    print('flatten res:',editing_config['flatten_res'])
    all_start = time.time()
    ###ddim inversion scheduler end

    if editing_config['use_freeu']:
        from video_diffusion.prompt_attention.free_lunch_utils import apply_freeu
        apply_freeu(pipeline, b1=1.2, b2=1.5, s1=1.0, s2=1.0)
    if editing_config.get('use_invertion_latents', False):
        # Precompute the latents for this video to align the initial latents in training and test
        logger.info("use inversion latents")
        assert batch["images"].shape[0] == 1, "Only support, overfiting on a single video"
        latents, attn_inversion_dict = pipeline.prepare_latents_ddim_inverted(
            image=rearrange(batch["images"].to(dtype=weight_dtype), "b c f h w -> (b f) c h w"),
            batch_size = 1,
            source_prompt = dataset_config.prompt,
            do_classifier_free_guidance=True,  
            control=batch['control'], controlnet_conditioning_scale=control_config['controlnet_conditioning_scale'], 
            use_pnp=editing_config['use_pnp'],
            cluster_inversion_feature=editing_config.get('cluster_inversion_feature', False),
            trajs=trajectories,
            old_qk=editing_config["old_qk"],
            flatten_res=editing_config['flatten_res']
            )

        batch['ddim_init_latents'] = latents
        print("use inversion latents")

    else:
        batch['ddim_init_latents'] = None


    ########### end of code for ddim inversion###########
    vae.eval()
    text_encoder.eval()
    unet.eval()
    controlnet.eval()

    # with accelerator.accumulate(unet):
    # Convert images to latent space
    images = batch["images"].to(dtype=weight_dtype)
    images = rearrange(images, "b c f h w -> (b f) c h w")

    masks = batch["masks"].to(dtype=weight_dtype)
    b = batch_size
    masks = rearrange(masks, f"c f h w -> {b} c f h w")

    layouts = batch["layouts"].to(dtype=weight_dtype) #layouts = f s c h w

    if accelerator.is_main_process:

        if validation_sample_logger is not None:
            unet.eval()
            validation_sample_logger.log_sample_images(
                image=images, # torch.Size([8, 3, 512, 512])
                masks = masks,
                layouts = layouts,
                pipeline=pipeline,
                device=accelerator.device,
                step=0,
                latents = batch['ddim_init_latents'],
                control = batch['control'],
                controlnet_conditioning_scale = control_config['controlnet_conditioning_scale'],
                blending_percentage = editing_config["blending_percentage"],
                trajs=trajectories,
                flatten_res = editing_config['flatten_res'],
                negative_prompt=[dataset_config['negative_promot']],
                source_prompt=dataset_config.prompt,
                inject_step=editing_config["inject_step"],
                old_qk=editing_config["old_qk"],
                use_pnp = editing_config['use_pnp'],
                cluster_inversion_feature = editing_config.get('cluster_inversion_feature', False),
                vis_cross_attn = editing_config.get('vis_cross_attn', False), 
                attn_inversion_dict = attn_inversion_dict,
            )

    accelerator.end_training()    

@click.command()
@click.option("--config", type=str, default="config/shape/exp_config/single_object/tennis_3.yaml")
def run(config):
    Omegadict = OmegaConf.load(config)
    if 'unet' in os.listdir(Omegadict['pretrained_model_path']):
        test(config=config, **Omegadict)
    else:
        # Go through all ckpt if possible
        checkpoint_list = sorted(glob(os.path.join(Omegadict['pretrained_model_path'], 'checkpoint_*')))
        print('checkpoint to evaluate:')
        for checkpoint in checkpoint_list:
            epoch = checkpoint.split('_')[-1]

        for checkpoint in tqdm(checkpoint_list):
            epoch = checkpoint.split('_')[-1]
            if 'pretrained_epoch_list' not in Omegadict or int(epoch) in Omegadict['pretrained_epoch_list']:
                print(f'Evaluate {checkpoint}')
                # Update saving dir and ckpt
                Omegadict_checkpoint = copy.deepcopy(Omegadict)
                Omegadict_checkpoint['pretrained_model_path'] = checkpoint

                if 'logdir' not in Omegadict_checkpoint:
                    logdir = config.replace('config', 'result').replace('.yml', '').replace('.yaml', '')
                    logdir +=  f"/{os.path.basename(checkpoint)}"

                Omegadict_checkpoint['logdir'] = logdir
                print(f'Saving at {logdir}')

                test(config=config, **Omegadict_checkpoint)


if __name__ == "__main__":
    run()
