"""
Additional visualization functions for cross-attention per head/layer.
This module provides enhanced visualization capabilities without modifying the original attention_util.py
"""

import os
import math
import numpy as np
import torch
import cv2
from PIL import Image
from typing import List
import video_diffusion.prompt_attention.ptp_utils as ptp_utils


def tensor_to_pil(image_tensor):
    """Convert tensor to PIL Image"""
    image_tensor = image_tensor.cpu()
    image_tensor = image_tensor.permute(1, 2, 0)
    image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())
    image_array = np.uint8(255 * image_tensor)
    image_pil = Image.fromarray(image_array)
    return image_pil


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
    vis = show_cam_on_image(image, image_relevance.cpu().numpy())
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def show_all_cross_attention_maps(tokenizer, prompts, org_images, attention_store,
                                  vis_frames: List[int], select: int = 0, save_path=None):
    """
    Visualize cross-attention maps for each layer, map, and head separately.

    Args:
        tokenizer: The tokenizer to decode tokens
        prompts: Text prompts (str or list)
        org_images: Original video frames tensor (F, C, H, W)
        attention_store: AttentionStore instance with accumulated attention
        vis_frames: List of frame indices to visualize
        select: Which prompt to use (for multi-prompt case)
        save_path: Directory to save visualizations
    """
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
