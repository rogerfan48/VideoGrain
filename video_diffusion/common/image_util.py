import os
import math
import textwrap
from collections import defaultdict
import imageio
import numpy as np
from typing import Sequence
import requests
import cv2
from PIL import Image, ImageDraw, ImageFont

import torch
import torchvision
import torch.nn.functional as Fu
from torchvision import transforms
from einops import rearrange
import torchvision
import imageio

import torchvision.transforms.functional as F
import random
from scipy.ndimage import binary_dilation
import sys

IMAGE_EXTENSION = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp", ".JPEG")

FONT_URL = "https://raw.github.com/googlefonts/opensans/main/fonts/ttf/OpenSans-Regular.ttf"
FONT_PATH = "./docs/OpenSans-Regular.ttf"

np.random.seed(200)
_palette = ((np.random.random((3*255))*0.7+0.3)*255).astype(np.uint8).tolist()
_palette = [0,0,0]+_palette

def save_prediction(pred_mask,output_dir,file_name):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask.save(os.path.join(output_dir,file_name))
def colorize_mask(pred_mask):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode='RGB')
    return np.array(save_mask)
def draw_mask(img, mask, alpha=0.5, id_countour=False):
    img_mask = np.zeros_like(img)
    img_mask = img
    if id_countour:
        # very slow ~ 1s per image
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids!=0]

        for id in obj_ids:
            # Overlay color on  binary mask
            if id <= 255:
                color = _palette[id*3:id*3+3]
            else:
                color = [0,0,0]
            foreground = img * (1-alpha) + np.ones_like(img) * alpha * np.array(color)
            binary_mask = (mask == id)

            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]

            countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
            img_mask[countours, :] = 0
    else:
        binary_mask = (mask!=0)
        countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
        foreground = img*(1-alpha)+colorize_mask(mask)*alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[countours,:] = 0
        
    return img_mask.astype(img.dtype)




def pad(image: Image.Image, top=0, right=0, bottom=0, left=0, color=(255, 255, 255)) -> Image.Image:
    new_image = Image.new(image.mode, (image.width + right + left, image.height + top + bottom), color)
    new_image.paste(image, (left, top))
    return new_image


def download_font_opensans(path=FONT_PATH):
    font_url = FONT_URL
    response = requests.get(font_url)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(response.content)


def annotate_image_with_font(image: Image.Image, text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    image_w = image.width
    _, _, text_w, text_h = font.getbbox(text)
    line_size = math.floor(len(text) * image_w / text_w)

    lines = textwrap.wrap(text, width=line_size)
    padding = text_h * len(lines)
    image = pad(image, top=padding + 3)

    ImageDraw.Draw(image).text((0, 0), "\n".join(lines), fill=(0, 0, 0), font=font)
    return image


def annotate_image(image: Image.Image, text: str, font_size: int = 15):
    if not os.path.isfile(FONT_PATH):
        download_font_opensans()
    font = ImageFont.truetype(FONT_PATH, size=font_size)
    return annotate_image_with_font(image=image, text=text, font=font)


def make_grid(images: Sequence[Image.Image], rows=None, cols=None) -> Image.Image:
    if isinstance(images[0], np.ndarray):
        images = [Image.fromarray(i) for i in images]

    if rows is None:
        assert cols is not None
        rows = math.ceil(len(images) / cols)
    else:
        cols = math.ceil(len(images) / rows)

    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        if image.size != (w, h):
            image = image.resize((w, h))
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def save_images_as_gif(
    images: Sequence[Image.Image],
    save_path: str,
    loop=0,
    duration=100,
    optimize=False,
) -> None:

    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        optimize=optimize,
        loop=loop,
        duration=duration,
    )

def save_images_as_mp4(
    images: Sequence[Image.Image],
    save_path: str,
) -> None:

    writer_edit = imageio.get_writer(
        save_path,
        fps=10)
    for i in images:
        init_image = i.convert("RGB")
        writer_edit.append_data(np.array(init_image))
    writer_edit.close()


def save_tensor_images_and_video(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=10):
    os.makedirs(path, exist_ok=True)
    
    # Rearrange video tensor for easier processing
    videos = rearrange(videos, "b c t h w -> t b c h w")

    # Lists to store each frame for saving as images and creating a video
    frame_list_for_images = []

    for i, x in enumerate(videos):
        # Create a grid of images for this frame
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        
        if rescale:
            x = (x + 1.0) / 2.0  # Rescale from [-1, 1] to [0, 1]
        
        x = (x * 255).numpy().astype(np.uint8)

        # Save individual frame as image
        save_path_image = os.path.join(path, f"{i}.jpg")
        imageio.imsave(save_path_image, x)

        # Append to frame lists
        frame_list_for_images.append(x)

    # Save the frames as a video
    save_path_video = os.path.join(path, "control.mp4")
    imageio.mimwrite(save_path_video, frame_list_for_images, fps=fps)

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


def save_images_as_folder(
    images: Sequence[Image.Image],
    save_path: str,
) -> None:
    os.makedirs(save_path, exist_ok=True)
    for index, image in enumerate(images):
        init_image = image
        if len(np.array(init_image).shape) == 3:
            cv2.imwrite(os.path.join(save_path, f"{index:05d}.jpg"), np.array(init_image)[:, :, ::-1])
        else:
            cv2.imwrite(os.path.join(save_path, f"{index:05d}.jpg"), np.array(init_image))

def log_infer_samples(
    infer_dataloader,
    save_path,
    num_batch: int = 4,
    fps: int = 8,
    save_input=True,
):
    infer_samples = []
    infer_masks = []
    infer_merge_masks = []
    for idx, batch in enumerate(infer_dataloader):
        if idx >= num_batch:
            break
        infer_samples.append(batch["images"])
        infer_masks.append(batch["layouts"])
        infer_merge_masks.append(batch["masks"])

    infer_samples = torch.cat(infer_samples).numpy()
    _,_,frames,height,width = infer_samples.shape
    infer_samples = rearrange(infer_samples, "b c f h w -> b f h w c")
    print('infer_samples',infer_samples.shape)
    infer_samples = (infer_samples * 0.5 + 0.5).clip(0, 1)
    # infer_samples = numpy_batch_seq_to_pil(infer_samples)
    # infer_samples = [make_grid(images, cols=int(np.ceil(np.sqrt(len(infer_samples))))) for images in zip(*infer_samples)]
    infer_merge_masks = torch.cat(infer_merge_masks).unsqueeze(0)
    infer_masks = torch.cat(infer_masks)
    # f, s, c, h ,w

    infer_masks = rearrange(infer_masks.squeeze(2), "f s h w -> f h w s")

    # 添加一个全为0的mask到第0维
    zero_mask = torch.zeros(infer_masks.shape[0], infer_masks.shape[1], infer_masks.shape[2], 1)
    infer_masks = torch.cat((zero_mask, infer_masks), dim=-1)
    infer_masks = torch.argmax(infer_masks, axis=-1).numpy()

    masked_frames = []
    for frame_idx in range(frames):
        image = np.array(infer_samples[0][frame_idx])
        mask = infer_masks[frame_idx]
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        image = (image * 255).astype(np.uint8)
        masked_frame = draw_mask(image, mask, id_countour=False)
        masked_frames.append(masked_frame)
    #infer_samples_save = rearrange(torch.tensor(infer_samples),'b t h w c -> b c t h w')
    if save_input:
        infer_samples = numpy_batch_seq_to_pil(infer_samples)
        infer_samples = [make_grid(images, cols=int(np.ceil(np.sqrt(len(infer_samples))))) for images in zip(*infer_samples)]
        save_gif_mp4_folder_type(infer_samples, os.path.join(save_path, 'input.gif'))
    imageio.mimsave(os.path.join(save_path, 'masked_video.mp4'),masked_frames,fps=fps)
    save_videos_grid(infer_merge_masks,os.path.join(save_path, 'merged_masks.mp4'), fps=fps)



def log_train_samples(
    train_dataloader,
    save_path,
    num_batch: int = 4,
):
    train_samples = []
    for idx, batch in enumerate(train_dataloader):
        if idx >= num_batch:
            break
        train_samples.append(batch["images"])

    train_samples = torch.cat(train_samples).numpy()
    train_samples = rearrange(train_samples, "b c f h w -> b f h w c")
    train_samples = (train_samples * 0.5 + 0.5).clip(0, 1)
    train_samples = numpy_batch_seq_to_pil(train_samples)
    train_samples = [make_grid(images, cols=int(np.ceil(np.sqrt(len(train_samples))))) for images in zip(*train_samples)]
    # save_images_as_gif(train_samples, save_path)
    save_gif_mp4_folder_type(train_samples, save_path)

def log_train_reg_samples(
    train_dataloader,
    save_path,
    num_batch: int = 4,
):
    train_samples = []
    for idx, batch in enumerate(train_dataloader):
        if idx >= num_batch:
            break
        train_samples.append(batch["class_images"])

    train_samples = torch.cat(train_samples).numpy()
    train_samples = rearrange(train_samples, "b c f h w -> b f h w c")
    train_samples = (train_samples * 0.5 + 0.5).clip(0, 1)
    train_samples = numpy_batch_seq_to_pil(train_samples)
    train_samples = [make_grid(images, cols=int(np.ceil(np.sqrt(len(train_samples))))) for images in zip(*train_samples)]
    # save_images_as_gif(train_samples, save_path)
    save_gif_mp4_folder_type(train_samples, save_path)


def save_gif_mp4_folder_type(images, save_path, save_gif=True):

    
    if isinstance(images[0], np.ndarray):
        images = [Image.fromarray(i) for i in images]
    elif isinstance(images[0], torch.Tensor):
        images = [transforms.ToPILImage()(i.cpu().clone()[0]) for i in images]
    save_path_mp4 = save_path.replace('gif', 'mp4')
    save_path_folder = save_path.replace('.gif', '')
    os.makedirs(save_path_folder, exist_ok=True)
    if save_gif: save_images_as_gif(images, save_path)
    save_images_as_mp4(images, save_path_mp4)
    save_images_as_folder(images, save_path_folder)

# copy from video_diffusion/pipelines/stable_diffusion.py
def numpy_seq_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

# copy from diffusers-0.11.1/src/diffusers/pipeline_utils.py
def numpy_batch_seq_to_pil(images):
    pil_images = []
    for sequence in images:
        pil_images.append(numpy_seq_to_pil(sequence))
    return pil_images


def downsample_image(image, target_size):
    image = Image.fromarray(image)
    resized_image = image.resize(target_size, Image.ANTIALIAS)
    return np.array(resized_image)

def visualize_check_downsample_keypoints(images, keypoint_data, target_res=(32, 32),final_res=(512, 512)):
    # 预处理帧列表
    processed_frames = []

    # 遍历每一帧
    for frame_idx, frame_tensor in enumerate(images):
        # 将张量转换为 NumPy 数组
        # print("frame",frame_tensor.shape)
        frame = frame_tensor.cpu().numpy().astype('uint8')

        # 下采样图片
        downsampled_frame = downsample_image(frame, target_res)

        # 绘制关键点
        for keypoint in keypoint_data[frame_idx]:
            h_coord, w_coord = keypoint
            downsampled_frame[h_coord, w_coord] = [255, 0, 0]  # 使用红色标记关键点

        # 将处理过的帧重新调整到最终分辨率
        final_frame = downsample_image(downsampled_frame, final_res)

        # 将处理过的帧添加到列表中
        processed_frames.append(final_frame)

    # 使用 imageio 保存处理过的帧为视频
    output_video_path = "./down_sample_check_hockey.mp4"
    imageio.mimsave(output_video_path, processed_frames, fps=10)


"""optical flow and trajectories sampling"""
def preprocess(img1_batch, img2_batch, transforms, height,width):
    img1_batch = F.resize(img1_batch, size=[height, width], antialias=False)
    img2_batch = F.resize(img2_batch, size=[height, width], antialias=False)
    return transforms(img1_batch, img2_batch)

def keys_with_same_value(dictionary):
    result = {}
    for key, value in dictionary.items():
        if value not in result:
            result[value] = [key]
        else:
            result[value].append(key)

    conflict_points = {}
    for k in result.keys():
        if len(result[k]) > 1:
            conflict_points[k] = result[k]
    return conflict_points

def find_duplicates(input_list):
    seen = set()
    duplicates = set()

    for item in input_list:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)

    return list(duplicates)

def neighbors_index(point, window_size, H, W):
    """return the spatial neighbor indices"""
    t, x, y = point
    neighbors = []
    for i in range(-window_size, window_size + 1):
        for j in range(-window_size, window_size + 1):
            if i == 0 and j == 0:
                continue
            if x + i < 0 or x + i >= H or y + j < 0 or y + j >= W:
                continue
            neighbors.append((t, x + i, y + j))
    return neighbors



@torch.no_grad()
def sample_trajectories(video_path, device,height,width):
    
    from torchvision.models.optical_flow import Raft_Large_Weights
    from torchvision.models.optical_flow import raft_large

    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()

    frames, _, _ = torchvision.io.read_video(str(video_path), output_format="TCHW")
    print(f"--- frames length is : {len(frames)} --- from image_util.py (line 440)")
    clips = list(range(len(frames)))

    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    model = model.eval()

    finished_trajectories = []

    current_frames, next_frames = preprocess(frames[clips[:-1]], frames[clips[1:]], transforms, 512,512)
    list_of_flows = model(current_frames.to(device), next_frames.to(device))
    predicted_flows = list_of_flows[-1]
    print('predicted_flows',predicted_flows.shape)
    predicted_flows = predicted_flows/512

    resolutions = [64, 32, 16, 8]
    res = {}
    window_sizes = {64: 2,
                    32: 1,
                    16: 1,
                    8: 1}

    for resolution in resolutions:
        print("="*30)
        trajectories = {}
        predicted_flow_resolu = torch.round(resolution*torch.nn.functional.interpolate(predicted_flows, scale_factor=(resolution/512, resolution/512)))

        T = predicted_flow_resolu.shape[0]+1
        H = predicted_flow_resolu.shape[2]
        W = predicted_flow_resolu.shape[3]

        is_activated = torch.zeros([T, H, W], dtype=torch.bool)

        for t in range(T-1):
            flow = predicted_flow_resolu[t]
            for h in range(H):
                for w in range(W):

                    if not is_activated[t, h, w]:
                        is_activated[t, h, w] = True
                        # this point has not been traversed, start new trajectory
                        x = h + int(flow[1, h, w])
                        y = w + int(flow[0, h, w])
                        if x >= 0 and x < H and y >= 0 and y < W:
                            # trajectories.append([(t, h, w), (t+1, x, y)])
                            trajectories[(t, h, w)]= (t+1, x, y)

        conflict_points = keys_with_same_value(trajectories)
        for k in conflict_points:
            index_to_pop = random.randint(0, len(conflict_points[k]) - 1)
            conflict_points[k].pop(index_to_pop)
            for point in conflict_points[k]:
                if point[0] != T-1:
                    trajectories[point]= (-1, -1, -1) # stupid padding with (-1, -1, -1)

        active_traj = []
        all_traj = []
        for t in range(T):
            pixel_set = {(t, x//H, x%H):0 for x in range(H*W)}
            new_active_traj = []
            for traj in active_traj:
                if traj[-1] in trajectories:
                    v = trajectories[traj[-1]]
                    new_active_traj.append(traj + [v])
                    pixel_set[v] = 1
                else:
                    all_traj.append(traj)
            active_traj = new_active_traj
            active_traj+=[[pixel] for pixel in pixel_set if pixel_set[pixel] == 0]
        all_traj += active_traj

        useful_traj = [i for i in all_traj if len(i)>1]
        for idx in range(len(useful_traj)):
            if useful_traj[idx][-1] == (-1, -1, -1):
                useful_traj[idx] = useful_traj[idx][:-1]
        print("how many points in all trajectories for resolution{}?".format(resolution), sum([len(i) for i in useful_traj]))
        print("how many points in the video for resolution{}?".format(resolution), T*H*W)

        # validate if there are no duplicates in the trajectories
        trajs = []
        for traj in useful_traj:
            trajs = trajs + traj
        assert len(find_duplicates(trajs)) == 0, "There should not be duplicates in the useful trajectories."

        # check if non-appearing points + appearing points = all the points in the video
        all_points = set([(t, x, y) for t in range(T) for x in range(H) for y in range(W)])
        left_points = all_points- set(trajs)
        print("How many points not in the trajectories for resolution{}?".format(resolution), len(left_points))
        for p in list(left_points):
            useful_traj.append([p])
        print("how many points in all trajectories for resolution{} after pending?".format(resolution), sum([len(i) for i in useful_traj]))


        longest_length = max([len(i) for i in useful_traj])
        sequence_length = (window_sizes[resolution]*2+1)**2 + longest_length - 1

        seqs = []
        masks = []

        # create a dictionary to facilitate checking the trajectories to which each point belongs.
        point_to_traj = {}
        for traj in useful_traj:
            for p in traj:
                point_to_traj[p] = traj

        for t in range(T):
            for x in range(H):
                for y in range(W):
                    neighbours = neighbors_index((t,x,y), window_sizes[resolution], H, W)
                    sequence = [(t,x,y)]+neighbours + [(0,0,0) for i in range((window_sizes[resolution]*2+1)**2-1-len(neighbours))]
                    sequence_mask = torch.zeros(sequence_length, dtype=torch.bool)
                    sequence_mask[:len(neighbours)+1] = True

                    traj = point_to_traj[(t,x,y)].copy()
                    traj.remove((t,x,y))
                    sequence = sequence + traj + [(0,0,0) for k in range(longest_length-1-len(traj))]
                    sequence_mask[(window_sizes[resolution]*2+1)**2: (window_sizes[resolution]*2+1)**2 + len(traj)] = True

                    seqs.append(sequence)
                    masks.append(sequence_mask)

        seqs = torch.tensor(seqs)
        masks = torch.stack(masks)
        res["traj{}".format(resolution)] = seqs
        res["mask{}".format(resolution)] = masks
    return res


@torch.no_grad()
def sample_trajectories_new(video_path, device,height,width):
    from torchvision.models.optical_flow import Raft_Large_Weights
    from torchvision.models.optical_flow import raft_large
    print("~~~~~~~~~~~~ sample new trajectory ~~~~~~~~~~~~~~~~")
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()

    frames, _, _ = torchvision.io.read_video(str(video_path), output_format="TCHW")
    print(f"--- frames length : {len(frames)} ---")
    clips = list(range(len(frames)))
    
    #=============== raft-large estimate forward optical flow============#
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    model = model.eval()
    finished_trajectories = []

    current_frames, next_frames = preprocess(frames[clips[:-1]], frames[clips[1:]], transforms, height,width)
    list_of_flows = model(current_frames.to(device), next_frames.to(device))
    print(f"--- optical flow iterate for {len(list_of_flows)} times ---")
    predicted_flows = list_of_flows[-1]
    print(f"predicte_flow shape is : {predicted_flows.shape}") # [14, 2, 512, 512])
    #=============== raft-large estimate forward optical flow============#

    predicted_flows = predicted_flows/max(height,width)

    resolutions =[(height//8,width//8),(height//16,width//16),(height//32,width//32),(height//64,width//64)]
    #resolutions = [64, 32, 16, 8]
    res = {}
    window_sizes = {(height//8,width//8): 2,
                    (height//16,width//16): 1,
                    (height//32,width//32): 1,
                    (height//64,width//64): 1}
    
    for resolution in resolutions:
        print("="*30)
        print(resolution)
        print('window_sizes[resolution]',window_sizes[resolution])
        trajectories = {}
        height_scale_factor = resolution[0] / height
        width_scale_factor = resolution[1] / width
        predicted_flow_resolu = torch.round(max(resolution[0], resolution[1])*torch.nn.functional.interpolate(predicted_flows, scale_factor=(height_scale_factor, width_scale_factor)))
        print(f"--- predicted flow resolution shape (T, H, W) : {predicted_flow_resolu.shape}")
        T = predicted_flow_resolu.shape[0]+1
        H = predicted_flow_resolu.shape[2]
        W = predicted_flow_resolu.shape[3]

        is_activated = torch.zeros([T, H, W], dtype=torch.bool)

        for t in range(T-1):
            flow = predicted_flow_resolu[t]  # (2, H, W)  dx and dy
            for h in range(H):
                for w in range(W):

                    if not is_activated[t, h, w]:
                        is_activated[t, h, w] = True
                        # this point has not been traversed, start new trajectory
                        x = h + int(flow[1, h, w])
                        y = w + int(flow[0, h, w])
                        if x >= 0 and x < H and y >= 0 and y < W:
                            # trajectories.append([(t, h, w), (t+1, x, y)])
                            trajectories[(t, h, w)]= (t+1, x, y)    # key : (t, h, w) 當前位置 ， value : (t+1, x, y)下一) -> None:

    writer_edit = imageio.get_writer(
        save_path,
        fps=10)
    for i in images:
        init_image = i.convert("RGB")
        writer_edit.append_data(np.array(init_image))
    writer_edit.close()


def save_tensor_images_and_video(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=10):
    os.makedirs(path, exist_ok=True)
    
    # Rearrange video tensor for easier processing
    videos = rearrange(videos, "b c t h w -> t b c h w")

    # Lists to store each frame for saving as images and creating a video
    frame_list_for_images = []

    for i, x in enumerate(videos):
        # Create a grid of images for this frame
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        
        if rescale:
            x = (x + 1.0) / 2.0  # Rescale from [-1, 1] to [0, 1]
        
        x = (x * 255).numpy().astype(np.uint8)

        # Save individual frame as image
        save_path_image = os.path.join(path, f"{i}.jpg")
        imageio.imsave(save_path_image, x)

        # Append to frame lists
        frame_list_for_images.append(x)

    # Save the frames as a video
    save_path_video = os.path.join(path, "control.mp4")
    imageio.mimwrite(save_path_video, frame_list_for_images, fps=fps)

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


def save_images_as_folder(
    images: Sequence[Image.Image],
    save_path: str,
) -> None:
    os.makedirs(save_path, exist_ok=True)
    for index, image in enumerate(images):
        init_image = image
        if len(np.array(init_image).shape) == 3:
            cv2.imwrite(os.path.join(save_path, f"{index:05d}.jpg"), np.array(init_image)[:, :, ::-1])
        else:
            cv2.imwrite(os.path.join(save_path, f"{index:05d}.jpg"), np.array(init_image))

def log_infer_samples(
    infer_dataloader,
    save_path,
    num_batch: int = 4,
    fps: int = 8,
    save_input=True,
):
    infer_samples = []
    infer_masks = []
    infer_merge_masks = []
    for idx, batch in enumerate(infer_dataloader):
        if idx >= num_batch:
            break
        infer_samples.append(batch["images"])
        infer_masks.append(batch["layouts"])
        infer_merge_masks.append(batch["masks"])

    infer_samples = torch.cat(infer_samples).numpy()
    _,_,frames,height,width = infer_samples.shape
    infer_samples = rearrange(infer_samples, "b c f h w -> b f h w c")
    print('infer_samples',infer_samples.shape)
    infer_samples = (infer_samples * 0.5 + 0.5).clip(0, 1)
    # infer_samples = numpy_batch_seq_to_pil(infer_samples)
    # infer_samples = [make_grid(images, cols=int(np.ceil(np.sqrt(len(infer_samples))))) for images in zip(*infer_samples)]
    infer_merge_masks = torch.cat(infer_merge_masks).unsqueeze(0)
    infer_masks = torch.cat(infer_masks)
    # f, s, c, h ,w

    infer_masks = rearrange(infer_masks.squeeze(2), "f s h w -> f h w s")

    # 添加一个全为0的mask到第0维
    zero_mask = torch.zeros(infer_masks.shape[0], infer_masks.shape[1], infer_masks.shape[2], 1)
    infer_masks = torch.cat((zero_mask, infer_masks), dim=-1)
    infer_masks = torch.argmax(infer_masks, axis=-1).numpy()

    masked_frames = []
    for frame_idx in range(frames):
        image = np.array(infer_samples[0][frame_idx])
        mask = infer_masks[frame_idx]
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        image = (image * 255).astype(np.uint8)
        masked_frame = draw_mask(image, mask, id_countour=False)
        masked_frames.append(masked_frame)
    #infer_samples_save = rearrange(torch.tensor(infer_samples),'b t h w c -> b c t h w')
    if save_input:
        infer_samples = numpy_batch_seq_to_pil(infer_samples)
        infer_samples = [make_grid(images, cols=int(np.ceil(np.sqrt(len(infer_samples))))) for images in zip(*infer_samples)]
        save_gif_mp4_folder_type(infer_samples, os.path.join(save_path, 'input.gif'))
    imageio.mimsave(os.path.join(save_path, 'masked_video.mp4'),masked_frames,fps=fps)
    save_videos_grid(infer_merge_masks,os.path.join(save_path, 'merged_masks.mp4'), fps=fps)



def log_train_samples(
    train_dataloader,
    save_path,
    num_batch: int = 4,
):
    train_samples = []
    for idx, batch in enumerate(train_dataloader):
        if idx >= num_batch:
            break
        train_samples.append(batch["images"])

    train_samples = torch.cat(train_samples).numpy()
    train_samples = rearrange(train_samples, "b c f h w -> b f h w c")
    train_samples = (train_samples * 0.5 + 0.5).clip(0, 1)
    train_samples = numpy_batch_seq_to_pil(train_samples)
    train_samples = [make_grid(images, cols=int(np.ceil(np.sqrt(len(train_samples))))) for images in zip(*train_samples)]
    # save_images_as_gif(train_samples, save_path)
    save_gif_mp4_folder_type(train_samples, save_path)

def log_train_reg_samples(
    train_dataloader,
    save_path,
    num_batch: int = 4,
):
    train_samples = []
    for idx, batch in enumerate(train_dataloader):
        if idx >= num_batch:
            break
        train_samples.append(batch["class_images"])

    train_samples = torch.cat(train_samples).numpy()
    train_samples = rearrange(train_samples, "b c f h w -> b f h w c")
    train_samples = (train_samples * 0.5 + 0.5).clip(0, 1)
    train_samples = numpy_batch_seq_to_pil(train_samples)
    train_samples = [make_grid(images, cols=int(np.ceil(np.sqrt(len(train_samples))))) for images in zip(*train_samples)]
    # save_images_as_gif(train_samples, save_path)
    save_gif_mp4_folder_type(train_samples, save_path)


def save_gif_mp4_folder_type(images, save_path, save_gif=True):

    
    if isinstance(images[0], np.ndarray):
        images = [Image.fromarray(i) for i in images]
    elif isinstance(images[0], torch.Tensor):
        images = [transforms.ToPILImage()(i.cpu().clone()[0]) for i in images]
    save_path_mp4 = save_path.replace('gif', 'mp4')
    save_path_folder = save_path.replace('.gif', '')
    os.makedirs(save_path_folder, exist_ok=True)
    if save_gif: save_images_as_gif(images, save_path)
    save_images_as_mp4(images, save_path_mp4)
    save_images_as_folder(images, save_path_folder)

# copy from video_diffusion/pipelines/stable_diffusion.py
def numpy_seq_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

# copy from diffusers-0.11.1/src/diffusers/pipeline_utils.py
def numpy_batch_seq_to_pil(images):
    pil_images = []
    for sequence in images:
        pil_images.append(numpy_seq_to_pil(sequence))
    return pil_images


def downsample_image(image, target_size):
    image = Image.fromarray(image)
    resized_image = image.resize(target_size, Image.ANTIALIAS)
    return np.array(resized_image)

def visualize_check_downsample_keypoints(images, keypoint_data, target_res=(32, 32),final_res=(512, 512)):
    # 预处理帧列表
    processed_frames = []

    # 遍历每一帧
    for frame_idx, frame_tensor in enumerate(images):
        # 将张量转换为 NumPy 数组
        # print("frame",frame_tensor.shape)
        frame = frame_tensor.cpu().numpy().astype('uint8')

        # 下采样图片
        downsampled_frame = downsample_image(frame, target_res)

        # 绘制关键点
        for keypoint in keypoint_data[frame_idx]:
            h_coord, w_coord = keypoint
            downsampled_frame[h_coord, w_coord] = [255, 0, 0]  # 使用红色标记关键点

        # 将处理过的帧重新调整到最终分辨率
        final_frame = downsample_image(downsampled_frame, final_res)

        # 将处理过的帧添加到列表中
        processed_frames.append(final_frame)

    # 使用 imageio 保存处理过的帧为视频
    output_video_path = "./down_sample_check_hockey.mp4"
    imageio.mimsave(output_video_path, processed_frames, fps=10)


"""optical flow and trajectories sampling"""
def preprocess(img1_batch, img2_batch, transforms, height,width):
    img1_batch = F.resize(img1_batch, size=[height, width], antialias=False)
    img2_batch = F.resize(img2_batch, size=[height, width], antialias=False)
    return transforms(img1_batch, img2_batch)

# def keys_with_same_value(dictionary):
#     result = {}
#     for key, value in dictionary.items():
#         if value not in result:
#             result[value] = [key]
#         else:
#             result[value].append(key)   #key: target, value: source

#     conflict_points = {}
#     for k in result.keys():
#         if len(result[k]) > 1:
#             conflict_points[k] = result[k]
#             print(f"---conflict point : {conflict_points[k]}") # ---conflict point : [(12, 19, 63), (12, 21, 62), (12, 21, 63)]
#             path1 = result[k][0]
#             path2 = result[k][1]
#             out = dictionary[path1]
#             print(f"trajectory:{dictionary[out]}")
#     return conflict_points

def find_duplicates(input_list):
    seen = set()
    duplicates = set()

    for item in input_list:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)

    return list(duplicates)

def neighbors_index(point, window_size, H, W):
    """return the spatial neighbor indices"""
    t, x, y = point
    neighbors = []
    for i in range(-window_size, window_size + 1):
        for j in range(-window_size, window_size + 1):
            if i == 0 and j == 0:
                continue
            if x + i < 0 or x + i >= H or y + j < 0 or y + j >= W:
                continue
            neighbors.append((t, x + i, y + j))
    return neighbors



@torch.no_grad()
def sample_trajectories(video_path, device,height,width):
    
    from torchvision.models.optical_flow import Raft_Large_Weights
    from torchvision.models.optical_flow import raft_large

    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()

    frames, _, _ = torchvision.io.read_video(str(video_path), output_format="TCHW")
    print(f"--- frames length is : {len(frames)} --- from image_util.py (line 440)")
    clips = list(range(len(frames)))

    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    model = model.eval()

    finished_trajectories = []

    current_frames, next_frames = preprocess(frames[clips[:-1]], frames[clips[1:]], transforms, 512,512)
    list_of_flows = model(current_frames.to(device), next_frames.to(device))
    predicted_flows = list_of_flows[-1]
    print('predicted_flows',predicted_flows.shape)
    predicted_flows = predicted_flows/512

    resolutions = [64, 32, 16, 8]
    res = {}
    window_sizes = {64: 2,
                    32: 1,
                    16: 1,
                    8: 1}

    for resolution in resolutions:
        print("="*30)
        trajectories = {}
        predicted_flow_resolu = torch.round(resolution*torch.nn.functional.interpolate(predicted_flows, scale_factor=(resolution/512, resolution/512)))

        T = predicted_flow_resolu.shape[0]+1
        H = predicted_flow_resolu.shape[2]
        W = predicted_flow_resolu.shape[3]

        is_activated = torch.zeros([T, H, W], dtype=torch.bool)

        for t in range(T-1):
            flow = predicted_flow_resolu[t]
            for h in range(H):
                for w in range(W):

                    if not is_activated[t, h, w]:
                        is_activated[t, h, w] = True
                        # this point has not been traversed, start new trajectory
                        x = h + int(flow[1, h, w])
                        y = w + int(flow[0, h, w])
                        if x >= 0 and x < H and y >= 0 and y < W:
                            # trajectories.append([(t, h, w), (t+1, x, y)])
                            trajectories[(t, h, w)]= (t+1, x, y)

        conflict_points = keys_with_same_value(trajectories)
        for k in conflict_points:
            index_to_pop = random.randint(0, len(conflict_points[k]) - 1)
            conflict_points[k].pop(index_to_pop)
            for point in conflict_points[k]:
                if point[0] != T-1:
                    trajectories[point]= (-1, -1, -1) # stupid padding with (-1, -1, -1)


        
        active_traj = []
        all_traj = []
        for t in range(T):
            pixel_set = {(t, x//H, x%H):0 for x in range(H*W)}
            new_active_traj = []
            for traj in active_traj:
                if traj[-1] in trajectories:
                    v = trajectories[traj[-1]]
                    new_active_traj.append(traj + [v])
                    pixel_set[v] = 1
                else:
                    all_traj.append(traj)
            active_traj = new_active_traj
            active_traj+=[[pixel] for pixel in pixel_set if pixel_set[pixel] == 0]
        all_traj += active_traj

        useful_traj = [i for i in all_traj if len(i)>1]
        for idx in range(len(useful_traj)):
            if useful_traj[idx][-1] == (-1, -1, -1):
                useful_traj[idx] = useful_traj[idx][:-1]
        print("how many points in all trajectories for resolution{}?".format(resolution), sum([len(i) for i in useful_traj]))
        print("how many points in the video for resolution{}?".format(resolution), T*H*W)

        # validate if there are no duplicates in the trajectories
        trajs = []
        for traj in useful_traj:
            trajs = trajs + traj
        assert len(find_duplicates(trajs)) == 0, "There should not be duplicates in the useful trajectories."

        # check if non-appearing points + appearing points = all the points in the video
        all_points = set([(t, x, y) for t in range(T) for x in range(H) for y in range(W)])
        left_points = all_points- set(trajs)
        print("How many points not in the trajectories for resolution{}?".format(resolution), len(left_points))
        for p in list(left_points):
            useful_traj.append([p])
        print("how many points in all trajectories for resolution{} after pending?".format(resolution), sum([len(i) for i in useful_traj]))


        longest_length = max([len(i) for i in useful_traj])
        sequence_length = (window_sizes[resolution]*2+1)**2 + longest_length - 1

        seqs = []
        masks = []

        # create a dictionary to facilitate checking the trajectories to which each point belongs.
        point_to_traj = {}
        for traj in useful_traj:
            for p in traj:
                point_to_traj[p] = traj

        for t in range(T):
            for x in range(H):
                for y in range(W):
                    neighbours = neighbors_index((t,x,y), window_sizes[resolution], H, W)
                    sequence = [(t,x,y)]+neighbours + [(0,0,0) for i in range((window_sizes[resolution]*2+1)**2-1-len(neighbours))]
                    sequence_mask = torch.zeros(sequence_length, dtype=torch.bool)
                    sequence_mask[:len(neighbours)+1] = True

                    traj = point_to_traj[(t,x,y)].copy()
                    traj.remove((t,x,y))
                    sequence = sequence + traj + [(0,0,0) for k in range(longest_length-1-len(traj))]
                    sequence_mask[(window_sizes[resolution]*2+1)**2: (window_sizes[resolution]*2+1)**2 + len(traj)] = True

                    seqs.append(sequence)
                    masks.append(sequence_mask)

        seqs = torch.tensor(seqs)
        masks = torch.stack(masks)
        res["traj{}".format(resolution)] = seqs
        res["mask{}".format(resolution)] = masks
    return res



################ visualize_trajectories ######################################################################################
def visualize_trajectories(trajs, H, W, out_dir, name_prefix="traj", scale=4, make_gif=True):
    """
    Saves:
      - PNG of all trajectories overlaid: {out_dir}/{name_prefix}_all.png
      - (Optional) animated GIF per-frame: {name_prefix}.gif
    Each trajectory is drawn with a different (deterministic) color.
    """
    os.makedirs(out_dir, exist_ok=True)
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    # assign colors
    rng = np.random.RandomState(42)
    colors = rng.rand(max(1, len(trajs)), 3)

    # static PNG with whole paths
    fig, ax = plt.subplots(figsize=(W*scale/100, H*scale/100), dpi=100)
    ax.set_xlim(-0.5, W-0.5); ax.set_ylim(H-0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for i, path in enumerate(trajs):
        ys = [p[2] for p in path]
        xs = [p[1] for p in path]
        ax.plot(ys, xs)
    plt.tight_layout(pad=0)
    png_path = os.path.join(out_dir, f"{name_prefix}_all.png")
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    if make_gif:
        # frame-by-frame GIF (dots per frame)
        frames = []
        for t in range(max(p[0] for path in trajs for p in path) + 1):
            fig, ax = plt.subplots(figsize=(W*scale/100, H*scale/100), dpi=100)
            ax.set_xlim(-0.5, W-0.5); ax.set_ylim(H-0.5, -0.5)
            ax.set_xticks([]); ax.set_yticks([])
            for i, path in enumerate(trajs):
                pts_t = [(x,y) for (tt, x, y) in path if tt == t]
                if pts_t:
                    xs = [x for x, _ in pts_t]
                    ys = [y for _, y in pts_t]
                    ax.scatter(ys, xs, s=8)
            plt.tight_layout(pad=0)
            fig.canvas.draw()
            # convert to numpy image
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
            plt.close(fig)

        gif_path = os.path.join(out_dir, f"{name_prefix}.gif")
        imageio.mimsave(gif_path, frames, fps=6)

    return {"png": png_path, "gif": os.path.join(out_dir, f"{name_prefix}.gif") if make_gif else None}
######################################################################################################################
@torch.no_grad()
def sample_trajectories_new(video_path, device, height, width):
    from torchvision.models.optical_flow import Raft_Large_Weights
    from torchvision.models.optical_flow import raft_large

    print("~~~~~~~~~~~~ sample new trajectory ~~~~~~~~~~~~~~~~")
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()

    # 讀影片
    frames, _, _ = torchvision.io.read_video(str(video_path), output_format="TCHW")
    print(f"--- frames length : {len(frames)} ---")
    clips = list(range(len(frames)))

    #=============== RAFT forward optical flow ============#
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    model = model.eval()

    current_frames, next_frames = preprocess(
        frames[clips[:-1]], frames[clips[1:]], transforms, height, width
    )
    with torch.no_grad():
        list_of_flows = model(current_frames.to(device), next_frames.to(device))
    print(f"--- optical flow iterate for {len(list_of_flows)} times ---")
    predicted_flows = list_of_flows[-1].detach().cpu()   # shape: (T-1, 2, H0, W0)
    print(f"predicted_flow shape is : {predicted_flows.shape}")

    # 重要：不做 /max(height,width) 的縮放（會把數值壓到接近 0）
    # predicted_flows = predicted_flows / max(height, width)   # ❌ 不要這行

    # 多尺度
    resolutions = [
        (height // 8,  width // 8),
        (height // 16, width // 16),
        (height // 32, width // 32),
        (height // 64, width // 64),
    ]
    res = {}
    window_sizes = {
        (height // 8,  width // 8): 2,
        (height // 16, width // 16): 1,
        (height // 32, width // 32): 1,
        (height // 64, width // 64): 1,
    }

    # ================= helpers for conflict resolution ================= #
    def _cos(a, b, eps=1e-6):
        """安全版 cosine：極小向量回 -0.5 而非 -inf。"""
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < eps or nb < eps:
            return -0.5
        return float(np.dot(a, b) / (na * nb))

    def _agg_vec(vecs, mode="ema", ema_beta=0.6, min_norm=1e-6):
        """聚合多步向量：過濾小範數向量後做 mean/median/EMA。vec: [dy,dx]."""
        vecs = [v for v in vecs if np.linalg.norm(v) >= min_norm]
        if len(vecs) == 0:
            return None
        if mode == "mean":
            return np.mean(np.stack(vecs, 0), axis=0)
        elif mode == "median":
            return np.median(np.stack(vecs, 0), axis=0)
        else:  # "ema"
            v = vecs[0].astype(float)
            beta = float(ema_beta)
            for i in range(1, len(vecs)):
                v = beta * v + (1.0 - beta) * vecs[i]
            return v

    def _collect_forward_vecs_from_traj(start_key, steps, trajectories):
        """從 (t,x,y) 沿 trajectories 往未來最多 steps 步，回傳 [dy,dx] 列表（離散差）。"""
        vecs = []
        cur = start_key  # (t,x,y)
        for _ in range(steps):
            nxt = trajectories.get(cur, None)
            if not nxt or nxt == (-1, -1, -1):
                break
            dy = nxt[2] - cur[2]
            dx = nxt[1] - cur[1]
            vecs.append(np.array([dy, dx], float))
            cur = nxt
        return vecs

    def _build_reverse_index(trajectories):
        """建立反向索引： (t+1,x,y) -> (t,h,w)"""
        rev = {}
        for k, v in trajectories.items():
            if v != (-1, -1, -1):
                rev[v] = k
        return rev

    def _collect_backward_vecs_from_traj(start_key, steps, reverse_index):
        """從 (t,h,w) 往過去最多 steps 步，回傳 [dy,dx] 列表（離散差）。"""
        vecs = []
        cur = start_key  # (t,h,w)
        for _ in range(steps):
            if cur not in reverse_index:
                break
            prev = reverse_index[cur]  # (t-1, h0, w0)
            dy = cur[2] - prev[2]
            dx = cur[1] - prev[1]
            vecs.append(np.array([dy, dx], float))
            cur = prev
        return vecs

    def _flow_vec_forward(flow, t, x, y, min_norm=1e-6):
        """從浮點光流抓 forward 向量（t -> t+1），回 [dy,dx]；太小回 None。"""
        if flow is None:
            return None
        if not (0 <= t < flow.shape[0]):
            return None
        dx = float(flow[t, 0, x, y])
        dy = float(flow[t, 1, x, y])
        if abs(dx) + abs(dy) < min_norm:
            return None
        return [dy, dx]

    def _flow_vec_backward(flow, t, x, y, min_norm=1e-6):
        """來源往過去的後備：用 (t-1 -> t) 的 forward flow 當 [dy,dx]。"""
        return _flow_vec_forward(flow, t-1, x, y, min_norm=min_norm)

    def complict_track_source_flow(
        trajectories,         # dict: {(t,h,w) -> (t+1,x,y) 或 (-1,-1,-1)}
        conflict_points,      # dict: {(t+1,x,y): [(t,h1,w1), (t,h2,w2), ...]}
        T,                    # 總幀數
        flow=None,            # (T-1,2,H,W) 浮點光流；用於後備
        k_tgt=5,              # 目標往未來看的步數
        k_src=3,              # 來源往過去看的步數
        agg="ema",            # "ema" | "mean" | "median"
        ema_beta=0.6,         # ema 係數
        w_dir=1.0,            # 方向一致性權重
        w_mag=0.5,            # 幅度一致性權重
        w_pos=0.5,            # 位置一致性權重
        sigma_mag=1.0,        # 幅度一致的溫度
        sigma_pos=1.5,        # 位置一致的σ（像素）
        debug=False
    ):
        reverse_index = _build_reverse_index(trajectories)

        def best_target_vec(k):
            tk, x, y = k
            vecs = _collect_forward_vecs_from_traj(k, steps=k_tgt, trajectories=trajectories)
            v = _agg_vec(vecs, mode=agg, ema_beta=ema_beta)
            if v is None:
                fv = _flow_vec_forward(flow, tk, x, y)
                if fv is not None:
                    v = np.array(fv, float)
            return v

        def best_source_vec(src):
            t0, h, w = src
            vecs = _collect_backward_vecs_from_traj(src, steps=k_src, reverse_index=reverse_index)
            v = _agg_vec(vecs, mode=agg, ema_beta=ema_beta)
            if v is None:
                fb = _flow_vec_backward(flow, t0, h, w)
                if fb is not None:
                    v = np.array(fb, float)
            return v

        def score(src_vec, tgt_vec, src_pos, tgt_pos):
            # 方向一致
            cosv = _cos(src_vec, tgt_vec)

            # 幅度一致
            m_src = float(np.linalg.norm(src_vec))
            m_tgt = float(np.linalg.norm(tgt_vec)) if tgt_vec is not None else 0.0
            mag_pen = math.exp(-abs(m_src - m_tgt) / float(sigma_mag))

            # 位置一致（用離散一步近似：src_pos + round(src_vec) ≈ tgt_pos）
            dy, dx = src_vec
            pred_x = src_pos[0] + int(round(dx))
            pred_y = src_pos[1] + int(round(dy))
            pos_err = math.sqrt((pred_x - tgt_pos[0])**2 + (pred_y - tgt_pos[1])**2)
            pos_pen = math.exp(-(pos_err**2) / (2.0 * (float(sigma_pos)**2)))

            return w_dir * cosv + w_mag * mag_pen + w_pos * pos_pen, (cosv, mag_pen, pos_pen)

        for k, src_list in conflict_points.items():
            if debug:
                print(f"--- tgt_pt : {k} comes from {src_list}.")

            tgt_vec = best_target_vec(k)
            if tgt_vec is None:
                # 還是拿不到 → fallback：離散一步幅度最大
                best_src, best_mag = None, -math.inf
                for (t0, h, w) in src_list:
                    v = np.array([k[2] - w, k[1] - h], float)  # [dy,dx]
                    mag = float(np.linalg.norm(v))
                    if mag > best_mag:
                        best_mag, best_src = mag, (t0, h, w)
                for (t0, h, w) in src_list:
                    if (t0, h, w) != best_src and t0 != T - 1:
                        trajectories[(t0, h, w)] = (-1, -1, -1)
                if debug:
                    print(f"--- [no tgt_vec] keep {best_src}, drop others for target {k}")
                continue

            if debug:
                print(f"--- [target {k}] tgt_vec={tgt_vec}")

            best_src, best_score = None, -math.inf
            for (t0, h, w) in src_list:
                s_vec = best_source_vec((t0, h, w))
                if s_vec is None:
                    if debug:
                        print(f"--- [no s_vec] fallback to discrete diff")
                    s_vec = np.array([k[2] - w, k[1] - h], float)

                s, parts = score(s_vec, tgt_vec, src_pos=(h, w), tgt_pos=(k[1], k[2]))
                if debug:
                    print(f"s_vec={s_vec}, tgt_vec={tgt_vec}, score={s:.4f}, parts(dir,mag,pos)={parts}")
                if s > best_score:
                    best_score = s
                    best_src = (t0, h, w)

            for (t0, h, w) in src_list:
                if (t0, h, w) != best_src and t0 != T - 1:
                    trajectories[(t0, h, w)] = (-1, -1, -1)

            if debug:
                print(f" -> keep {best_src} for target {k}, score={best_score:.4f}")
                print("=" * 100)

        return trajectories
    # ================= end helpers ================= #

    for resolution in resolutions:
        print("=" * 30)
        print(resolution)
        print('window_sizes[resolution]', window_sizes[resolution])

        # 產兩份 flow：flow_f(浮點, 給打分) 與 flow_idx(整數, 給索引連結)
        resH, resW = resolution
        flow_f = Fu.interpolate(
            predicted_flows, size=(resH, resW), mode='bilinear', align_corners=False
        ).clone()  # (T-1, 2, resH, resW)
        # 轉成在新座標系下的像素位移
        flow_f[:, 0].mul_(resW / float(width))   # dx
        flow_f[:, 1].mul_(resH / float(height))  # dy

        flow_idx = torch.round(flow_f)           # 只給整數索引連結使用

        T = flow_idx.shape[0] + 1
        H = flow_idx.shape[2]
        W = flow_idx.shape[3]

        trajectories = {}
        is_activated = torch.zeros([T, H, W], dtype=torch.bool)

        # 用整數位移建立離散連結
        for t in range(T - 1):
            flow = flow_idx[t]  # (2, H, W)  [dx, dy]（整數）
            for h in range(H):
                for w in range(W):
                    if not is_activated[t, h, w]:
                        is_activated[t, h, w] = True
                        x = h + int(flow[1, h, w])  # dy
                        y = w + int(flow[0, h, w])  # dx
                        if 0 <= x < H and 0 <= y < W:
                            trajectories[(t, h, w)] = (t + 1, x, y)

        print(f"--- trajectories length : {len(trajectories)}")
        print(f"--- trajectories ten keys : {list(trajectories.keys())[:10]}")

        conflict_points = keys_with_same_value(trajectories)
        print(f"[debug] #targets with conflicts: {len(conflict_points)} / {len(trajectories)}")

        # 用浮點 flow_f（不 round）做後備與打分解衝突
        trajectories = complict_track_source_flow(
            trajectories, conflict_points, T,
            flow=flow_f,                 # 關鍵：用連續光流
            k_tgt=5, k_src=3,            # 先小一點比較穩；之後可調大
            agg="ema", ema_beta=0.6,
            w_dir=1.0, w_mag=0.5, w_pos=0.5,
            sigma_mag=1.0, sigma_pos=1.5,
            debug=False
        )

        # ====== 後續與你原本一致 ====== #
        active_traj = []
        all_traj = []
        for t in range(T):
            pixel_set = {(t, x // H, x % H): 0 for x in range(H * W)}
            new_active_traj = []
            for traj in active_traj:
                if traj[-1] in trajectories:
                    v = trajectories[traj[-1]]
                    new_active_traj.append(traj + [v])
                    pixel_set[v] = 1
                else:
                    all_traj.append(traj)
            active_traj = new_active_traj
            active_traj += [[pixel] for pixel in pixel_set if pixel_set[pixel] == 0]
        all_traj += active_traj

        useful_traj = [i for i in all_traj if len(i) > 1]
        for idx in range(len(useful_traj)):
            if useful_traj[idx][-1] == (-1, -1, -1):
                useful_traj[idx] = useful_traj[idx][:-1]

        print(f"--- num trajectories : {len(useful_traj)} ---")
        print(f"--- longest length  : {max(len(t) for t in useful_traj) if useful_traj else 0} ---")
        print(f"--- total points    : {sum(len(t) for t in useful_traj)}")
        for i, traj in enumerate(useful_traj[:3]):
            print(f"[traj {i}] len={len(traj)}  head={traj[:5]}")

        print("how many points in all trajectories for resolution{}?".format(resolution),
              sum([len(i) for i in useful_traj]))
        print("how many points in the video for resolution{}?".format(resolution), T * H * W)

        # validate if there are no duplicates in the trajectories
        trajs = []
        for traj in useful_traj:
            trajs = trajs + traj
        print(f"--- trajs shape : {len(trajs)} ---")
        assert len(find_duplicates(trajs)) == 0, \
            "There should not be duplicates in the useful trajectories."

        # check if non-appearing points + appearing points = all the points in the video
        all_points = set([(t, x, y) for t in range(T) for x in range(H) for y in range(W)])
        left_points = all_points - set(trajs)
        print("How many points not in the trajectories for resolution{}?".format(resolution),
              len(left_points))
        for p in list(left_points):
            useful_traj.append([p])
        print("how many points in all trajectories for resolution{} after pending?".format(resolution),
              sum([len(i) for i in useful_traj]))

        longest_length = max([len(i) for i in useful_traj])
        sequence_length = (window_sizes[resolution] * 2 + 1) ** 2 + longest_length - 1
        print(f"--- longest length : {longest_length} ---")
        print(f"--- sequence length : {sequence_length} ---")

        seqs = []
        masks = []

        # create a dictionary to facilitate checking the trajectories to which each point belongs.
        point_to_traj = {}
        for traj in useful_traj:
            for p in traj:
                point_to_traj[p] = traj

        for t in range(T):
            for x in range(H):
                for y in range(W):
                    neighbours = neighbors_index((t, x, y), window_sizes[resolution], H, W)
                    sequence = [(t, x, y)] + neighbours + \
                               [(0, 0, 0) for _ in range((window_sizes[resolution] * 2 + 1) ** 2 - 1 - len(neighbours))]
                    sequence_mask = torch.zeros(sequence_length, dtype=torch.bool)
                    sequence_mask[:len(neighbours) + 1] = True

                    traj = point_to_traj[(t, x, y)].copy()
                    traj.remove((t, x, y))
                    sequence = sequence + traj + \
                               [(0, 0, 0) for _ in range(longest_length - 1 - len(traj))]
                    sequence_mask[(window_sizes[resolution] * 2 + 1) ** 2:
                                  (window_sizes[resolution] * 2 + 1) ** 2 + len(traj)] = True

                    seqs.append(sequence)
                    masks.append(sequence_mask)

        seqs = torch.tensor(seqs)
        masks = torch.stack(masks)
        res[f"traj{resolution[0]}"] = seqs
        res[f"mask{resolution[0]}"] = masks

    return res
# def sample_trajectories_new(video_path, device,height,width):
#     from torchvision.models.optical_flow import Raft_Large_Weights
#     from torchvision.models.optical_flow import raft_large
#     print("~~~~~~~~~~~~ sample new trajectory ~~~~~~~~~~~~~~~~")
#     weights = Raft_Large_Weights.DEFAULT
#     transforms = weights.transforms()

#     frames, _, _ = torchvision.io.read_video(str(video_path), output_format="TCHW")
#     print(f"--- frames length : {len(frames)} ---")
#     clips = list(range(len(frames)))
    
#     #=============== raft-large estimate forward optical flow============#
#     model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
#     model = model.eval()
#     finished_trajectories = []

#     current_frames, next_frames = preprocess(frames[clips[:-1]], frames[clips[1:]], transforms, height,width)
#     list_of_flows = model(current_frames.to(device), next_frames.to(device))
#     print(f"--- optical flow iterate for {len(list_of_flows)} times ---")
#     predicted_flows = list_of_flows[-1]
#     print(f"predicte_flow shape is : {predicted_flows.shape}") # [14, 2, 512, 512])
#     #=============== raft-large estimate forward optical flow============#

#     predicted_flows = predicted_flows/max(height,width)

#     resolutions =[(height//8,width//8),(height//16,width//16),(height//32,width//32),(height//64,width//64)]
#     #resolutions = [64, 32, 16, 8]
#     res = {}
#     window_sizes = {(height//8,width//8): 2,
#                     (height//16,width//16): 1,
#                     (height//32,width//32): 1,
#                     (height//64,width//64): 1}
    
#     for resolution in resolutions:
#         print("="*30)
#         print(resolution)
#         print('window_sizes[resolution]',window_sizes[resolution])
#         trajectories = {}
#         height_scale_factor = resolution[0] / height
#         width_scale_factor = resolution[1] / width
#         predicted_flow_resolu = torch.round(max(resolution[0], resolution[1])*torch.nn.functional.interpolate(predicted_flows, scale_factor=(height_scale_factor, width_scale_factor)))
        
#         ############### no round for flow counting ##########################################################
#         # resH, resW = resolution
        
#         # # 先把光流 resize 到目標解析度（雙線性，保持浮點）
#         # flow_resized = Fu.interpolate(
#         #     predicted_flows, size=(resH, resW), mode='bilinear', align_corners=False
#         # ).clone()  # 形狀 (T-1, 2, resH, resW)
        
#         # # 再把 dx/dy 分別依寬/高比例縮放到新座標系
#         # scale_x = resW / float(width)   # 對應 dx
#         # scale_y = resH / float(height)  # 對應 dy
#         # flow_resized[:, 0].mul_(scale_x)  # dx
#         # flow_resized[:, 1].mul_(scale_y)  # dy
        
#         # predicted_flow_resolu = flow_resized  # ← 不做 round，保留浮點
#         # print(f"--- predicted flow resolution shape (T, xy, H, W) : {predicted_flow_resolu.shape}")
        
#         #########################################################################################
#         T = predicted_flow_resolu.shape[0]+1
#         H = predicted_flow_resolu.shape[2]
#         W = predicted_flow_resolu.shape[3]

#         is_activated = torch.zeros([T, H, W], dtype=torch.bool)

#         for t in range(T-1):
#             flow = predicted_flow_resolu[t]  # (2, H, W)  dx and dy
#             for h in range(H):
#                 for w in range(W):

#                     if not is_activated[t, h, w]:
#                         is_activated[t, h, w] = True
#                         # this point has not been traversed, start new trajectory
#                         x = h + int(flow[1, h, w])
#                         y = w + int(flow[0, h, w])
#                         if x >= 0 and x < H and y >= 0 and y < W:
#                             # trajectories.append([(t, h, w), (t+1, x, y)])
#                             trajectories[(t, h, w)]= (t+1, x, y)    # key : (t, h, w) 當前位置 ， value : (t+1, x, y) 下一幀 (ㄓㄥˋ )位置
#         print(f"--- trajectories length : {len(trajectories)}")
#         print(f"--- trajectories ten keys : {list(trajectories.keys())[:10]}")
#         conflict_points = keys_with_same_value(trajectories)
#         print(f"[debug] #targets with conflicts: {len(conflict_points)} / {len(trajectories)}")

#         # for k in conflict_points:
#         #     index_to_pop = random.randint(0, len(conflict_points[k]) - 1)
#         #     conflict_points[k].pop(index_to_pop)
#         #     for point in conflict_points[k]:
#         #         if point[0] != T-1:
#         #             trajectories[point]= (-1, -1, -1) # stupid padding with (-1, -1, -1)
#      ##############easy cosine similarity to decide source flow #############################################################
#         def _cos(a, b, eps=1e-6):
#             """
#             安全版餘弦相似度：若任一向量太小，回傳低分(-0.5)而非 -inf，避免整體決策崩潰。
#             """
#             na = float(np.linalg.norm(a))
#             nb = float(np.linalg.norm(b))
#             if na < eps or nb < eps:
#                 return -0.5
#             return float(np.dot(a, b) / (na * nb))
        
#         def _agg_vec(vecs, mode="ema", ema_beta=0.6, min_norm=1e-6):
#             """
#             聚合多步向量：先過濾極小向量，再用 mean/median/EMA 聚合。
#             vecs: list of np.array([dy, dx], float)
#             """
#             vecs = [v for v in vecs if np.linalg.norm(v) >= min_norm]
#             if len(vecs) == 0:
#                 return None
#             if mode == "mean":
#                 return np.mean(np.stack(vecs, 0), axis=0)
#             elif mode == "median":
#                 return np.median(np.stack(vecs, 0), axis=0)
#             elif mode == "ema":
#                 v = vecs[0].astype(float)
#                 beta = float(ema_beta)
#                 for i in range(1, len(vecs)):
#                     v = beta * v + (1.0 - beta) * vecs[i]
#                 return v
#             return np.mean(np.stack(vecs, 0), axis=0)
        
#         def _collect_forward_vecs_from_traj(start_key, steps, trajectories):
#             """
#             從 (t,x,y) 沿著 trajectories 往未來最多 steps 步，回傳 [dy,dx] 列表（離散座標差）。
#             """
#             vecs = []
#             cur = start_key  # (t,x,y)
#             for _ in range(steps):
#                 nxt = trajectories.get(cur, None)
#                 if not nxt or nxt == (-1, -1, -1):
#                     break
#                 dy = nxt[2] - cur[2]
#                 dx = nxt[1] - cur[1]
#                 vecs.append(np.array([dy, dx], float))
#                 cur = nxt
#             return vecs
        
#         def _build_reverse_index(trajectories):
#             """
#             建立反向索引： (t+1,x,y) -> (t,h,w) ，只收有效連結。
#             """
#             rev = {}
#             for k, v in trajectories.items():
#                 if v != (-1, -1, -1):
#                     rev[v] = k
#             return rev
        
#         def _collect_backward_vecs_from_traj(start_key, steps, reverse_index):
#             """
#             從 (t,h,w) 往過去最多 steps 步，回傳 [dy,dx] 列表（離散座標差）。
#             """
#             vecs = []
#             cur = start_key  # (t,h,w)
#             for _ in range(steps):
#                 if cur not in reverse_index:
#                     break
#                 prev = reverse_index[cur]      # (t-1, h0, w0)
#                 dy = cur[2] - prev[2]
#                 dx = cur[1] - prev[1]
#                 vecs.append(np.array([dy, dx], float))
#                 cur = prev
#             return vecs
        
#         def _flow_vec_forward(flow, t, x, y, min_norm=1e-6):
#             """
#             從浮點光流抓 forward 向量（t -> t+1），回傳 [dy,dx]；太小則回 None。
#             flow: (T-1, 2, H, W), flow[t,0]=dx, flow[t,1]=dy
#             """
#             if flow is None:
#                 return None
#             if not (0 <= t < flow.shape[0]):
#                 return None
#             dx = float(flow[t, 0, x, y])
#             dy = float(flow[t, 1, x, y])
#             if abs(dx) + abs(dy) < min_norm:
#                 return None
#             return [dy, dx]
        
#         def _flow_vec_backward(flow, t, x, y, min_norm=1e-6):
#             """
#             來源往過去的後備：用 (t-1 -> t) 的 forward flow 當 [dy,dx]。
#             """
#             return _flow_vec_forward(flow, t-1, x, y, min_norm=min_norm)
        
#         def complict_track_source_flow(
#             trajectories,         # dict: {(t,h,w) -> (t+1,x,y) 或 (-1,-1,-1)}
#             conflict_points,      # dict: {(t+1,x,y): [(t,h1,w1), (t,h2,w2), ...]}
#             T,                    # 總幀數
#             flow=None,            # (可選) 浮點光流 (T-1,2,H,W)；建議提供
#             k_tgt=3,              # 目標往未來看的步數
#             k_src=2,              # 來源往過去看的步數
#             agg="ema",            # "ema" | "mean" | "median"
#             ema_beta=0.6,         # ema 係數
#             w_dir=1.0,            # 方向一致性權重
#             w_mag=0.5,            # 幅度一致性權重
#             w_pos=0.5,            # 位置一致性權重
#             sigma_mag=1.0,        # 幅度一致的溫度
#             sigma_pos=1.0,        # 位置一致的σ（像素）
#             debug=False
#         ):
#             """
#             規則：
#               - 目標向量：優先用 (t+1→t+2→...) 多步 [dy,dx] 聚合；若無則用 flow[tk] 後備。
#               - 來源向量：優先用 (t0←t0-1←...) 多步 [dy,dx] 聚合；若無則用 flow[t0-1] 後備。
#               - 打分：w_dir*cos + w_mag*exp(-|Δmag|/σm) + w_pos*exp(-d^2/(2σp^2))
#               - 最後保留最佳來源，其餘來源標記 (-1,-1,-1)。
#             """
#             reverse_index = _build_reverse_index(trajectories)
        
#             def best_target_vec(k):
#                 tk, x, y = k
#                 vecs = _collect_forward_vecs_from_traj(k, steps=k_tgt, trajectories=trajectories)
#                 v = _agg_vec(vecs, mode=agg, ema_beta=ema_beta)
#                 if v is None:  # 用 flow 後備
#                     fv = _flow_vec_forward(flow, tk, x, y)
#                     if fv is not None:
#                         v = np.array(fv, float)
#                 return v
        
#             def best_source_vec(src):
#                 t0, h, w = src
#                 vecs = _collect_backward_vecs_from_traj(src, steps=k_src, reverse_index=reverse_index)
#                 v = _agg_vec(vecs, mode=agg, ema_beta=ema_beta)
#                 if v is None:  # 用 flow 後備
#                     fb = _flow_vec_backward(flow, t0, h, w)
#                     if fb is not None:
#                         v = np.array(fb, float)
#                 return v
        
#             def score(src_vec, tgt_vec, src_pos, tgt_pos):
#                 # 方向一致
#                 cosv = _cos(src_vec, tgt_vec)
        
#                 # 幅度一致
#                 m_src = float(np.linalg.norm(src_vec))
#                 m_tgt = float(np.linalg.norm(tgt_vec)) if tgt_vec is not None else 0.0
#                 mag_pen = math.exp(-abs(m_src - m_tgt) / float(sigma_mag))
        
#                 # 位置一致（用離散一步近似：src_pos + round(src_vec) ≈ tgt_pos）
#                 dy, dx = src_vec
#                 pred_x = src_pos[0] + int(round(dx))
#                 pred_y = src_pos[1] + int(round(dy))
#                 pos_err = math.sqrt((pred_x - tgt_pos[0])**2 + (pred_y - tgt_pos[1])**2)
#                 pos_pen = math.exp(-(pos_err**2) / (2.0 * (float(sigma_pos)**2)))
        
#                 return w_dir * cosv + w_mag * mag_pen + w_pos * pos_pen, (cosv, mag_pen, pos_pen)
        
#             for k, src_list in conflict_points.items():
#                 if debug:
#                     print(f"--- tgt_pt : {k} comes from {src_list}.")
        
#                 # 目標向量
#                 tgt_vec = best_target_vec(k)
#                 if tgt_vec is None:
#                     # 還是拿不到 → fallback：幅度最大（離散一步）
#                     best_src, best_mag = None, -math.inf
#                     for (t0, h, w) in src_list:
#                         v = np.array([k[2]-w, k[1]-h], float)  # [dy,dx]
#                         mag = float(np.linalg.norm(v))
#                         if mag > best_mag:
#                             best_mag, best_src = mag, (t0, h, w)
#                     for (t0, h, w) in src_list:
#                         if (t0, h, w) != best_src and t0 != T-1:
#                             trajectories[(t0, h, w)] = (-1, -1, -1)
#                     if debug:
#                         print(f"--- [no tgt_vec] keep {best_src}, drop others for target {k}")
#                     continue
        
#                 if debug:
#                     print(f"--- [target {k}] tgt_vec={tgt_vec}")
        
#                 # 正常打分
#                 best_src, best_score = None, -math.inf
#                 for (t0, h, w) in src_list:
#                     s_vec = best_source_vec((t0, h, w))
        
#                     if s_vec is None:
#                         # 最後手段：用離散差 (來源→目標)，但這很粗糙
#                         if debug:
#                             print(f"--- [no s_vec] fallback to discrete diff")
#                         s_vec = np.array([k[2]-w, k[1]-h], float)
        
#                     s, parts = score(s_vec, tgt_vec, src_pos=(h, w), tgt_pos=(k[1], k[2]))
#                     if debug:
#                         print(f"s_vec={s_vec}, tgt_vec={tgt_vec}, score={s:.4f}, parts(dir,mag,pos)={parts}")
#                     if s > best_score:
#                         best_score = s
#                         best_src = (t0, h, w)
        
#                 # 保留最佳來源，剪掉其他
#                 for (t0, h, w) in src_list:
#                     if (t0, h, w) != best_src and t0 != T-1:
#                         trajectories[(t0, h, w)] = (-1, -1, -1)
        
#                 if debug:
#                     print(f" -> keep {best_src} for target {k}, score={best_score:.4f}")
#                     print("=" * 100)
        
#             return trajectories
#         # def easy_track_source_flow(
#         #     trajectories,        # dict: {(t,h,w) -> (t+1,x,y) 或 (-1,-1,-1)}
#         #     conflict_points,     # dict: {(t+1,x,y): [(t,h1,w1), (t,h2,w2), ...]}
#         #     T,                   # 總幀數
#         # ):
#         #     """
#         #     用 (t+1,x,y) → (t+2,x2,y2) 的 '未來方向' 當基準，挑選來源。
#         #     若 (t+1,x,y) 沒有下一跳，且提供了 flow_{t+1}，就用該點的 (dx,dy) 當基準。
#         #     """
#         #     import numpy as np
#         #     import math
#         #     for k, src_list in conflict_points.items():
#         #         print(f"--- (t, x, y) {len(src_list)} points : {src_list}")
#         #         print(f"--- (t+1, x, y) : {k}")
#         #         # k = (t+1, x, y)
#         #         tk, x, y = k
#         #         # assert tk == t + 1, "conflict key 時間步應為 t+1"
        
#         #         # 1) 直接從 trajectories 取 (t+2,x2,y2) 算 target_vec
#         #         target_vec = None
#         #         if (tk < T-1) and (k in trajectories):
#         #             nxt = trajectories[k]  # (t+2, x2, y2) 或 (-1,-1,-1)
#         #             print(f"--- (t+2, x, y) : {nxt}")
#         #             if nxt != (-1, -1, -1) and nxt[0] == tk + 1:
#         #                 _, x2, y2 = nxt
#         #                 target_vec = np.array([y2 - y, x2 - x], dtype=float)
#         #                 print(f"--- target vector : {target_vec}")
#         #         # assert target_vec is not None, "target vec is None"
      
        
#         #         # 2) 若仍然沒有基準（最後一幀或取不到），那這個衝突就跳過或退回備選規則
#         #         if target_vec is None:
#         #             print("------ Target Vector is None ---")
#         #             # 可選：退回用幅度決勝
#         #             best_src = None
#         #             best_mag = -math.inf
#         #             for (t0, h, w) in src_list:
#         #                 v = np.array([y - w, x - h], dtype=float)
#         #                 mag = float(np.linalg.norm(v))
#         #                 if mag > best_mag:
#         #                     best_mag = mag
#         #                     best_src = (t0, h, w)
#         #             # 截斷其他
#         #             for (t0, h, w) in src_list:
#         #                 if (t0, h, w) != best_src and t0 != T-1:
#         #                     trajectories[(t0, h, w)] = (-1, -1, -1)
#         #             print("*" * 50)
#         #             continue
        
#         #         # 基於 target_vec 比 cosine
#         #         best_src = None
#         #         best_cos = -math.inf
#         #         best_mag = -math.inf
#         #         print(f"--- cauculate {len(src_list)} times cosine similarity ---")
#         #         for (t0, h, w) in src_list:
#         #             src_vec = np.array([y - w, x - h], dtype=float)  # source -> target
#         #             cosv = _cos(src_vec, target_vec)
#         #             print(f"{src_vec} and {target_vec} cosine similarity is {cosv}")
#         #             mag  = float(np.linalg.norm(src_vec))
#         #             if (cosv > best_cos) or (cosv == best_cos and mag > best_mag):
#         #                 best_cos = cosv
#         #                 best_mag = mag
#         #                 best_src = (t0, h, w)
#         #         print("*" * 50)
        
#         #         # 截斷其他來源
#         #         for (t0, h, w) in src_list:
#         #             if (t0, h, w) != best_src and t0 != T-1:
#         #                 trajectories[(t0, h, w)] = (-1, -1, -1)
        
#         #     return trajectories
                    
#         # trajectories = easy_track_source_flow(trajectories, conflict_points, T)
#         trajectories = complict_track_source_flow(
#             trajectories, conflict_points, T,
#             # flow=predicted_flow_resolu,  # 建議提供；沒有也可
#             k_tgt=15,                     # 目標往未來看 3 步
#             k_src=15,                     # 來源往過去看 2 步
#             agg="mean",                   # "ema"/"mean"/"median" 可選
#             ema_beta=0.6,
#             w_dir=1.0, w_mag=0.5, w_pos=0.5,
#             sigma_mag=1.0, sigma_pos=1.0,
#             debug=True                # 想看細節就 True
#         )
#         ########################################################################################################################
#         active_traj = []
#         all_traj = []
#         for t in range(T):
#             pixel_set = {(t, x//H, x%H):0 for x in range(H*W)}
#             new_active_traj = []
#             for traj in active_traj:
#                 if traj[-1] in trajectories:
#                     v = trajectories[traj[-1]]
#                     new_active_traj.append(traj + [v])
#                     pixel_set[v] = 1
#                 else:
#                     all_traj.append(traj)
#             active_traj = new_active_traj
#             active_traj+=[[pixel] for pixel in pixel_set if pixel_set[pixel] == 0]
#         all_traj += active_traj # [[(0,0,0), (0,1,0)...(-1, -1, -1)], [(0,2,0), (1,1,0)...]]
        
#         useful_traj = [i for i in all_traj if len(i)>1]
#         for idx in range(len(useful_traj)):
#             if useful_traj[idx][-1] == (-1, -1, -1):
#                 useful_traj[idx] = useful_traj[idx][:-1]
#         print(f"--- num trajectories : {len(useful_traj)} ---")
#         print(f"--- longest length  : {max(len(t) for t in useful_traj) if useful_traj else 0} ---")
#         print(f"--- total points    : {sum(len(t) for t in useful_traj)}")
#         for i, traj in enumerate(useful_traj[:3]):
#             print(f"[traj {i}] len={len(traj)}  head={traj[:5]}")

#         print("how many points in all trajectories for resolution{}?".format(resolution), sum([len(i) for i in useful_traj]))
#         print("how many points in the video for resolution{}?".format(resolution), T*H*W)

#         # validate if there are no duplicates in the trajectories
#         trajs = []
#         for traj in useful_traj:
#             trajs = trajs + traj
#         print(f"--- trajs shape : {len(trajs)} ---")
#         assert len(find_duplicates(trajs)) == 0, "There should not be duplicates in the useful trajectories."

#         # check if non-appearing points + appearing points = all the points in the video
#         all_points = set([(t, x, y) for t in range(T) for x in range(H) for y in range(W)])
#         left_points = all_points- set(trajs)
#         print("How many points not in the trajectories for resolution{}?".format(resolution), len(left_points))
#         for p in list(left_points):
#             useful_traj.append([p])
#         print("how many points in all trajectories for resolution{} after pending?".format(resolution), sum([len(i) for i in useful_traj]))


#         longest_length = max([len(i) for i in useful_traj])
#         sequence_length = (window_sizes[resolution]*2+1)**2 + longest_length - 1
#         print(f"--- longest length : {longest_length} ---")
#         print(f"--- sequence length : {sequence_length} ---")
#         seqs = []
#         masks = []

#         # create a dictionary to facilitate checking the trajectories to which each point belongs.
#         point_to_traj = {}
#         for traj in useful_traj:
#             for p in traj:   # p : (0, 1, 1)
#                 point_to_traj[p] = traj # value : (0, 1, 1), key : traj 

#         for t in range(T):
#             for x in range(H):
#                 for y in range(W):
#                     neighbours = neighbors_index((t,x,y), window_sizes[resolution], H, W)
#                     sequence = [(t,x,y)]+neighbours + [(0,0,0) for i in range((window_sizes[resolution]*2+1)**2-1-len(neighbours))]
#                     sequence_mask = torch.zeros(sequence_length, dtype=torch.bool)
#                     sequence_mask[:len(neighbours)+1] = True

#                     traj = point_to_traj[(t,x,y)].copy()
#                     traj.remove((t,x,y))
#                     sequence = sequence + traj + [(0,0,0) for k in range(longest_length-1-len(traj))]
#                     sequence_mask[(window_sizes[resolution]*2+1)**2: (window_sizes[resolution]*2+1)**2 + len(traj)] = True

#                     seqs.append(sequence)
#                     masks.append(sequence_mask)

#         seqs = torch.tensor(seqs)
#         masks = torch.stack(masks)
#         res["traj{}".format(resolution[0])] = seqs
#         res["mask{}".format(resolution[0])] = masks
#     return res