import os
import math
import textwrap

import imageio
import numpy as np
from typing import Sequence
import requests
import cv2
from PIL import Image, ImageDraw, ImageFont

import torch
from torchvision import transforms
from einops import rearrange
import torchvision
import imageio

import torchvision.transforms.functional as F
import random
from scipy.ndimage import binary_dilation
import sys

import os, time, math, random, csv
import cv2
import torch
import torchvision
import torch.nn.functional as Fu
import numpy as np



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



# ---------- 可視化光流基本工具 ----------

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _unique_stem(stem):
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{stem}_{ts}_{random.randint(1000,9999)}"

def _get_fps(video_path, default_fps=24):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 1e-3 else default_fps

def resize_flow(flow, new_h, new_w):
    """
    flow: [N, 2, H, W], 單位=像素（dx, dy）
    重新取樣到 (new_h, new_w)，同時對向量做尺度補償。
    """
    n, c, h, w = flow.shape
    flow_rs = Fu.interpolate(flow, size=(new_h, new_w), mode="bilinear", align_corners=False)
    scale_x = new_w / float(w)
    scale_y = new_h / float(h)
    flow_rs[:, 0, :, :] *= scale_x
    flow_rs[:, 1, :, :] *= scale_y
    return flow_rs

def flow_to_color(flow, clip_mag=None):
    """
    flow: [2, H, W] (dx, dy)，像素單位
    回傳 BGR uint8 影像 (H, W, 3)，使用 HSV/色輪可視化：
      Hue = 方向，Value = 依位移量調整
    clip_mag: 若給定，將 magnitude clip 到該值以避免過曝
    """
    dx = flow[0]
    dy = flow[1]
    mag = np.sqrt(dx**2 + dy**2)
    ang = np.arctan2(dy, dx)  # [-pi, pi]

    if clip_mag is not None and clip_mag > 0:
        mag = np.clip(mag, 0, clip_mag)

    # 正規化到 [0,1]
    mag_norm = mag / (mag.max() + 1e-8)

    # 角度映射到 [0,180]（OpenCV 的 HSV: H ∈ [0,180]）
    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
    hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (mag_norm * 255).astype(np.uint8)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def draw_flow_arrows(frame_bgr, flow, step=16, mag_scale=1.0, thickness=1):
    """
    在 BGR 影像上疊加箭頭表示光流。
    frame_bgr: (H,W,3) uint8
    flow: [2, H, W] (dx, dy) 以像素為單位
    step: 取樣間隔（越大箭頭越少）
    mag_scale: 箭頭長度縮放
    """
    H, W = frame_bgr.shape[:2]
    out = frame_bgr.copy()

    for y in range(step//2, H, step):
        for x in range(step//2, W, step):
            dx = float(flow[0, y, x]) * mag_scale
            dy = float(flow[1, y, x]) * mag_scale
            x2 = int(round(x + dx))
            y2 = int(round(y + dy))
            cv2.arrowedLine(out, (x, y), (x2, y2), (0, 255, 0), thickness, tipLength=0.3)
    return out

# ---------- 主流程：把 RAFT flow 視覺化並存檔 ----------

@torch.no_grad()
def visualize_raft_flow_on_video(
    video_path,
    predicted_flows,   # 來自 RAFT 的張量 [T-1, 2, Hf, Wf]，**像素單位** (若你之前有 /max(hw)，記得傳回放大的)
    device="cuda",
    out_dir="flow_viz",
    color_clip_mag=None,   # 可選：色彩圖的 magnitude clip 上限，e.g. 20
    arrow_step=16,
    arrow_scale=1.0,
    arrow_thickness=1,
    overlay_alpha=0.5      # 疊加權重：0.0 只看原圖；1.0 只看色彩流圖
):
    """
    會輸出兩種影片：
      1) flow_color.mp4        —— 純色彩可視化
      2) flow_on_video.mp4     —— 原影像 + 色彩流圖疊加 + 箭頭
    """
    _ensure_dir(out_dir)
    stem = _unique_stem("raft_flow")

    # 讀進原始影片（Tensorvision）& 另用 OpenCV 讀 FPS
    frames_TCHW = torchvision.io.read_video(str(video_path), output_format="TCHW")[0]  # [T, C, H, W]
    T, C, H, W = frames_TCHW.shape
    fps = _get_fps(video_path)

    # 若 flow 尺寸與原影片不一致，放到影格大小 & 做向量尺度補償
    flows = predicted_flows
    if flows.shape[-2:] != (H, W):
        flows = resize_flow(flows, H, W)  # [T-1, 2, H, W]

    # 轉到 CPU numpy
    flows_np = flows.detach().cpu().numpy()         # [T-1, 2, H, W]
    frames_np = (frames_TCHW.permute(0, 2, 3, 1).detach().cpu().numpy()).astype(np.uint8)  # [T, H, W, C] RGB
    # 轉 BGR（OpenCV）
    frames_bgr = [cv2.cvtColor(frames_np[t], cv2.COLOR_RGB2BGR) for t in range(T)]

    # 準備兩條輸出影片
    color_path = os.path.join(out_dir, f"{stem}_flow_color.mp4")
    overlay_path = os.path.join(out_dir, f"{stem}_flow_on_video.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw_color = cv2.VideoWriter(color_path, fourcc, fps, (W, H))
    vw_overlay = cv2.VideoWriter(overlay_path, fourcc, fps, (W, H))

    try:
        # 第 0~T-2 張用 flow(t->t+1)；最後一張沿用上一個
        for t in range(T):
            if t < T - 1:
                flow_t = flows_np[t]  # [2, H, W]
            else:
                flow_t = flows_np[-1]

            # 1) 純色彩圖
            color_bgr = flow_to_color(flow_t, clip_mag=color_clip_mag)  # (H,W,3) BGR
            vw_color.write(color_bgr)

            # 2) 疊加在原影像 + 箭頭
            base = frames_bgr[t]
            overlay = color_bgr
            mix = cv2.addWeighted(overlay, overlay_alpha, base, 1 - overlay_alpha, 0.0)
            mix = draw_flow_arrows(mix, flow_t, step=arrow_step, mag_scale=arrow_scale, thickness=arrow_thickness)
            vw_overlay.write(mix)
    finally:
        vw_color.release()
        vw_overlay.release()

    print(f"[OK] Flow color video: {color_path}")
    print(f"[OK] Flow overlay video: {overlay_path}")
    return color_path, overlay_path

def _ensure_dir(p): os.makedirs(p, exist_ok=True)
def _unique_stem(stem): return f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}_{random.randint(1000,9999)}"
def _get_fps(video_path, default_fps=24):
    cap = cv2.VideoCapture(str(video_path)); fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
    return fps if fps and fps > 1e-3 else default_fps

def _flows_to_video_size(predicted_flows, H, W):
    """predicted_flows: [T-1, 2, Hf, Wf]（像素單位）-> resize 到 (H,W) 並對向量做尺度補償"""
    if predicted_flows.shape[-2:] == (H, W):
        return predicted_flows
    flows = Fu.interpolate(predicted_flows, size=(H, W), mode="bilinear", align_corners=False)
    scale_x = W / float(predicted_flows.shape[-1])
    scale_y = H / float(predicted_flows.shape[-2])
    flows[:, 0] *= scale_x
    flows[:, 1] *= scale_y
    return flows

def _to_grid_xy(x, y, W, H, device):
    gx = (x / (W - 1)) * 2 - 1
    gy = (y / (H - 1)) * 2 - 1
    return torch.tensor([gx], device=device, dtype=torch.float32), torch.tensor([gy], device=device, dtype=torch.float32)

# ===== 主函式：4×5 網格軌跡影片 =====
@torch.no_grad()
def make_grid_traj_video(
    video_path,
    predicted_flows_pix,    # [T-1, 2, Hf, Wf]，像素單位（若先前 /max(hw)，請先 *max(hw) 還原）
    out_dir="flow_viz",
    rows=4,
    cols=5,
    margin_ratio=0.08,      # 內縮邊界比例，避免種子點太靠邊緣
    tail_length=30,         # 尾跡長度（幀數），None/<=0 表示全程
    point_radius=3,
    line_thickness=2,
    color_mode="by_row",    # "single" | "by_row" | "by_col" | "cycle"
    single_color=(0, 255, 0),
    save_csv=True           # 另存每條軌跡 (x,y) 到 CSV
):
    """
    產生一支 mp4：在原影片上疊 4×5=20 條軌跡。
    p_{t+1} = p_t + flow_t(p_t)（flow 用 bilinear 在子像素取樣）。
    color_mode:
      - "single": 全部同色
      - "by_row": 同一列同色
      - "by_col": 同一行同色
      - "cycle" : 每條循環不同色
    """
    assert rows >= 1 and cols >= 1
    _ensure_dir(out_dir)
    stem = _unique_stem("grid_trajs")

    # 讀影片
    frames_TCHW = torchvision.io.read_video(str(video_path), output_format="TCHW")[0]  # [T,C,H,W] RGB uint8
    T, C, H, W = frames_TCHW.shape
    fps = _get_fps(video_path)

    # 轉 BGR frames
    frames_np = frames_TCHW.permute(0,2,3,1).cpu().numpy().astype(np.uint8)
    frames_bgr = [cv2.cvtColor(frames_np[t], cv2.COLOR_RGB2BGR) for t in range(T)]

    # 調整 flow 尺寸到影片大小
    flows = _flows_to_video_size(predicted_flows_pix, H, W).detach().to(torch.float32)  # [T-1,2,H,W]
    device = flows.device

    # 建立 4×5 網格種子點（留一點邊界）
    mx = margin_ratio * W
    my = margin_ratio * H
    xs = np.linspace(mx, W-1-mx, cols, dtype=np.float32)
    ys = np.linspace(my, H-1-my, rows, dtype=np.float32)
    seeds = [(float(x), float(y)) for y in ys for x in xs]  # 依 row-major 排列（先列後行）

    # 顏色方案
    base_palette = [
        (0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 128), (0, 255, 0),
        (128, 255, 0), (255, 255, 0), (255, 128, 0), (255, 0, 0), (255, 0, 128),
        (255, 0, 255), (128, 0, 255), (0, 0, 128), (0, 128, 128), (0, 128, 0),
        (128, 128, 0), (128, 0, 0), (128, 0, 128), (0, 64, 128), (64, 0, 128),
    ]
    colors = []
    if color_mode == "single":
        colors = [single_color] * (rows * cols)
    elif color_mode == "by_row":
        row_colors_left  = base_palette[:rows]          # 左半的4種顏色
        row_colors_right = base_palette[rows:rows*2]    # 右半的4種顏色
        half = cols // 2
        for r in range(rows):
            colors += [row_colors_left[r]] * half + [row_colors_right[r]] * (cols - half)
    elif color_mode == "by_col":
        col_colors = base_palette[:cols]
        for r in range(rows):
            for c in range(cols):
                colors.append(col_colors[c])
    else:  # "cycle"
        for i in range(rows * cols):
            colors.append(base_palette[i % len(base_palette)])

    # 初始化每條軌跡容器
    trajs = [[seeds[i]] for i in range(rows * cols)]  # 每條 list 存 T 個 (x,y)

    # 推進所有軌跡
    for t in range(T-1):
        flow_t = flows[t:t+1]  # [1,2,H,W]
        for k in range(rows * cols):
            x_prev, y_prev = trajs[k][-1]
            if not (0 <= x_prev < W and 0 <= y_prev < H):
                # 出界就停在邊界
                x_prev = max(0, min(W-1, x_prev))
                y_prev = max(0, min(H-1, y_prev))
                trajs[k].append((x_prev, y_prev))
                continue

            gx, gy = _to_grid_xy(x_prev, y_prev, W, H, device)
            grid = torch.stack([gx, gy], dim=-1).view(1,1,1,2)  # [1,1,1,2]
            sampled = Fu.grid_sample(flow_t, grid, mode="bilinear", align_corners=True)  # [1,2,1,1]
            dx = float(sampled[0,0,0,0].item())
            dy = float(sampled[0,1,0,0].item())
            x_new = max(0, min(W-1, x_prev + dx))
            y_new = max(0, min(H-1, y_prev + dy))
            trajs[k].append((x_new, y_new))

    # 對齊長度（理論上已是 T）
    for k in range(rows * cols):
        if len(trajs[k]) < T:
            last = trajs[k][-1]
            trajs[k] += [last] * (T - len(trajs[k]))

    # 寫出影片
    out_path = os.path.join(out_dir, f"{stem}_{rows}x{cols}_grid_trajs.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    try:
        for t in range(T):
            frame = frames_bgr[t].copy()
            for k, color in enumerate(colors):
                # 畫尾跡（限制長度）
                start = 0 if (not tail_length or tail_length <= 0) else max(0, t - tail_length + 1)
                pts = np.array([[int(round(x)), int(round(y))] for (x, y) in trajs[k][start:t+1]], dtype=np.int32)
                if len(pts) >= 2:
                    cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=line_thickness)
                # 畫當前點
                cx, cy = trajs[k][t]
                cv2.circle(frame, (int(round(cx)), int(round(cy))), point_radius, color, -1)
            vw.write(frame)
    finally:
        vw.release()

    # 另存 CSV（每條一個檔）
    csv_paths = []
    if save_csv:
        csv_dir = os.path.join(out_dir, f"{stem}_csv")
        _ensure_dir(csv_dir)
        for k in range(rows * cols):
            r, c = divmod(k, cols)
            csv_path = os.path.join(csv_dir, f"traj_r{r}_c{c}.csv")
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "x", "y"])
                for t in range(T):
                    x, y = trajs[k][t]
                    w.writerow([t, f"{x:.3f}", f"{y:.3f}"])
            csv_paths.append(csv_path)

    print(f"[OK] Grid-trajectory video: {out_path}")
    if save_csv:
        print(f"[OK] CSV saved under: {os.path.dirname(csv_paths[0])}")
    return out_path, trajs, (csv_paths if save_csv else None)

@torch.no_grad()
def sample_trajectories_new(video_path, device,height,width):
    from torchvision.models.optical_flow import Raft_Large_Weights
    from torchvision.models.optical_flow import raft_large

    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()

    frames, _, _ = torchvision.io.read_video(str(video_path), output_format="TCHW")

    clips = list(range(len(frames)))
    
    #=============== raft-large estimate forward optical flow============#
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    model = model.eval()
    finished_trajectories = []

    current_frames, next_frames = preprocess(frames[clips[:-1]], frames[clips[1:]], transforms, height,width)
    list_of_flows = model(current_frames.to(device), next_frames.to(device))
    predicted_flows = list_of_flows[-1]
    #=============== raft-large estimate forward optical flow============#

    predicted_flows = predicted_flows/max(height,width)

    #####################################################
    # 還原到像素單位（若你有做 /max）
    # predicted_flows_pix = predicted_flows * max(height, width)
    
    # # （可選）放大看看效果，例如 * 1.5
    # # predicted_flows_pix = predicted_flows_pix * 1.5
    
    # # 產生兩種可視化影片
 
    # visualize_raft_flow_on_video(
    #     video_path=video_path,
    #     predicted_flows=predicted_flows_pix,   # [T-1, 2, Hf, Wf]
    #     device=device,
    #     out_dir="flow_viz_2cls",
    #     color_clip_mag=20,     # 視情況調整，避免色彩圖過曝；不想 clip 就設 None
    #     arrow_step=16,         # 箭頭網格密度（數字越大箭頭越少）
    #     arrow_scale=1.0,       # 箭頭長度放大倍率
    #     arrow_thickness=1,     # 箭頭粗細
    #     overlay_alpha=0.5      # 疊加透明度（0=看原圖，1=只看流）
    # )
    # grid_vid_path, grid_trajs, grid_csvs = make_grid_traj_video(
    #     video_path=video_path,
    #     predicted_flows_pix=predicted_flows_pix,
    #     out_dir="flow_viz_2cls_2",
    #     rows=4, cols=5,
    #     margin_ratio=0.08,   # 若點太靠邊可加大，例如 0.12
    #     tail_length=40,      # 只顯示最近 40 幀尾跡；想看全程就設 None 或 <=0
    #     line_thickness=2,
    #     color_mode="by_row", # "single" / "by_row" / "by_col" / "cycle"
    #     single_color=(0,255,0),
    #     save_csv=True
    # )
    #####################################################
    # sys.exit()
    resolutions =[(height//8,width//8),(height//16,width//16),(height//32,width//32),(height//64,width//64)]
    #resolutions = [64, 32, 16, 8]
    res = {}
    window_sizes = {(height//8,width//8): 2,
                    (height//16,width//16): 1,
                    (height//32,width//32): 1,
                    (height//64,width//64): 1}
    
    for resolution in resolutions:
        print("="*30)
        # print(resolution)
        # print('window_sizes[resolution]',window_sizes[resolution])
        trajectories = {}
        height_scale_factor = resolution[0] / height
        width_scale_factor = resolution[1] / width
        predicted_flow_resolu = torch.round(max(resolution[0], resolution[1])*torch.nn.functional.interpolate(predicted_flows, scale_factor=(height_scale_factor, width_scale_factor)))

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
        res["traj{}".format(resolution[0])] = seqs
        res["mask{}".format(resolution[0])] = masks
    return res



##########################################################
def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=5):
    """Draw a dashed line on the image."""
    import cv2
    import numpy as np

    x1, y1 = pt1
    x2, y2 = pt2

    # Calculate line length and direction
    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if dist == 0:
        return

    # Number of dashes
    num_dashes = int(dist / (dash_length * 2))
    if num_dashes == 0:
        cv2.line(img, pt1, pt2, color, thickness)
        return

    # Draw dashed line
    for i in range(num_dashes):
        start_ratio = (i * 2 * dash_length) / dist
        end_ratio = min(((i * 2 + 1) * dash_length) / dist, 1.0)

        start_x = int(x1 + (x2 - x1) * start_ratio)
        start_y = int(y1 + (y2 - y1) * start_ratio)
        end_x = int(x1 + (x2 - x1) * end_ratio)
        end_y = int(y1 + (y2 - y1) * end_ratio)

        cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)


def _visualize_cotracker_flow(video_path, tracks, visibility, scale_factor, logdir, visibility_threshold=0.5):
    """
    Visualize CoTracker trajectories on the original video.

    Args:
        video_path (str): Path to input video
        tracks (torch.Tensor): Tracked points [T, N, 2] in (x, y) format
        visibility (torch.Tensor): Visibility mask [T, N]
        scale_factor (float): Scale factor used during tracking
        logdir (str): Output directory
        visibility_threshold (float): Threshold for solid vs dashed lines (default: 0.5)
    """
    import cv2
    import numpy as np
    import os
    import torch

    print("\n" + "="*30)
    print("Generating flow visualization...")

    # Read video
    import torchvision
    frames, _, info = torchvision.io.read_video(str(video_path), output_format="TCHW")
    T, C, H, W = frames.shape

    # Convert to numpy arrays for drawing (T, H, W, C) in RGB
    frames_np = frames.permute(0, 2, 3, 1).numpy().astype(np.uint8)

    # Create an 8x8 grid of points uniformly distributed across the frame
    # Including edge points
    grid_h = 8  # rows
    grid_w = 8  # columns

    # Calculate grid positions (evenly spaced, including edges)
    # Use grid_h-1 to get positions from edge to edge
    margin_h = H // (grid_h - 1) if grid_h > 1 else 0
    margin_w = W // (grid_w - 1) if grid_w > 1 else 0

    grid_points = []  # Will store (grid_y, grid_x, point_idx) for each grid position
    grid_colors = []  # Color for each grid point

    print(f"Creating {grid_w}x{grid_h} grid of visualization points")

    # Generate grid points and assign colors
    for row in range(grid_h):
        for col in range(grid_w):
            # Calculate position in frame (evenly distributed, including edges)
            # For 8x8 grid on 512x512: positions are 0, 73, 146, 219, 292, 365, 438, 511
            grid_y = margin_h * row
            grid_x = margin_w * col

            # Find nearest tracked point at t=0
            # tracks is [T, N, 2] where 2 = (x, y)
            # tracks are already in the original video resolution (H x W)
            # because CoTracker was run on video_resized which has size (target_h, target_w) = (H*scale, W*scale) = (H, W) for original video
            initial_tracks = tracks[0].numpy()  # [N, 2] in video resolution space

            # Grid position is already in video space (0 to H-1, 0 to W-1)
            # No scaling needed since tracks are also in the same space
            track_x = grid_x
            track_y = grid_y

            # Find nearest point
            distances = np.sqrt((initial_tracks[:, 0] - track_x)**2 +
                              (initial_tracks[:, 1] - track_y)**2)
            nearest_idx = np.argmin(distances)

            grid_points.append((grid_y, grid_x, nearest_idx))

            # Generate saturated color based on position
            # Hue varies across the grid: left=red (0°), right=blue (240°)
            hue = int(240 * col / max(1, grid_w - 1))  # 0 to 240

            # Convert HSV to RGB (S=255, V=255 for saturated colors)
            import colorsys
            rgb = colorsys.hsv_to_rgb(hue / 360.0, 1.0, 1.0)
            color = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
            grid_colors.append(color)

    num_points = len(grid_points)
    print(f"Grid points mapped to {num_points} tracked trajectories")

    # Debug: Print some sample grid points and their mappings
    print(f"Sample grid points (first 3):")
    for i in range(min(3, num_points)):
        grid_y, grid_x, track_idx = grid_points[i]
        print(f"  Grid point {i}: pos=({grid_x}, {grid_y}), mapped to track_idx={track_idx}, color={grid_colors[i]}")

    # Debug: Print track info
    print(f"Tracks shape: {tracks.shape}, range: x=[{tracks[:,:,0].min():.1f}, {tracks[:,:,0].max():.1f}], y=[{tracks[:,:,1].min():.1f}, {tracks[:,:,1].max():.1f}]")
    print(f"Video resolution: {H}x{W}, scale_factor: {scale_factor}")

    # Prepare output directory
    output_dir = os.path.join(logdir, "sample")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "flow_visualization.mp4")

    # Pre-compute full trajectories for all grid points (from frame 0 to T-1)
    full_trajectories = []
    full_trajectories_with_invisible = []  # Also track invisible points for debugging

    for grid_idx in range(num_points):
        grid_y, grid_x, track_idx = grid_points[grid_idx]

        # Build complete trajectory from frame 0 to T-1
        trajectory = []
        trajectory_all = []  # Include invisible points

        for t in range(T):
            track_pos = tracks[t, track_idx].numpy()  # [2] = [x, y]
            x = int(track_pos[0])
            y = int(track_pos[1])

            # Clamp to valid range
            x = max(0, min(W-1, x))
            y = max(0, min(H-1, y))

            vis = visibility[t, track_idx].item()
            trajectory_all.append((t, x, y, vis))  # Store with visibility

            if vis > visibility_threshold:  # Only add visible points to main trajectory
                trajectory.append((t, x, y))

        full_trajectories.append(trajectory)
        full_trajectories_with_invisible.append(trajectory_all)

        # Debug: Print first few points with visibility info
        if grid_idx < 3:
            visible_count = sum(1 for (_, _, _, v) in trajectory_all if v > visibility_threshold)
            invisible_count = T - visible_count
            print(f"  Grid {grid_idx}: track_idx={track_idx}")
            print(f"    Visible frames: {visible_count}/{T}, Invisible: {invisible_count}")
            if len(trajectory) > 0:
                print(f"    First visible pos: {trajectory[0]}, Last visible pos: {trajectory[-1]}")
            # Show visibility pattern
            vis_pattern = ''.join(['█' if v > visibility_threshold else '·' for (_, _, _, v) in trajectory_all])
            print(f"    Visibility pattern: {vis_pattern}")

    print(f"\nPre-computed {len(full_trajectories)} trajectories")

    # Summary statistics
    avg_visible = sum(len(traj) for traj in full_trajectories) / len(full_trajectories)
    print(f"Average visible frames per point: {avg_visible:.1f}/{T}")

    # Draw trajectories on each frame
    output_frames = []
    show_invisible_tracks = True  # Set to True to show invisible tracks as dashed lines

    for current_t in range(T):
        frame = frames_np[current_t].copy()  # RGB format

        # Convert to BGR for OpenCV drawing
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw accumulated trajectories up to current frame
        for grid_idx in range(num_points):
            color_rgb = grid_colors[grid_idx]
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])  # RGB to BGR
            color_bgr_faded = tuple(int(c * 0.4) for c in color_bgr)  # Faded color for invisible

            trajectory = full_trajectories[grid_idx]
            trajectory_all = full_trajectories_with_invisible[grid_idx]

            # Filter trajectory points up to current frame
            points_so_far = [(x, y) for (t, x, y) in trajectory if t <= current_t]

            if show_invisible_tracks:
                # Also draw invisible points with faded color
                all_points_so_far = [(x, y, vis) for (t, x, y, vis) in trajectory_all if t <= current_t]

                if len(all_points_so_far) > 1:
                    # Draw all segments, using faded color for invisible parts
                    for i in range(len(all_points_so_far) - 1):
                        x1, y1, vis1 = all_points_so_far[i]
                        x2, y2, vis2 = all_points_so_far[i+1]

                        # Choose color based on visibility (using the same threshold as VideoGrain)
                        if vis1 > visibility_threshold and vis2 > visibility_threshold:
                            # Both visible: solid line with gradient thickness
                            alpha = (i + 1) / len(all_points_so_far)
                            thickness = max(2, int(4 * alpha))  # Thicker lines: 2-4
                            cv2.line(frame_bgr, (x1, y1), (x2, y2), color_bgr, thickness)
                        else:
                            # At least one invisible: thinner dashed line
                            draw_dashed_line(frame_bgr, (x1, y1), (x2, y2), color_bgr_faded, 1, dash_length=4)

                # Draw current point
                if len(all_points_so_far) > 0:
                    current_x, current_y, current_vis = all_points_so_far[-1]
                    if current_vis > visibility_threshold:
                        # Visible: larger filled circle with darker outline for better contrast
                        cv2.circle(frame_bgr, (current_x, current_y), 6, color_bgr, -1)  # Filled circle
                        # Add subtle darker outline (same color but darker)
                        outline_color = tuple(int(c * 0.6) for c in color_bgr)
                        cv2.circle(frame_bgr, (current_x, current_y), 7, outline_color, 1)
                    else:
                        # Invisible: smaller faded circle
                        cv2.circle(frame_bgr, (current_x, current_y), 3, color_bgr_faded, -1)

            else:
                # Original behavior: only show visible points
                if len(points_so_far) == 0:
                    continue

                # Draw trajectory trail (all points from frame 0 to current_t)
                if len(points_so_far) > 1:
                    # Draw all segments with gradient thickness
                    for i in range(len(points_so_far) - 1):
                        # Fading effect: older points are thinner
                        alpha = (i + 1) / len(points_so_far)
                        thickness = max(2, int(4 * alpha))  # Thicker lines: 2-4
                        cv2.line(frame_bgr, points_so_far[i], points_so_far[i+1], color_bgr, thickness)

                # Draw current point (the latest visible position)
                current_x, current_y = points_so_far[-1]
                cv2.circle(frame_bgr, (current_x, current_y), 6, color_bgr, -1)  # Filled circle
                # Add subtle darker outline
                outline_color = tuple(int(c * 0.6) for c in color_bgr)
                cv2.circle(frame_bgr, (current_x, current_y), 7, outline_color, 1)

        # Convert back to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        output_frames.append(frame_rgb)

    # Convert to torch tensor and save using torchvision
    output_frames_tensor = torch.from_numpy(np.stack(output_frames))  # [T, H, W, C]
    output_frames_tensor = output_frames_tensor.permute(0, 3, 1, 2)  # [T, C, H, W]

    # Save using torchvision with the same encoding as other videos
    fps = info['video_fps'] if 'video_fps' in info else 30.0
    torchvision.io.write_video(
        output_path,
        output_frames_tensor.permute(0, 2, 3, 1),  # [T, H, W, C] for write_video
        fps=fps,
        video_codec='h264',
        options={'crf': '18'}  # High quality
    )

    print(f"✅ Flow visualization saved to: {output_path}")
    print("="*30 + "\n")


@torch.no_grad()
def sample_trajectories_cotracker(video_path, device, height, width, grid_size=None, use_online=False, low_memory=False, cotracker_device=None, visualize_flow=False, logdir=None):
    """
    Sample trajectories using CoTracker3 for better occlusion handling.

    CoTracker3 advantages over RAFT:
    - Tracks points jointly, leveraging dependencies between tracks
    - Handles occlusions by inferring positions of hidden points
    - Maintains tracks even when points leave field of view
    - Provides visibility predictions for each point

    Args:
        video_path (str): Path to the video file (MP4)
        device (torch.device): CUDA device for computation
        height (int): Video frame height
        width (int): Video frame width
        grid_size (int, optional): Grid size for point sampling. If None, auto-calculated based on resolution.
        use_online (bool): If True, use online mode (memory efficient). If False, use offline mode (better accuracy).
        low_memory (bool): If True, use aggressive memory optimization (lower resolution tracking).
        cotracker_device (torch.device, optional): Separate GPU device for CoTracker. If None, use same as main device.
        visualize_flow (bool): If True, generate a visualization video showing tracked trajectories.
        logdir (str, optional): Output directory for visualization. Required if visualize_flow is True.

    Returns:
        dict: Dictionary containing trajectory and mask tensors for multiple resolutions
            Format matches sample_trajectories_new output:
            {
                "traj64": torch.Tensor [T*H*W, sequence_length, 3],
                "mask64": torch.Tensor [T*H*W, sequence_length],
                "traj32": ...,
                "mask32": ...,
                etc.
            }
    """
    import torchvision

    # Determine which device to use for CoTracker
    if cotracker_device is None:
        cotracker_device = device
    else:
        print(f"Using separate GPU for CoTracker: {cotracker_device}")

    # Read video frames
    frames, _, _ = torchvision.io.read_video(str(video_path), output_format="TCHW")
    T = frames.shape[0]

    # Prepare video tensor: [1, T, C, H, W] (batch dimension required by CoTracker)
    # Load to CoTracker's device (might be different from main device)
    video = frames.unsqueeze(0).float()

    # Define resolutions to process (matching original function)
    resolutions = [
        (height//8, width//8),    # 64x64 for 512x512 input
        (height//16, width//16),  # 32x32
        (height//32, width//32),  # 16x16
        (height//64, width//64)   # 8x8
    ]

    window_sizes = {
        (height//8, width//8): 2,
        (height//16, width//16): 1,
        (height//32, width//32): 1,
        (height//64, width//64): 1
    }

    res = {}

    # Load CoTracker model (use CoTracker3 for best occlusion handling)
    print(f"Loading CoTracker3 model on {cotracker_device}...")

    # Clear cache before loading to maximize available memory
    torch.cuda.empty_cache()

    if use_online:
        cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(cotracker_device)
        cotracker.eval()
    else:
        cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(cotracker_device)
        cotracker.eval()

    print(f"CoTracker model loaded successfully on {cotracker_device}")

    # Store tracks for visualization (only need highest resolution)
    visualization_tracks = None
    visualization_visibility = None

    for resolution in resolutions:
        print("="*30)
        print(f"Processing resolution: {resolution}")

        H, W = resolution

        # Calculate grid_size if not provided
        # Grid size determines how densely we sample points
        if grid_size is None:
            # Auto-calculate to get good point density
            # With separate GPU, we can afford denser tracking
            auto_grid_size = max(H, W) // 8
        else:
            auto_grid_size = grid_size

        # Adaptive resolution scaling
        # With dedicated GPU, we can use high quality tracking
        if low_memory:
            # Memory saving mode
            if H >= 64:
                scale_factor = 4  # 64x64 -> 256x256
            elif H >= 32:
                scale_factor = 4  # 32x32 -> 128x128
            elif H >= 16:
                scale_factor = 4  # 16x16 -> 64x64
            else:
                scale_factor = 4  # 8x8 -> 32x32
        else:
            # High quality mode (when using separate GPU, we can afford this)
            if H >= 64:
                scale_factor = 8  # 64x64 -> 512x512 (full resolution)
            elif H >= 32:
                scale_factor = 8  # 32x32 -> 256x256
            elif H >= 16:
                scale_factor = 8  # 16x16 -> 128x128
            else:
                scale_factor = 8  # 8x8 -> 64x64

        # Calculate target size (must be integers)
        target_h = int(H * scale_factor)
        target_w = int(W * scale_factor)

        # Resize video to current resolution for tracking
        # Move to CoTracker's device
        video_resized = torch.nn.functional.interpolate(
            video.reshape(T, 3, height, width),
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        ).unsqueeze(0).to(cotracker_device)  # [1, T, C, H*scale, W*scale] on CoTracker device

        # Run CoTracker to get trajectories
        if use_online:
            # Online mode: process in sliding windows
            cotracker(video_chunk=video_resized, is_first_step=True, grid_size=auto_grid_size)
            pred_tracks_list = []
            pred_visibility_list = []

            for ind in range(0, video_resized.shape[1] - cotracker.step, cotracker.step):
                tracks, visibility = cotracker(
                    video_chunk=video_resized[:, ind : ind + cotracker.step * 2]
                )
                pred_tracks_list.append(tracks)
                pred_visibility_list.append(visibility)

            # Concatenate results
            pred_tracks = torch.cat(pred_tracks_list, dim=1)  # [1, T, N, 2]
            pred_visibility = torch.cat(pred_visibility_list, dim=1)  # [1, T, N]
        else:
            # Offline mode: process entire video at once (better for occlusion handling)
            pred_tracks, pred_visibility = cotracker(video_resized, grid_size=auto_grid_size)
            # pred_tracks: [1, T, N, 2] where N is number of tracked points
            # pred_visibility: [1, T, N] (1 if visible, 0 if occluded)

        # Remove batch dimension carefully to handle N=1 case
        pred_tracks = pred_tracks[0]  # [T, N, 2] - use indexing instead of squeeze
        # CoTracker3 returns visibility as [B, T, N] not [B, T, N, 1]
        pred_visibility = pred_visibility[0]  # [T, N] - explicit indexing

        N = pred_tracks.shape[1]  # Number of tracked points

        print(f"CoTracker tracked {N} points across {T} frames")
        print(f"Visibility ratio: {pred_visibility.float().mean().item():.2%}")

        # Store tracks for visualization (only from highest resolution)
        if visualize_flow and visualization_tracks is None and resolution == resolutions[0]:
            visualization_tracks = pred_tracks.cpu()  # [T, N, 2]
            visualization_visibility = pred_visibility.cpu()  # [T, N]
            visualization_scale = scale_factor  # Remember the scale factor for drawing

        # Convert CoTracker tracks to trajectory format
        # pred_tracks are in (x, y) format, need to convert to grid coordinates
        # Scale tracks from (H*scale_factor, W*scale_factor) space to (H, W) grid
        tracks_scaled = pred_tracks / float(scale_factor)
        tracks_scaled = torch.round(tracks_scaled).long()

        # Build point-to-trajectory mapping
        # Each tracked point forms a trajectory across time
        point_trajectories = {}

        # Visibility threshold for accepting trajectory points
        # Lower threshold = more permissive (accept more "uncertain" points)
        # Higher threshold = more strict (only accept confident points)
        visibility_threshold = -1  # Changed from 0.5 to be more permissive

        for point_idx in range(N):
            trajectory = []
            for t in range(T):
                if pred_visibility[t, point_idx] > visibility_threshold:  # Point is visible enough
                    y, x = tracks_scaled[t, point_idx]  # CoTracker outputs (x, y)
                    x, y = y.item(), x.item()  # Swap to (y, x) and convert to int

                    # Clamp to valid range
                    x = max(0, min(H-1, x))
                    y = max(0, min(W-1, y))

                    trajectory.append((t, x, y))

            if len(trajectory) > 0:
                # Store trajectory indexed by first point
                first_point = trajectory[0]
                point_trajectories[first_point] = trajectory

        print(f"Created {len(point_trajectories)} valid trajectories")

        # Build sequences for each spatial-temporal location
        # This matches the output format of sample_trajectories_new
        all_points = set([(t, x, y) for t in range(T) for x in range(H) for y in range(W)])
        tracked_points = set()
        for traj in point_trajectories.values():
            tracked_points.update(traj)

        # For points not tracked, create singleton trajectories
        untracked_points = all_points - tracked_points
        for point in untracked_points:
            point_trajectories[point] = [point]

        print(f"Total points covered: {len(all_points)}")
        print(f"Tracked points: {len(tracked_points)}")
        print(f"Untracked points: {len(untracked_points)}")

        # Create point-to-trajectory lookup
        point_to_traj = {}
        for traj in point_trajectories.values():
            for p in traj:
                point_to_traj[p] = traj

        # Calculate sequence length
        # IMPORTANT: Match RAFT's behavior - longest_length should be T (clip_length)
        # RAFT's longest trajectory is T frames, so sequence_length = spatial + T - 1
        # This ensures flow attention gets correct temporal indices
        longest_length = T  # Force to match RAFT
        sequence_length = (window_sizes[resolution]*2+1)**2 + longest_length - 1

        seqs = []
        masks = []

        # Build sequences for each point (matching original format)
        for t in range(T):
            for x in range(H):
                for y in range(W):
                    # Get spatial neighbors
                    neighbours = neighbors_index((t,x,y), window_sizes[resolution], H, W)

                    # Build spatial part of sequence
                    sequence = [(t,x,y)] + neighbours + [(0,0,0) for i in range((window_sizes[resolution]*2+1)**2-1-len(neighbours))]
                    sequence_mask = torch.zeros(sequence_length, dtype=torch.bool)
                    sequence_mask[:len(neighbours)+1] = True

                    # Build temporal part of sequence (trajectory)
                    if (t,x,y) in point_to_traj:
                        traj = point_to_traj[(t,x,y)].copy()
                        traj.remove((t,x,y))
                    else:
                        traj = []

                    # Add trajectory to sequence
                    sequence = sequence + traj + [(0,0,0) for k in range(longest_length-1-len(traj))]
                    sequence_mask[(window_sizes[resolution]*2+1)**2: (window_sizes[resolution]*2+1)**2 + len(traj)] = True

                    seqs.append(sequence)
                    masks.append(sequence_mask)

        # Convert to tensors on CPU first, matching RAFT behavior
        # PyTorch will automatically move to GPU when needed, which may have better memory management
        seqs = torch.tensor(seqs)
        masks = torch.stack(masks)

        # Store in result dict (already on main device)
        res["traj{}".format(resolution[0])] = seqs
        res["mask{}".format(resolution[0])] = masks

        print(f"Resolution {resolution} completed: {seqs.shape[0]} sequences of length {sequence_length}")

        # Aggressive memory cleanup after each resolution
        del pred_tracks, pred_visibility, tracks_scaled, point_trajectories
        del seqs, masks, video_resized, point_to_traj

        # Clear GPU cache on CoTracker's device
        if torch.cuda.is_available():
            with torch.cuda.device(cotracker_device):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    # CRITICAL: Explicitly delete CoTracker model and free all GPU memory
    print("\n" + "="*30)
    print("Cleaning up CoTracker model to free GPU memory...")
    del cotracker
    del video
    torch.cuda.empty_cache()

    # Force garbage collection
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Report memory freed
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(cotracker_device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(cotracker_device) / 1024**3
        print(f"GPU memory after cleanup: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
    print("="*30 + "\n")

    # Generate visualization if requested
    if visualize_flow and visualization_tracks is not None and logdir is not None:
        _visualize_cotracker_flow(
            video_path,
            visualization_tracks,
            visualization_visibility,
            visualization_scale,
            logdir,
            visibility_threshold  # Pass the threshold used by VideoGrain
        )

    return res