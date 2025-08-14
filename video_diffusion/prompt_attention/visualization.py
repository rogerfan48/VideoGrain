from typing import List
import os
import datetime
import numpy as np
from PIL import Image

import torch

import video_diffusion.prompt_attention.ptp_utils as ptp_utils
from video_diffusion.common.image_util import save_gif_mp4_folder_type
from video_diffusion.prompt_attention.attention_store import AttentionStore
import cv2
from IPython.display import display
from typing import List, Tuple, Union

def aggregate_attention(prompts, attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            #print('item',item.shape)
            if item.dim() == 3:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                    out.append(cross_maps)
            elif item.dim() == 4:
                t, h, res_sq, token = item.shape
                if item.shape[2] == num_pixels:
                    cross_maps = item.reshape(len(prompts), t, -1, res, res, item.shape[-1])[select]
                    out.append(cross_maps)
                    
    out = torch.cat(out, dim=-4)
    out = out.sum(-4) / out.shape[-4]
    return out.cpu()


def show_cross_attention(tokenizer, prompts, attention_store: AttentionStore, 
                         res: int, from_where: List[str], select: int = 0, save_path = None):
    """
        attention_store (AttentionStore): 
            ["down", "mid", "up"] X ["self", "cross"]
            4,         1,    6
            head*res*text_token_len = 8*res*77
            res=1024 -> 64 -> 1024
        res (int): res
        from_where (List[str]): "up", "down'
    """
    if isinstance(prompts, str):
        prompts = [prompts,]
    tokens = tokenizer.encode(prompts[select]) 
    decoder = tokenizer.decode
    
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select)
    os.makedirs('trash', exist_ok=True)
    attention_list = []
    if attention_maps.dim()==3: attention_maps=attention_maps[None, ...]
    for j in range(attention_maps.shape[0]):
        images = []
        for i in range(len(tokens)):
            image = attention_maps[j, :, :, i]
            image = 255 * image / image.max()
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.numpy().astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((256, 256)))
            image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
            images.append(image)
        ptp_utils.view_images(np.stack(images, axis=0), save_path=save_path)
        atten_j = np.concatenate(images, axis=1)
        attention_list.append(atten_j)
    if save_path is not None:
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        video_save_path = f'{save_path}/{now}.gif'
        save_gif_mp4_folder_type(attention_list, video_save_path)
    return attention_list

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
    image = image.resize((relevnace_res ** 2, relevnace_res ** 2))
    image = np.array(image)

    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1], image_relevance.shape[-1])
    image_relevance = image_relevance.cuda() # because float16 precision interpolation is not supported on cpu
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevnace_res ** 2, mode='bilinear')
    image_relevance = image_relevance.cpu() # send it back to cpu
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image_relevance = image_relevance.reshape(relevnace_res ** 2, relevnace_res ** 2)
    image = (image - image.min()) / (image.max() - image.min()+1e-8)
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis



def show_cross_attention_plus_org_img(tokenizer, prompts,org_images, attention_store: AttentionStore, 
                         res: int, from_where: List[str], select: int = 0, save_path = None, attention_maps=None):
    """
        attention_store (AttentionStore): 
            ["down", "mid", "up"] X ["self", "cross"]
            4,         1,    6
            head*res*text_token_len = 8*res*77
            res=1024 -> 64 -> 1024
        res (int): res
        from_where (List[str]): "up", "down'
        image: f c h w
    """

    
    if isinstance(prompts, str):
        prompts = [prompts,]
    tokens = tokenizer.encode(prompts[select]) 
    decoder = tokenizer.decode
    if attention_maps is None:
        print('res',res)
        attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select)
    else:
        attention_maps = attention_maps
    os.makedirs('trash', exist_ok=True)
    attention_list = []
    if attention_maps.dim()==3: attention_maps=attention_maps[None, ...]
    for j in range(attention_maps.shape[0]):
        images = []
        for i in range(len(tokens)):
            image = attention_maps[j, :, :, i]
            orig_image = org_images[j]
            image = show_image_relevance(image, orig_image)

            # image = 255 * image / image.max()
            # image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((256, 256)))
            image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
            images.append(image)
        frame_save_path = os.path.join(save_path,f'frame_{j}_cross_attn.jpg')
        ptp_utils.view_images(np.stack(images, axis=0), save_path=frame_save_path)
        atten_j = np.concatenate(images, axis=1)
        attention_list.append(atten_j)
    if save_path is not None:
        # now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        video_save_path = os.path.join(save_path,'cross_attn.gif')
        save_gif_mp4_folder_type(attention_list, video_save_path, save_gif=False)
    return attention_list


def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1))


def view_images(images: Union[np.ndarray, List],
                num_rows: int = 1,
                offset_ratio: float = 0.02,
                display_image: bool = True) -> Image.Image:
    """ Displays a list of images in a grid. """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    if display_image:
        display(pil_img)
    return pil_img



def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, fontScale=1, thickness=2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img
