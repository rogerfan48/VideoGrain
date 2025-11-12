import os
import math
import textwrap
import imageio
import numpy as np
from typing import Sequence
import requests
import cv2
from PIL import Image, ImageDraw, ImageFont
from fractions import Fraction
import torch
from torchvision import transforms
from einops import rearrange
import torchvision
import imageio
import re
import torchvision.transforms.functional as F
import random
from scipy.ndimage import binary_dilation
import sys
from scipy.ndimage import distance_transform_edt
import os, time, math, random, csv
import cv2
import torch
import torchvision
import torch.nn.functional as Fu
import numpy as np
from typing import List, Dict

def _natural_key(s):
    # 讓 "frame2.png" < "frame10.png"
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def _load_mask_image(path, to_shape=None, threshold=0):  # threshold=0: 只要 >0 即視為前景
    im = Image.open(path)
    # 若有 alpha，就用 alpha；否則轉成 L 灰階
    if 'A' in im.getbands():  # RGBA / LA
        ch = im.split()[-1]   # 取 alpha
    else:
        ch = im.convert('L')
    if to_shape is not None:
        H, W = to_shape
        ch = ch.resize((W, H), resample=Image.NEAREST)
    arr = np.array(ch)
    mask = (arr > threshold)
    return mask

def load_instance_masks(
    root_dir,                 # 大資料夾路徑：底下有兩個子資料夾，各放該 instance 的所有幀
    instance_dirs=None,       # 例如 ["A", "B"]；不給就自動偵測兩個子資料夾
    to_shape=None,            # (H, W) 若指定就 resize (NEAREST) 成這個大小以配合軌跡座標
    valid_exts=('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'),
    threshold=0
):
    # 掃兩個子資料夾
    if instance_dirs is None:
        subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        subdirs.sort(key=_natural_key)
        if len(subdirs) < 2:
            raise ValueError(f"在 {root_dir} 底下找不到兩個子資料夾")
        instance_dirs = subdirs[:2]  # 取前兩個
    inst_paths = [os.path.join(root_dir, d) for d in instance_dirs]
    print(root_dir)
    print(inst_paths)
    masks = []
    lengths = []
    shapes = []
    filenames_store = []

    # 逐個 instance 讀取/排序/轉 bool
    for p in inst_paths:
        files = [f for f in os.listdir(p) if f.lower().endswith(valid_exts)]
        if not files:
            raise ValueError(f"{p} 裡沒有找到影像檔")
        files.sort(key=_natural_key)
        filenames_store.append(files)

        # 若未指定 to_shape，就用第一張圖的大小
        if to_shape is None:
            tmp_im = Image.open(os.path.join(p, files[0]))
            if 'A' in tmp_im.getbands():
                tmp_im = tmp_im.split()[-1]
            else:
                tmp_im = tmp_im.convert('L')
            H0, W0 = np.array(tmp_im).shape
            target_shape = (H0, W0)
        else:
            target_shape = to_shape

        stack = np.stack([
            _load_mask_image(os.path.join(p, f), to_shape=target_shape, threshold=threshold)
            for f in files
        ], axis=0).astype(bool)  # [T, H, W]

        masks.append(stack)
        lengths.append(stack.shape[0])
        shapes.append(stack.shape[1:])  # (H, W)

    # 檢查兩個 instance 的幀數要一樣
    if lengths[0] != lengths[1]:
        raise ValueError(f"兩個 instance 幀數不同：{lengths[0]} vs {lengths[1]}")

    # 檢查兩個 instance 的形狀要一樣
    if shapes[0] != shapes[1]:
        raise ValueError(f"兩個 instance 解析度不同：{shapes[0]} vs {shapes[1]}")

    maskA, maskB = masks[0], masks[1]  # [T,H,W]
    T, H, W = maskA.shape

    return maskA, maskB, {"T": T, "H": H, "W": W, "instance_dirs": instance_dirs, "filenames": filenames_store}



def _centroid_from_mask(mask_t: np.ndarray):
    """
    mask_t: [H, W] bool
    return (y, x) in float; (nan, nan) if empty
    """
    ys, xs = np.nonzero(mask_t)
    if ys.size == 0:
        return (np.nan, np.nan)
    return (ys.mean(), xs.mean())

def _farthest_from_edge(mask_t: np.ndarray):
    """
    在 mask_t 內找「離邊緣最遠的點」(最大內切圓中心)。
    盡量在骨架上挑（若有 skimage），否則退回整張距離圖的 argmax。
    return (y, x) in float; (nan, nan) if empty
    """
    if not np.any(mask_t):
        return (np.nan, np.nan)

    dist = distance_transform_edt(mask_t)

    # # 優先用骨架（若可用）
    # try:
    #     from skimage.morphology import medial_axis
    #     skel, dist_on_skel = medial_axis(mask_t, return_distance=True)
    #     if np.any(skel):
    #         y, x = np.unravel_index(np.argmax(dist_on_skel), dist_on_skel.shape)
    #         return (float(y), float(x))
    # except Exception:
    #     pass

    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return (float(y), float(x))


def _points_contamination(points: np.ndarray, past_other: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    points: [T, 2] float  (y, x) 可能含 NaN
    past_other: [T, H, W] bool  (對方在「過去」幀的累積前景)
    return: [T] bool
    """
    T = points.shape[0]
    contam = np.zeros(T, dtype=bool)
    for t in range(T):
        y, x = points[t]
        if np.isfinite(y) and np.isfinite(x):
            yi = np.clip(int(np.rint(y)), 0, H - 1)
            xi = np.clip(int(np.rint(x)), 0, W - 1)
            contam[t] = bool(past_other[t, yi, xi])
    return contam


def compute_centers_and_contamination(maskA: np.ndarray, maskB: np.ndarray):
    """
    同時計算：
      - centroid 與 farthest（最大內切圓中心）
      - 兩種點位各自的 contamination

    輸入：
      maskA, maskB: [T, H, W] bool

    邏輯：
      1) 移除「A、B 都全黑」的幀
      2) 對每個有效幀，算：
         - centroids_A/B: [T', 2] float
         - farthest_A/B:  [T', 2] float
      3) contamination 以「在當前點位取樣對方過去累積前景」判定：
         - contam_centroid_A: 以 A 的 centroid 取樣 pastB
         - contam_farthest_A: 以 A 的 farthest 取樣 pastB
         - B 同理
      4) 回傳 kept_indices 以對應原始時間軸
    """
    assert maskA.shape == maskB.shape and maskA.ndim == 3
    T, H, W = maskA.shape

    # ------- Step 1: 過濾全黑幀 -------
    valid_mask = (maskA.any(axis=(1, 2)) | maskB.any(axis=(1, 2)))
    kept_indices = np.nonzero(valid_mask)[0]
    if kept_indices.size == 0:
        return {
            "centroids_A": np.zeros((0, 2)),
            "centroids_B": np.zeros((0, 2)),
            "farthest_A": np.zeros((0, 2)),
            "farthest_B": np.zeros((0, 2)),
            "contam_centroid_A": np.zeros((0,), dtype=bool),
            "contam_centroid_B": np.zeros((0,), dtype=bool),
            "contam_farthest_A": np.zeros((0,), dtype=bool),
            "contam_farthest_B": np.zeros((0,), dtype=bool),
            "kept_indices": kept_indices,  # 空
        }

    maskA = maskA[valid_mask]
    maskB = maskB[valid_mask]
    T_eff = maskA.shape[0]

    # ------- Step 2: 計算兩種點位 -------
    centroids_A = np.empty((T_eff, 2), dtype=float)
    centroids_B = np.empty((T_eff, 2), dtype=float)
    farthest_A  = np.empty((T_eff, 2), dtype=float)
    farthest_B  = np.empty((T_eff, 2), dtype=float)

    for t in range(T_eff):
        A_t = maskA[t]; B_t = maskB[t]
        centroids_A[t] = _centroid_from_mask(A_t)
        centroids_B[t] = _centroid_from_mask(B_t)
        farthest_A[t]  = _farthest_from_edge(A_t)
        farthest_B[t]  = _farthest_from_edge(B_t)

    # ------- Step 3: 過去累積前景 (for contamination) -------
    cumA = np.maximum.accumulate(maskA.astype(np.uint8), axis=0).astype(bool)
    cumB = np.maximum.accumulate(maskB.astype(np.uint8), axis=0).astype(bool)
    pastA = np.zeros_like(maskA, dtype=bool); pastA[1:] = cumA[:-1]
    pastB = np.zeros_like(maskB, dtype=bool); pastB[1:] = cumB[:-1]

    # ------- Step 4: 各方法分別計算 contamination -------
    contam_centroid_A = _points_contamination(centroids_A, pastB, H, W)
    contam_centroid_B = _points_contamination(centroids_B, pastA, H, W)
    contam_farthest_A = _points_contamination(farthest_A,  pastB, H, W)
    contam_farthest_B = _points_contamination(farthest_B,  pastA, H, W)

    return {
        # 位置
        "centroids_A": centroids_A,
        "centroids_B": centroids_B,
        "farthest_A":  farthest_A,
        "farthest_B":  farthest_B,

        # contamination（各自以自己的點位去取樣）
        "contam_centroid_A": contam_centroid_A,
        "contam_centroid_B": contam_centroid_B,
        "contam_farthest_A": contam_farthest_A,
        "contam_farthest_B": contam_farthest_B,

        # 時間軸對應
        "kept_indices": kept_indices,
    }



def filter_clean_frames_by_nonempty(
    S_clean: List[int],
    B_clean: List[int],
    maskA: np.ndarray,   # [T,H,W] bool，類別 A 的二值遮罩（例如 S）
    maskB: np.ndarray,   # [T,H,W] bool，類別 B 的二值遮罩（例如 B）
    min_ratio: float = 0.05,
) -> Dict[str, List[int]]:
    """
    在「已判定的乾淨幀清單」中，剔除全黑（或近乎全黑）的幀。
    - min_ratio: 視為「非黑」所需的最小正像素比例（0 表示只要有 1 個像素就算非黑）。
      例如 0.001 代表至少 0.1% 的像素為 True 才算非黑。

    回傳：
      kept_S / removed_S：過濾後保留/剔除的 S 幀索引
      kept_B / removed_B：過濾後保留/剔除的 B 幀索引
    """
    assert maskA.shape == maskB.shape and maskA.ndim == 3
    T, H, W = maskA.shape
    total_px = H * W
    thr = int(np.ceil(total_px * min_ratio))

    # 每幀 True 像素數
    cntA = maskA.reshape(T, -1).sum(axis=1)
    cntB = maskB.reshape(T, -1).sum(axis=1)

    # 定義「非黑」條件
    nonemptyA = cntA >= max(1 if min_ratio == 0 else 0, thr)
    nonemptyB = cntB >= max(1 if min_ratio == 0 else 0, thr)

    # 在既有 clean 清單上做篩選（保持原本順序）
    kept_S    = [t for t in S_clean if 0 <= t < T and nonemptyA[t]]
    removed_S = [t for t in S_clean if 0 <= t < T and not nonemptyA[t]]

    kept_B    = [t for t in B_clean if 0 <= t < T and nonemptyB[t]]
    removed_B = [t for t in B_clean if 0 <= t < T and not nonemptyB[t]]

    return {
        "kept_S": kept_S,
        "removed_S": removed_S,
        "kept_B": kept_B,
        "removed_B": removed_B,
    }




# @torch.no_grad()
# def exam_cross(
#     video_path,
#     device,
#     grid_size=None,
#     use_online=False,
#     cotracker_device=None,
#     visualize_flow=False,   # 保留參數（若你後面要做視覺化）
#     mask_log=None             # 保留參數（若你後面要輸出影片）
# ):
#     """
#     只用『原影片尺寸』跑 CoTracker3，輸出原尺寸座標的軌跡與可見度，
#     並與同尺寸的 mask 一起丟進 detect_instance_crossings。
#     """


#     # 決定 CoTracker 使用的裝置
#     if cotracker_device is None:
#         cotracker_device = device
#     else:
#         print(f"Using separate GPU for CoTracker: {cotracker_device}")

#     # 讀取影片（原尺寸）
#     frames, _, _ = torchvision.io.read_video(str(video_path), output_format="TCHW")
#     # frames: [T, C, H, W]
#     T, C, H, W = frames.shape
#     print(f"Video loaded: T={T}, C={C}, H={H}, W={W}")

#     # 準備輸入：CoTracker 需要 [B, T, C, H, W]，數值 0~1 的 float
#     video = frames.unsqueeze(0).float() / 255.0  # [1, T, C, H, W]
#     video = video.to(cotracker_device)

#     # 載入 CoTracker3
#     print(f"Loading CoTracker3 model on {cotracker_device}...")
#     torch.cuda.empty_cache()
#     if use_online:
#         cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(cotracker_device)
#     else:
#         cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(cotracker_device)
#     cotracker.eval()
#     print("CoTracker model loaded.")

#     # 設定取樣點的 grid_size（若未提供就自動估一個跟解析度有關的合理密度）
#     if grid_size is None:
#         grid_size = max(H, W) // 8  # 你可以依需求調整
#         grid_size = max(4, grid_size)  # 不要太小

#     # 跑 CoTracker3（全片、原尺寸）
#     if use_online:
#         cotracker(video_chunk=video, is_first_step=True, grid_size=grid_size)
#         pred_tracks_list, pred_visibility_list = [], []
#         # 這裡依 CoTracker online 模式的 step 切窗
#         step = cotracker.step
#         for ind in range(0, video.shape[1] - step, step):
#             tracks, visibility = cotracker(video_chunk=video[:, ind : ind + step * 2])
#             pred_tracks_list.append(tracks)       # [B, t, N, 2]
#             pred_visibility_list.append(visibility)  # [B, t, N]
#         pred_tracks = torch.cat(pred_tracks_list, dim=1)         # [B, T, N, 2]
#         pred_visibility = torch.cat(pred_visibility_list, dim=1) # [B, T, N]
#     else:
#         pred_tracks, pred_visibility = cotracker(video, grid_size=grid_size)  # [B, T, N, 2], [B, T, N]

#     print(f"pred_tracks: {pred_tracks.shape}, pred_visibility: {pred_visibility.shape}")

#     # 移除 batch 維度
#     pred_tracks = pred_tracks[0]         # [T, N, 2]，單位為『像素座標』（相對於輸入影像）
#     pred_visibility = pred_visibility[0] # [T, N]

#     T_, N, _ = pred_tracks.shape
#     assert T_ == T, "Time dimension mismatch"
#     print(f"CoTracker tracked {N} points across {T} frames")
#     print(f"Visibility ratio: {pred_visibility.float().mean().item():.2%}")

#     # === 這裡開始：與『同原尺寸』的 mask 做互動 ===
#     # 假設你已有以下兩個函式（與你原本程式一致）：
#     #   load_instance_masks(mask_dir, to_shape=None) -> (maskA, maskB, meta)
#     #   detect_instance_crossings(tracks_xy, visibility, maskA, maskB, vis_thresh, frac_thresh) -> dict
#     # 其中 maskA/maskB 的形狀為 [T, H, W]，且 H,W 與影片相同（to_shape=None 保持原尺寸）
#     maskA, maskB, meta = load_instance_masks(mask_log, to_shape=None)  # 保持原尺寸

#     # 直接使用『原尺寸』座標（因為我們輸入 CoTracker 的就是原尺寸，不需要縮放回推）
#     tracks_xy_full = pred_tracks.detach().to(dtype=torch.float32, device='cpu').numpy()  # [T, N, 2]
#     vis_full = pred_visibility.detach().to(dtype=torch.float32, device='cpu').numpy()    # [T, N]



#     H, W = maskA.shape[1], maskA.shape[2]  # [T, H, W]

#     def compute_centroid_from_mask(mask_2d):
#         """回傳 (cx, cy) 浮點質心；若 mask 為空回傳 None。"""
#         m = mask_2d.astype(bool)
#         ys, xs = np.nonzero(m)
#         if xs.size == 0:
#             return None
#         return float(xs.mean()), float(ys.mean())

#     centroid_A = compute_centroid_from_mask(maskA[0])
#     centroid_B = compute_centroid_from_mask(maskB[0])
#     if centroid_A is None or centroid_B is None:
#         raise ValueError("第 0 幀的 A 或 B mask 為空，無法計算質心。")

#     # 從第 0 幀挑出距離質心最近的兩條 track
#     xy0 = tracks_xy_full[0]  # [N, 2]
#     dxA = xy0[:, 0] - centroid_A[0]; dyA = xy0[:, 1] - centroid_A[1]
#     dxB = xy0[:, 0] - centroid_B[0]; dyB = xy0[:, 1] - centroid_B[1]
#     idx_A = int(np.argmin(dxA * dxA + dyA * dyA))
#     idx_B = int(np.argmin(dxB * dxB + dyB * dyB))

#     print(f"[選點] A用質心選到的track index = {idx_A}, 起點={tracks_xy_full[0, idx_A].tolist()}")
#     print(f"[選點] B用質心選到的track index = {idx_B}, 起點={tracks_xy_full[0, idx_B].tolist()}")

#     def clamp_xy_to_bounds(x, y, W, H):
#         """把浮點 (x,y) 取最近像素並夾在邊界內，回傳整數索引 (xi, yi)。"""
#         xi = int(np.rint(x)); yi = int(np.rint(y))
#         if xi < 0: xi = 0
#         elif xi >= W: xi = W - 1
#         if yi < 0: yi = 0
#         elif yi >= H: yi = H - 1
#         return xi, yi

#     def contamination_flags_by_past_masks(track_xy, other_masks, self_visibility=None, vis_thresh=-1):
#         """
#         track_xy: [T, 2] (x, y)
#         other_masks: [T, H, W] (bool/uint8)
#         self_visibility: [T] (0..1)，若提供則不可見幀直接標為 False（不算汙染）
#         規則：第 n 幀的位置 (x_n, y_n)，回看 other_masks[0..n-1, yi, xi] 是否曾為 True。
#         回傳 contaminated: [T] bool，其中 contaminated[0] = False。
#         """
#         T = track_xy.shape[0]
#         other = other_masks.astype(bool)
#         H, W = other.shape[1], other.shape[2]
#         contaminated = np.zeros(T, dtype=bool)
#         for n in range(T):
#             if n == 0:
#                 contaminated[n] = False
#                 continue
#             x_n, y_n = track_xy[n]
#             xi, yi = clamp_xy_to_bounds(x_n, y_n, W, H)
#             contaminated[n] = np.any(other[:n, yi, xi])
#             if self_visibility is not None and self_visibility[n] < vis_thresh:
#                 contaminated[n] = False  # 你也可以換成 np.nan 代表不可見
#         return contaminated

#     # 取出兩條軌跡與可見度
#     trackA_xy = tracks_xy_full[:, idx_A, :]  # [T, 2]
#     trackB_xy = tracks_xy_full[:, idx_B, :]  # [T, 2]
#     visA = vis_full[:, idx_A] if vis_full is not None else None
#     visB = vis_full[:, idx_B] if vis_full is not None else None
#     print(trackA_xy)
#     print(trackB_xy)
#     print(visA)
#     print(visB)
#     # 做汙染檢查：A 的點回看 B 的 mask；B 的點回看 A 的 mask
#     contam_A = contamination_flags_by_past_masks(trackA_xy, maskB, self_visibility=visA)
#     contam_B = contamination_flags_by_past_masks(trackB_xy, maskA, self_visibility=visB)

#     # 整理出幀索引
#     contam_idx_A = np.flatnonzero(contam_A).tolist()
#     clean_idx_A  = np.flatnonzero(~contam_A).tolist()
#     contam_idx_B = np.flatnonzero(contam_B).tolist()
#     clean_idx_B  = np.flatnonzero(~contam_B).tolist()

#     # print(f"[A光流] 被汙染幀: {contam_idx_A}")
#     # print(f"[A光流] 未汙染幀: {clean_idx_A}")
#     # print(f"[B光流] 被汙染幀: {contam_idx_B}")
#     # print(f"[B光流] 未汙染幀: {clean_idx_B}")

#     # （可選）把結果包回傳；你也可以把它合併進 detect_instance_crossings 的流程
#     results = {
#         "A": {
#             "track_index": idx_A,
#             "centroid": tuple(centroid_A),
#             "track_xy": trackA_xy,                   # [T, 2] numpy
#             "visibility": visA,                      # [T] numpy or None
#             "contaminated_flags": contam_A,          # [T] bool
#             "contaminated_frames": contam_idx_A,     # list[int]
#             "clean_frames": clean_idx_A,             # list[int]
#         },
#         "B": {
#             "track_index": idx_B,
#             "centroid": tuple(centroid_B),
#             "track_xy": trackB_xy,
#             "visibility": visB,
#             "contaminated_flags": contam_B,
#             "contaminated_frames": contam_idx_B,
#             "clean_frames": clean_idx_B,
#         },

# }
#     return results
