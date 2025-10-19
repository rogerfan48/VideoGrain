# CoTracker3 Integration for VideoGrain

- 2025-10-19 - Sheng Wei Fan

## 概述

本次修改整合了 CoTracker3 作為 VideoGrain 的 trajectory tracking 方法，替代原本的 RAFT optical flow。CoTracker3 提供更好的 occlusion handling（遮擋處理），能在物體被遮擋或離開畫面時持續追蹤。

## 問題背景

原本的 RAFT-based flow tracking 在以下情況會失敗：
- 物體被其他物體遮擋時會失去追蹤
- 物體暫時離開畫面後，可能會被錯誤配對到其他物體
- 無法預測被遮擋點的位置

這導致在複雜場景（多個移動物體、相互遮擋）中，temporal consistency 不佳。

## 解決方案

### CoTracker3 優勢

CoTracker3 採用 joint tracking 方式，具有以下優點：
1. **聯合追蹤**: 同時追蹤所有點，利用點之間的關係維持一致性
2. **遮擋處理**: 即使點被遮擋，仍能推測其位置
3. **可見性預測**: 明確知道每個點何時可見/被遮擋
4. **長期記憶**: 物體暫時離開畫面後仍能保持追蹤

### Multi-GPU 架構

為避免 GPU 記憶體衝突，採用雙 GPU 架構：
- **GPU 0**: 主要 diffusion model (Stable Diffusion + ControlNet)
- **GPU 1**: CoTracker3 trajectory tracking

## 主要修改

### 1. 新增 CoTracker 實作 (`video_diffusion/common/image_util.py`)

新增 `sample_trajectories_cotracker()` 函數（295 行新程式碼），功能包括：

**關鍵設計決策**：
- **序列長度統一**: 強制 `longest_length = T` 以匹配 RAFT 行為
  ```python
  longest_length = T  # 與 RAFT 保持一致
  sequence_length = spatial_neighbors + T - 1
  ```
  這確保 flow attention 能正確提取 temporal indices（Line 928-929）

- **CPU tensor 創建**: 在 CPU 上創建 trajectories，而非直接在 GPU
  ```python
  seqs = torch.tensor(seqs)  # CPU，與 RAFT 一樣
  masks = torch.stack(masks)
  ```
  讓 PyTorch 自動管理移動到 GPU 的時機，避免記憶體碎片化（Line 962-963）

- **GPU 隔離與清理**: CoTracker 在 GPU1 執行後完全清理記憶體
  ```python
  del cotracker
  torch.cuda.empty_cache()
  torch.cuda.synchronize()
  ```
  確保 GPU1 釋放所有資源（Line 970-987）

### 2. 動態 Device 轉換 (`video_diffusion/prompt_attention/attention_register.py`)

在 flow attention 使用 trajectories 前，自動將其移到正確的 device：

```python
# Line 462-467
if traj.device != query.device:
    traj = traj.to(query.device, non_blocking=True)
if mask.device != query.device:
    mask = mask.to(query.device, non_blocking=True)
```

**為何需要此修改**：
- RAFT trajectories 在 CPU 創建
- CoTracker trajectories 也改為在 CPU 創建（匹配 RAFT）
- PyTorch 需要所有參與運算的 tensor 在同一 device
- 使用 `non_blocking=True` 提升傳輸效率

### 3. ControlNet xformers 支援 (`test.py`)

**問題發現**: 原程式碼只為 UNet 啟用 xformers，ControlNet 的 attention 會嘗試分配 15GB 記憶體（30 frames）。

**解決方案** (Line 149-173):
```python
if is_xformers_available():
    pipeline.enable_xformers_memory_efficient_attention()  # UNet
    controlnet.set_use_memory_efficient_attention_xformers(True)  # ControlNet
else:
    # Fallback: sliced attention
    enable_sliced_attention_recursive(controlnet, slice_size=1)
```

這將 ControlNet attention 的記憶體需求從 15GB 降到 ~0.5GB。

### 4. Multi-GPU 設定 (`run.sh`)

```bash
# 原本: export CUDA_VISIBLE_DEVICES=0
# 現在: export CUDA_VISIBLE_DEVICES=0,1

# 減少記憶體碎片化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 5. Config 支援 (`config/flow/cotracker.yaml`)

新增 CoTracker 相關參數：
```yaml
editing_config:
  use_cotracker: true           # 啟用 CoTracker（false 則使用 RAFT）
  cotracker_online: false       # offline mode（更準確）
  cotracker_grid_size: null     # 自動計算 point density
  cotracker_low_memory: false   # 使用高品質模式
  cotracker_gpu_id: 1           # 在 GPU1 執行
```

## 使用方式

### 基本使用

```bash
# 確保兩張 GPU 可用
export CUDA_VISIBLE_DEVICES=0,1

# 執行
bash run.sh
```

### Config 設定

```yaml
editing_config:
  use_cotracker: true          # true: CoTracker, false: RAFT
  cotracker_gpu_id: 1         # CoTracker 使用的 GPU ID
```

### 切換回 RAFT

若遇到問題，可隨時切換回 RAFT：
```yaml
editing_config:
  use_cotracker: false
```

## 效能比較

| 方法 | Occlusion Handling | 速度 | 記憶體 | 適用場景 |
|------|-------------------|------|--------|---------|
| RAFT | ❌ 差 | ⚡ 快 | 💾 低 | 簡單場景，無遮擋 |
| CoTracker3 | ✅ 優秀 | 🐌 較慢 | 💾💾 中高 | 複雜場景，有遮擋 |

**建議**:
- 簡單場景、追求速度 → RAFT
- 複雜場景、多物體遮擋 → CoTracker3

## 技術細節

### Trajectory 格式

兩種方法的輸出格式完全一致：
```python
{
    "traj64": torch.Tensor,  # [122880, 54, 3] for 64x64, 30 frames
    "mask64": torch.Tensor,  # [122880, 54]
    "traj32": ...,
    "mask32": ...,
    ...
}
```

每個 sequence 包含：
- **Spatial part** (25): 當前點 + 24 個鄰近點
- **Temporal part** (29): 未來 29 個時間步的軌跡

### Flow Attention 運作

在 `attention_register.py` 的 flow attention 中：
```python
# Line 474-477: 提取 temporal trajectory
traj_key_sequence_inds = torch.cat([
    traj[:, :, 0, :].unsqueeze(-2),      # 當前點
    traj[:, :, -clip_length+1:, :]       # 最後 29 個點（temporal）
], dim=-2)  # 結果: [B, N, 30, 3]
```

這 30 個點用於計算每個像素的 temporal attention，追蹤其在時間維度的對應關係。

## 已知限制

1. **需要兩張 GPU**: 若只有一張 GPU，CoTracker 和 diffusion model 會競爭記憶體
2. **速度較慢**: CoTracker3 比 RAFT 慢約 2-3 倍
3. **首次執行需下載**: CoTracker3 model 約 500MB，首次使用會自動下載

## 故障排除

### OOM 錯誤
- 確認 `CUDA_VISIBLE_DEVICES=0,1` 設定正確
- 檢查 `cotracker_gpu_id: 1` 在 config 中
- 嘗試降低 frame 數或使用 `cotracker_low_memory: true`

### Device 錯誤
- 確認兩張 GPU 都可用: `nvidia-smi`
- 檢查 xformers 已安裝: `pip list | grep xformers`

### 品質問題
- CoTracker offline mode 品質最佳，但較慢
- 調整 `cotracker_grid_size` 改變 point density

## 檔案修改總結

| 檔案 | 修改內容 | 行數 |
|------|---------|------|
| `video_diffusion/common/image_util.py` | 新增 CoTracker 實作 | +295 |
| `video_diffusion/prompt_attention/attention_register.py` | Device 自動轉換 | +8 |
| `test.py` | ControlNet xformers、CoTracker 整合 | +86 |
| `run.sh` | Multi-GPU 設定 | +10 |
| `config/flow/cotracker.yaml` | 新增 config 範例 | +63 |

**總計**: ~462 行核心程式碼修改

## 結論

本次整合成功將 CoTracker3 作為 VideoGrain 的可選 trajectory tracking 方法，並通過仔細的記憶體管理和 GPU 隔離，在不增加硬體需求的情況下，實現了更好的 occlusion handling 能力。

關鍵成功因素：
1. **完全模仿 RAFT 行為**: 確保兩種方法的記憶體使用模式一致
2. **Multi-GPU 隔離**: 避免記憶體競爭
3. **ControlNet xformers**: 解決原有的記憶體瓶頸

使用者可根據場景複雜度，靈活選擇 RAFT 或 CoTracker3，無需修改其他程式碼。
