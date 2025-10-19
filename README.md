# VideoGrain

VideoGrain is a video editing framework using Stable Diffusion with ControlNet for spatial-temporal video editing. It supports class-level, instance-level, and part-level video manipulation through text prompts while preserving temporal consistency.

**New**: Now supports CoTracker3 for improved trajectory tracking with better occlusion handling. See [CoTracker Integration Guide](1019-cotracker.md) for details.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Trajectory Tracking Methods](#trajectory-tracking-methods)
- [Configuration](#configuration)
- [Data Preparation](#data-preparation)
- [Troubleshooting](#troubleshooting)
- [GPU Monitoring](#gpu-monitoring)

## Installation

### Prerequisites

- CUDA-capable GPU(s) with 24GB+ VRAM
- Linux system (tested on Ubuntu)
- Conda or Miniconda

### Setup Steps

```bash
# 1. Download and install Miniconda (if not already installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow prompts: Enter -> yes -> Enter (default path) -> yes (initialization)
source ~/.bashrc
rm Miniconda3-latest-Linux-x86_64.sh

# 2. Clone repository
git clone https://github.com/knightyxp/VideoGrain.git
cd VideoGrain

# 3. Create and activate environment
conda create -n videograin python==3.10
conda activate videograin

# 4. Install git-lfs for downloading large model files
conda install -c conda-forge git-lfs -y

# 5. Install PyTorch with CUDA support
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# 6. Install xformers (critical for memory efficiency)
pip install --pre -U xformers==0.0.27

# 7. Fix requirements.txt and install dependencies
# Edit requirements.txt: change 'pyav' to 'av'
vim requirements.txt  # or use sed: sed -i 's/pyav/av/g' requirements.txt
pip install -r requirements.txt

# 8. Install ONNX Runtime with GPU support
pip install onnxruntime-gpu --force-reinstall

# 9. Download pre-trained models
# Downloads SD 1.5, ControlNet depth/pose v10/v11
bash download_all.sh

# 10. (Optional) Download VideoGrain example data
gdown https://drive.google.com/uc?id=1dzdvLnXWeMFR3CE2Ew0Bs06vyFSvnGXA
tar -zxvf videograin_data.tar.gz
rm videograin_data.tar.gz
```

## Quick Start

### Single GPU Inference

```bash
export CUDA_VISIBLE_DEVICES=0
accelerate launch --num_processes=1 test.py --config config/02_walk_walk_config.yaml
```

### Multi-GPU Inference (Recommended for CoTracker)

```bash
# Use multiple GPUs: GPU 0 for main model, GPU 1 for CoTracker
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=no \
    --dynamo_backend=no \
    test.py --config config/flow/cotracker.yaml
```

## Trajectory Tracking Methods

VideoGrain supports two trajectory tracking methods:

| Method | Occlusion Handling | Speed | Memory | Best For |
|--------|-------------------|-------|--------|----------|
| **RAFT** (default) | ❌ Poor | ⚡ Fast | 💾 Lower | Simple scenes without occlusion |
| **CoTracker3** (new) | ✅ Excellent | 🐌 Slower | 💾 Higher | Complex scenes with occlusion |

### RAFT (Default)

RAFT uses optical flow for trajectory estimation. Fast but may lose tracking when objects are occluded.

**Config:**
```yaml
editing_config:
  use_cotracker: false  # or omit this line
```

### CoTracker3 (Recommended for Complex Scenes)

CoTracker3 provides superior occlusion handling by tracking points jointly and predicting visibility.

**Requirements:**
- 2 GPUs recommended (can run on 1 GPU with lower frame count)
- Auto-downloads model on first use (~500MB)

**Config:**
```yaml
editing_config:
  # Enable CoTracker
  use_cotracker: true

  # Use offline mode for best accuracy (recommended)
  cotracker_online: false

  # Auto-calculate grid size (or specify: 6, 8, 10 for different densities)
  cotracker_grid_size: null

  # Memory mode (false = high quality, true = memory saving)
  cotracker_low_memory: false

  # GPU assignment (null = same as main, 1 = use GPU 1)
  cotracker_gpu_id: 1

  # Visualization (optional): Generate trajectory visualization video
  visualize_cotracker_flow: true  # Creates logdir/sample/flow_visualization.mp4
```

**Trajectory Visualization:**

Enable trajectory visualization to debug and verify CoTracker's tracking quality:

```yaml
editing_config:
  use_cotracker: true
  visualize_cotracker_flow: true  # Enable visualization
```

**What you'll see:**
- **64 tracked points** (8×8 grid) covering the entire frame including edges
- **Color-coded by position**: Red (left) → Orange → Yellow → Green → Cyan → Blue (right)
- **Trajectory trails**: Show point movement from frame 0 to current frame
  - Solid lines (bright): Visible trajectories (high confidence)
  - Dashed lines (faded): Invisible trajectories (low confidence, e.g., occluded)
  - Line thickness increases over time (older = thin, recent = thick)
- **Current positions**: Circles with darker outline for better visibility

**Output:** `{logdir}/sample/flow_visualization.mp4`

**Use cases:**
- Verify tracking quality through occlusions
- Debug why editing fails in certain frames
- Understand which points CoTracker considers "visible" vs "invisible"
- Check if trajectory discontinuities affect your edit

**Visibility threshold:** The system uses `visibility > 0.1` to accept trajectory points. CoTracker marks points as "invisible" when occluded, blurred, or leaving the frame - these still get tracked but with lower confidence.

See [1019-cotracker.md](1019-cotracker.md) for detailed comparison and technical details.

## Configuration

### Basic Config Structure

```yaml
# Model paths
pretrained_model_path: "ckpt/stable-diffusion-v1-5"
logdir: ./result/my_experiment

# Video input
dataset_config:
  path: "data/my_video/frames"
  prompt: "A person walking"
  n_sample_frame: 16                    # Number of frames to process
  sampling_rate: 2                      # Frame sampling rate
  layout_mask_dir: "./data/my_video/masks"
  layout_mask_order: ["person", "background"]

# ControlNet settings
control_config:
  control_type: "dwpose"                # dwpose, openpose, depth, canny, etc.
  pretrained_controlnet_path: "ckpt/control_v11p_sd15_openpose"
  controlnet_conditioning_scale: 1.0

# Editing settings
editing_config:
  use_invertion_latents: true
  flatten_res: [1]                      # [1]=64x64 (part-level), [2]=16x16 (class-level)
  guidance_scale: 7.5
  num_inference_steps: 50

  # Trajectory tracking (choose one)
  use_cotracker: false                  # true for CoTracker, false for RAFT
  cotracker_gpu_id: 1                   # GPU for CoTracker (if enabled)
  visualize_cotracker_flow: false       # true to generate trajectory visualization

  # Editing prompts
  editing_prompts:
    [["Spider-man walking", "person", "Spider-man"]]
```

### Key Parameters

- **n_sample_frame**: Number of frames to process
  - Lower values (8-16): Faster, less memory
  - Higher values (20-30): Better temporal consistency

- **flatten_res**: Controls editing granularity
  - `[1]`: 64x64 resolution for part-level edits (fine details)
  - `[2]`: 16x16 resolution for class-level edits (whole objects)
  - `[1, 2]`: Multi-scale editing

- **use_cotracker**: Trajectory tracking method
  - `false` or omitted: Use RAFT (faster)
  - `true`: Use CoTracker3 (better quality)

- **visualize_cotracker_flow**: Generate trajectory visualization (only works when `use_cotracker: true`)
  - `false` or omitted: No visualization (default)
  - `true`: Generate `{logdir}/sample/flow_visualization.mp4`
  - Shows 64 tracked points (8×8 grid) with color-coded trajectories
  - Useful for debugging tracking issues and understanding visibility

## Data Preparation

### Step 1: Extract Frames from Video

```bash
export DISPLAY=""
python image_util/sample_video2frames.py \
    --video_path "./data/my_video/input.mp4" \
    --output_dir "./data/my_video/frames"
```

**Common Issues:**

If you encounter OpenCV import errors:
```bash
conda install -c conda-forge libgl -y
conda install -c conda-forge mesa-libgl-cos7-x86_64 mesa-dri-drivers-cos7-x86_64 -y
```

### Step 2: Create Masks with SAM2

#### Install SAM2

```bash
# Create separate environment for SAM2
conda create -n SAM2 python=3.10 -y
conda activate SAM2

# Clone and install SAM2
cd ~ && git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python matplotlib jupyter notebook

# Download SAM2 checkpoints
cd checkpoints
bash download_ckpts.sh
```

#### Create Masks

```bash
# Run SAM2 to create masks
python create_four_masks_sam2.py

# Convert masks to PNG format
python convert_masks_to_png.py
```

### Step 3: Run Inference

```bash
# Switch back to videograin environment
conda activate videograin

# Single GPU
export CUDA_VISIBLE_DEVICES=0
accelerate launch test.py --config config/my_config.yaml

# Multi-GPU (for CoTracker)
export CUDA_VISIBLE_DEVICES=0,1
bash run.sh
```

## Troubleshooting

### Common Errors

#### 1. Missing libGL.so.1

```bash
sudo apt-get update
sudo apt-get install -y libgl1
```

#### 2. CUDA Provider Not Available

```bash
conda install cudatoolkit=11.8 -c nvidia -y
conda install cudnn=8.* -c conda-forge -y
```

#### 3. Out of Memory (OOM)

**Solutions:**
1. Reduce frame count: `n_sample_frame: 16` → `n_sample_frame: 8`
2. Use multiple GPUs: `export CUDA_VISIBLE_DEVICES=0,1`
3. Enable CoTracker low memory mode: `cotracker_low_memory: true`
4. Verify xformers is installed: `pip list | grep xformers`

#### 4. CoTracker Device Error

Ensure both GPUs are visible:
```bash
nvidia-smi  # Check available GPUs
export CUDA_VISIBLE_DEVICES=0,1
```

#### 5. Slow Performance

- Ensure xformers is installed: `pip install xformers==0.0.27`
- Use RAFT instead of CoTracker for faster (but lower quality) tracking
- Consider using `cotracker_online: true` for lower memory/faster speed

### Memory Usage Guide

| Frame Count | RAFT Memory | CoTracker Memory | Recommendation |
|-------------|-------------|------------------|----------------|
| 8 frames    | ~12 GB      | ~15 GB          | Single GPU OK |
| 16 frames   | ~16 GB      | ~20 GB          | Single GPU OK |
| 24 frames   | ~20 GB      | ~26 GB          | Multi-GPU recommended |
| 30 frames   | ~24 GB      | ~30 GB          | Multi-GPU required |

## GPU Monitoring

```bash
# Real-time GPU monitoring
nvidia-smi -l 1  # Update every second

# Or use these tools
sudo apt install nvtop  # Interactive GPU monitor
pip install nvitop      # Python-based GPU monitor
```