# VideoGrain

```bash
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
## Enter -> yes -> Enter (default installation path) -> yes (default initialization)
source ~/.bashrc
rm Miniconda3-latest-Linux-x86_64.sh

# Install VideoGrain
git clone https://github.com/knightyxp/VideoGrain.git
cd VideoGrain

# Create environment
conda create -n videograin python==3.10
conda activate videograin
## install git-lfs due to no privilege to use sudo, to download large files in `download_all.sh`
conda install -c conda-forge git-lfs -y

# Install dependencies
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install --pre -U xformers==0.0.27

vim requirements.txt
## Alter `pyav` -> `av`, due to name change
pip install -r requirements.txt
pip install onnxruntime-gpu --force-reinstall

# Download base models
#   download sd 1.5, controlnet depth/pose v10/v11
bash download_all.sh

# Download VideoGrain Data
gdown https://drive.google.com/uc?id=1dzdvLnXWeMFR3CE2Ew0Bs06vyFSvnGXA
tar -zxvf videograin_data.tar.gz
rm videograin_data.tar.gz
```

## Inference

- takeaway:
    - if encounter `ImportError: libGL.so.1: cannot open shared object file: No such file or directory`
        ```bash
        sudo apt-get update
        sudo apt-get install -y libgl1
        ```
    - if encounter `UserWarning: Specified provider 'CUDAExecutionProvider' is not in available provider names. Available providers: 'AzureExecutionProvider, CPUExecutionProvider'`
        ```bash
        conda install cudatoolkit=11.8 -c nvidia -y
        conda install cudnn=8.* -c conda-forge -y
        ```

```bash
# based on how many GPUs you have, if having 2, use `0,1`; if having 4, use `0,1,2,3`
export CUDA_VISIBLE_DEVICES=0,1
accelerate launch test.py --config config/part_level/adding_new_object/run_two_man/spider_polar_sunglass.yaml
```

## Data Preparation

### Step 1: prepare video

```bash
export DISPLAY="" && python image_util/sample_video2frames.py --video_path "./data/person_across/person_across.mp4" --output_dir "./data/person_across/person_across"
export DISPLAY="" && python image_util/sample_video2frames.py --video_path "./data/01_stand_walk/01_stand_walk.mp4" --output_dir "./data/01_stand_walk/01_stand_walk"
```

- If having:
    ```bash
    Traceback (most recent call last):
      File "/work/rogerfan48/VideoGrain/image_util/sample_video2frames.py", line 1, in <module>
        import cv2
      File "/home/rogerfan48/.local/lib/python3.10/site-packages/cv2/__init__.py", line 181, in <module>
        bootstrap()
      File "/home/rogerfan48/.local/lib/python3.10/site-packages/cv2/__init__.py", line 153, in bootstrap
        native_module = importlib.import_module("cv2")
      File "/home/rogerfan48/miniconda3/envs/videograin/lib/python3.10/importlib/__init__.py", line 126, in import_module
        return _bootstrap._gcd_import(name[level:], package, level)
    ```
    - then run:
        ```bash
        conda install -c conda-forge libgl -y
        conda install -c conda-forge mesa-libgl-cos7-x86_64 mesa-dri-drivers-cos7-x86_64 -y
        ```
- additional: reduce frame rate
    ```bash
    chmod +x reduce_frames.sh
    ./reduce_frames.sh`

### Step 2: create masks

- install SAM2 dependency
    ```bash
    conda create -n SAM2 python=3.10 -y
    conda activate SAM2
    conda install -c conda-forge ffmpeg
    cd /work/rogerfan48 && git clone https://github.com/facebookresearch/segment-anything-2.git
    cd /work/rogerfan48/segment-anything-2
    pip install -e .
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install opencv-python matplotlib jupyter notebook
    ```
- download SAM2 model
    ```bash
    cd /work/rogerfan48/segment-anything-2/checkpoints
    bash download_ckpts.sh
    ```
- execute SAM2 to create masks
    ```bash
    python create_four_masks_sam2.py
    ```
- transform image from jpg to png
    ```bash
    python convert_masks_to_png.py
    ```

### Step 3: execute

```bash
export CUDA_VISIBLE_DEVICES=0
accelerate launch test.py --config config/person_across_config.yaml
```

## Additional

```bash
# GPU monitoring tools
sudo apt install nvtop
pip install nvitop
```

## Causal attention modified
```python
hidden_states = _memory_efficient_attention_xformers(query, key, value, attention_mask, time_causal = True)
```
- if time_causal =True : open previous frame attention mask
- if time_causal =False : close previous frame attention mask
