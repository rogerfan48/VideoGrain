# Cross-Attention Visualization for Jay Branch

## Quick Start

### 1. Enable in Config
```yaml
editing_config:
    vis_cross_attn: true
    vis_frames: [4, 8, 12]  # Which frames to visualize
```

### 2. Run

- edit `run.sh` for the config path and `bash run.sh`

### 3. Results
```
{logdir}/visualization_denoise/
├── down_cross_map_0_head_0/frame_4_cross_attn.jpg
├── down_cross_map_0_head_1/frame_4_cross_attn.jpg
...
└── up_cross_map_X_head_Y/frame_12_cross_attn.jpg
```

- use `echo "./result/cross_attn/1112_3/sample/visualization_denoise/" | python vis_cross_combine.py` to combine those attn-maps.

---

## Implementation Details

### Key Components

**1. AttentionStoreVis** (`attention_store.py`)
- Tracks generation/inversion phase
- Only accumulates attention during generation
- Averages across denoising steps

**2. Visualization Function** (`attention_util_vis.py`)
- `show_all_cross_attention_maps()`: Main function
- Processes each layer, map, and head separately
- Creates heatmap overlays on original frames

**3. Pipeline Integration** (`ddim_spatial_temporal.py`)
```python
if vis_cross_attn:
    self.store_controller = AttentionStoreVis()
    self.store_controller.set_phase("generation")
```

### Critical Fixes Applied

**Fix 1: Attention Collection** (`attention_util.py`)
```python
# ST_Layout_Attn_ControlEdit.forward()
if self.additional_attention_store is not None:
    self.additional_attention_store.forward(sim, is_cross, place_in_unet, ...)

# ST_Layout_Attn_ControlEdit.between_steps()
if self.additional_attention_store is not None:
    self.additional_attention_store.between_steps()
```

**Fix 2: Parameter Forwarding**
- `test.py`: Extract `vis_frames` from config
- `validation_loop.py`: Accept and forward `vis_frames`

---

## Architecture Guarantees

### ✅ Non-Invasive Design
- **Zero impact** on Jay model logic when `vis_cross_attn=false`
- Uses existing `additional_attention_store` mechanism
- All visualization code in separate module

### ✅ Compatibility
- Works with Jay's head-wise attention control
- Works with semantic flow attention
- Works with instance-level masks

---

## Troubleshooting

### No visualization output?
Check console for:
```
[GENERATION] Accumulating attention at step 1
[DEBUG] store_controller generation_steps: 50
```
If steps = 0, the fixes weren't applied correctly.

### Using default frames instead of config?
Ensure `vis_frames` is in `editing_config` and parameter forwarding fixes are applied.

### Memory issues?
- Reduce `vis_frames` count
- Reduce `n_sample_frame`
- Temporarily disable with `vis_cross_attn: false`

---

## Technical Notes

### Attention Storage
- Filtered by resolution: `attn.shape[2] <= 32²`
- Stored on CPU: `.cpu().detach()`
- Phase-aware: Only generation phase data used

### Output Format
- Directory: `{key}_map_{map_idx}_head_{head_idx}/`
- Image: `frame_{frame_idx}_cross_attn.jpg`
- Layout: All tokens horizontally, one row per frame

### Performance
- Generation: Minimal impact (CPU copy only)
- Visualization: 1-5 minutes post-generation
- Depends on: frame count, token count, layer count

---

## Example Config
```yaml
pretrained_model_path: "./ckpt/stable-diffusion-v1-5"
logdir: ./result/my_vis_test

dataset_config:
    path: "data/video/frames"
    n_sample_frame: 16
    layout_mask_dir: "./data/video/masks"
    layout_mask_order: ['obj1', 'obj2']

editing_config:
    editing_prompts: [
        ['Your prompt', 'Obj1', 'Obj2'],
    ]
    num_inference_steps: 50
    vis_cross_attn: true
    vis_frames: [0, 5, 10, 15]
```

---

## Files Modified
```
video_diffusion/prompt_attention/
  ├── attention_store.py       (+ AttentionStoreVis class)
  ├── attention_util.py        (+ additional_attention_store calls)
  └── attention_util_vis.py    (NEW: visualization functions)

video_diffusion/pipelines/
  ├── ddim_spatial_temporal.py (+ visualization integration)
  └── validation_loop.py        (+ vis_frames forwarding)

test.py                         (+ vis_frames extraction)
```
