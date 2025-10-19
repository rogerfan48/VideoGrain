#!/bin/bash

# Multi-GPU Configuration:
# GPU 0: Main diffusion model (SD 1.5 + ControlNet)
# GPU 1: CoTracker3 trajectory tracking
export CUDA_VISIBLE_DEVICES=0,1

# Reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Launch with single process (GPU distribution handled internally)
accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=no \
    --dynamo_backend=no \
    test.py --config config/flow/cotracker.yaml
