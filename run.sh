#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=no \
    --dynamo_backend=no \
    test.py --config config/cross_attn/02_walk_walk_2_config.yaml
