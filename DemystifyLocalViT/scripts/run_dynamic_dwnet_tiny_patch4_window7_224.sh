#!/usr/bin/env bash
    
MKL_THREADING_LAYER=GNU python main.py \
    --cfg ./configs/dynamic_dwnet_tiny_patch4_window7_224.yaml \
    --data-path "/path/to/imagenet" \
    --output "output/dynamic_dwnet_tiny_patch4_window7_224" \
    --data-set IMNET \
    --batch-size 128 \
    --amp-opt-level O1