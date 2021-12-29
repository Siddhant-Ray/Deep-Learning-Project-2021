#!/usr/bin/env bash
    
MKL_THREADING_LAYER=GNU python main.py \
    --cfg ./configs/dwnet_tiny_patch4_window7_224.yaml \
    --output "output/dwnet_tiny_patch4_window7_224" \
    --data-path "CIFAR10_data" \
    --data-set CIFAR \
    --batch-size 128 \
    --amp-opt-level O1
