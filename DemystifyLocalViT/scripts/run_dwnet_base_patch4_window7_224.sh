#!/usr/bin/env bash
    
MKL_THREADING_LAYER=GNU python main.py \
    --cfg ./configs/dwnet_base_patch4_window7_224.yaml \
    --data-path "CIFAR10_data" \
    --output "output/dwnet_base_patch4_window7_224" \
    --data-set CIFAR \
    --batch-size 64 \
    --amp-opt-level O0
