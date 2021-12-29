#!/usr/bin/env bash

MKL_THREADING_LAYER=GNU python main.py \
    --cfg ./configs/dynamic_dwnet_base_patch4_window7_224.yaml \
    --data-path "CIFAR10_data" \
    --output "output/dynamic_dwnet_base_patch4_window7_224" \
    --epoch 85 \
    --data-set CIFAR \
    --batch-size 128 \
    --amp-opt-level O0
