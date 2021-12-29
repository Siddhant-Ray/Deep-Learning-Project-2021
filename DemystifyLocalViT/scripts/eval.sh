#!/usr/bin/env bash

MKL_THREADING_LAYER=GNU python main.py \
    --resume output/dynamic_dwnet_base_patch4_window7_224/ddwnet_base_patch4_window7_224/default/ckpt_epoch_80.pth  \
    --cfg ./configs/dynamic_dwnet_base_patch4_window7_224.yaml \
    --data-path "CIFAR10_data/combined_cifar_eval/" \
    --output "output/dynamic_dwnet_base_patch4_window7_224" \
    --data-set largerimages \
    --batch-size 16 \
    --amp-opt-level O0 \
    --eval 

