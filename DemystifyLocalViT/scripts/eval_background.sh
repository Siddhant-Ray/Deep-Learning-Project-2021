#!/usr/bin/env bash

NCCL_SOCKET_IFNAME=ib0


MASTER_IP=127.0.0.1
MASTER_PORT=12345
NODE_RANK=${OMPI_COMM_WORLD_RANK} && echo NODE_RANK: ${NODE_RANK}
PER_NODE_GPU=1 && echo PER_NODE_GPU: ${PER_NODE_GPU}
NUM_NODE=${OMPI_COMM_WORLD_SIZE} && echo NUM_NODE: ${NUM_NODE}

MKL_THREADING_LAYER=GNU python -m torch.distributed.launch \
    --nproc_per_node $PER_NODE_GPU \
    --nnodes=1 \
    --node_rank=$NODE_RANK \
    --master_port=$MASTER_PORT \
    main.py \
    --resume output/dynamic_dwnet_base_patch4_window7_224/ddwnet_base_patch4_window7_224/default/ckpt_epoch_80.pth  \
    --cfg ./configs/dynamic_dwnet_base_patch4_window7_224.yaml \
    --data-path "CIFAR10_data/background_cifar_eval/" \
    --output "output/dynamic_dwnet_base_patch4_window7_224" \
    --data-set largerimages \
    --batch-size 16 \
    --amp-opt-level O0 \
    --eval 

