#!/bin/bash
echo "Generating background dataset"

if ls lsf.* 1> /dev/null 2>&1
then
    echo "lsf files do exist"
    echo "removing older lsf files"
    rm lsf.*
fi

module load gcc/8.2.0 python_gpu/3.8.5 cuda/11.3.1 eth_proxy
source ../venv/bin/activate

args=(
    -G s_stud_infk
    -n 1
    -W 0:30
    -R "rusage[mem=4500]"
)

bsub "${args[@]}" python gen_background_cifar.py
