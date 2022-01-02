#!/bin/bash
echo "Running Local ViT (eval)"

if ls lsf.* 1> /dev/null 2>&1
then
    echo "lsf files do exist"
    echo "removing older lsf files"
    rm lsf.*
fi

module load gcc/8.2.0 python_gpu/3.8.5 cuda 11.3.1 eth_proxy openmpi/4.0.2
source ../venv/bin/activate

args=(
    -G s_stud_infk
    -n 1
    -W 4:00
    -R "rusage[mem=4500]"
)


if [ -z "$1" ]; then echo "CPU mode selected"; fi
while [ ! -z "$1" ]; do
    case "$1" in
        gpu)
            echo "GPU mode selected"
            args+=(-R "rusage[mem=4500, ngpus_excl_p=8]")
            ;;
        intr)
            echo "Interactive mode selected"
            args+=(-Is)
            ;;
    esac
    shift
done

bsub "${args[@]}" mpirun bash scripts/eval.sh
bsub "${args[@]}" mpirun bash scripts/eval_background.sh
