#!/bin/bash
echo "script to run srl tasks"

if ls lsf.* 1> /dev/null 2>&1
then
    echo "lsf files do exist"
    echo "removing older lsf files"
    rm lsf.*
fi

module load gcc/8.2.0 python_gpu/3.8.5 eth_proxy openmpi/4.0.2
source ../venv/bin/activate

args=(
    -G s_stud_infk
    -n 1
    -W 120:00
    -R "rusage[mem=4500, ngpus_excl_p=8]"
)


if [ -z "$1" ]; then echo "CPU mode selected"; fi
while [ ! -z "$1" ]; do
    case "$1" in
        gpu)
            echo "GPU mode selected"
            args+=(-R "rusage[ngpus_excl_p=4]")
            ;;
        intr)
            echo "Interactive mode selected"
            args+=(-Is)
            ;;
    esac
    shift
done
echo "here"

#bsub "${args[@]}" mpirun bash scripts/run_dynamic_dwnet_base_patch4_window7_224.sh

#bsub "${args[@]}" mpirun bash scripts/eval.sh
bsub "${args[@]}" mpirun bash scripts/eval_background.sh





