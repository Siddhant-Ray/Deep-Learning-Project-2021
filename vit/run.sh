#!/bin/bash
echo "script to run srl tasks"

if ls lsf.* 1> /dev/null 2>&1
then
    echo "lsf files do exist"
    echo "removing older lsf files"
    rm lsf.*
fi

module load gcc/8.2.0 python_gpu/3.8.5 eth_proxy
source ../venv/bin/activate

args=(
    -G s_stud_infk
    -n 4
    -W 20:00
    -R "rusage[mem=4500]"
)

if [ -z "$1" ]; then echo "CPU mode selected"; fi
while [ ! -z "$1" ]; do
    case "$1" in
        gpu)
            echo "GPU mode selected"
            args+=(-R "rusage[ngpus_excl_p=4]")
	    args+=(-R "select[gpu_mtotal0>=10240]")
            ;;
        intr)
            echo "Interactive mode selected"
            args+=(-Is)
            ;;
    esac
    shift
done

bsub "${args[@]}" "python main.py config2.json" 
