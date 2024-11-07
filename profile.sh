#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WARP_CACHE_PATH="$SCRIPT_DIR/_warp_source_cache/"  # Warp library env variable
echo "Kernel written to $WARP_CACHE_PATH"

PROFILE_OUTPUT_DIR="$SCRIPT_DIR/_profile_outputs"   # directory to save profiler ncu-rep files

# create output file save directories if needed
if [[ ! -e $PROFILE_OUTPUT_DIR ]]; then
    echo "Creating directory $PROFILE_OUTPUT_DIR"
    mkdir $PROFILE_OUTPUT_DIR
fi 

if [[ ! -e $SCRIPT_DIR ]]; then
    echo "Creating directory $SCRIPT_DIR"
    mkdir $SCRIPT_DIR
fi

ncu --set full --target-processes all \
    --metrics dram__bytes_read.sum,dram__bytes_written.sum,sm__inst_executed_pipe_tensor_op_hmma.avg,sm__cycles_elapsed.avg,l2_tex_read_bytes.sum,l2_tex_write_bytes.sum,lts__t_bytes.sum,lts__t_sectors_pipe_lsu_mem_rd.sum,lts__t_sectors_pipe_lsu_mem_wr.sum \
    --nvtx --call-stack \
    --export $PROFILE_OUTPUT_DIR/ncu_warp_run --import-source yes \
    --source-folders $WARP_CACHE_PATH \
    python examples/run.py --scene=48 --object=0 --FRAME_RATE=50