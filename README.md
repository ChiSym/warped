# Examples
We can run and profile all the [examples](https://github.com/NVIDIA/warp/tree/main/warp/examples) in Warp. Let's go through the raycast bunny example!

## Raycast Bunny

<img src="https://github.com/user-attachments/assets/6ed61f6f-e07a-4e2f-b794-cad886594d4d" width="50%">

Profile the [raycast bunny](https://github.com/NVIDIA/warp/tree/main/examples/core/example_raycast):
```shell
sudo ~/.pixi/bin/pixi run ncu \
  --set full \
  --target-processes all \
  --metrics dram__bytes_read.sum,dram__bytes_written.sum,sm__inst_executed_pipe_tensor_op_hmma.avg,sm__cycles_elapsed.avg,l2_tex_read_bytes.sum,l2_tex_write_bytes.sum,lts__t_bytes.sum,lts__t_sectors_pipe_lsu_mem_rd.sum,lts__t_sectors_pipe_lsu_mem_wr.sum  \
  --nvtx \
  --call-stack \
  --export example_raycast_bunny \
  python -m warp.examples.core.example_raycast
```

Then open the `example_raycast_bunny.ncu-rep` report from Nsight Compute UI!
