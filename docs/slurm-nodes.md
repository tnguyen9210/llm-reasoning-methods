# SLURM nodes — current access

Nodes backing my currently-running jobs in the `gpu_windfall`
partition (`squeue` snapshot, captured 2026-07-10). All jobs are
`jupyter` under user `tnguyen9`. Hardware and co-tenancy checked
via `scontrol show node/job` and `squeue -w <node>`.

## Unique nodes

All 5 nodes are identical hardware: 4× Volta GPU
(`gpu:volta:4`), 96 CPU (94 usable), 504GB RAM
(`gpu_windfall`+`gpu_high_priority` partitions). Each of my jobs
requests 1 GPU + 22 CPUs.

| node | my jobs | my GPUs used | other tenant | node fully mine? |
|---|---|---|---|---|
| r5u35n1 | 4 | 4 of 4 | — | yes |
| r5u03n1 | 2 | 2 of 4 | krishnaghant (46 CPU) | no |
| r5u09n1 | 2 | 2 of 4 | rhyaanm (46 CPU) | no |
| r5u11n1 | 2 | 2 of 4 | rhyaanm (46 CPU) | no |
| r5u37n1 | 2 | 2 of 4 | rhyaanm (46 CPU) | no |

So of my 12 jobs, only `r5u35n1` is a node where I hold every
GPU; the other 4 nodes are shared 2-and-2 with another user's job
(a single ~46-CPU job on each, likely also claiming the other 2
GPUs, though that job's own GPU count wasn't checked here).

## Full job listing (with live GPU usage)

Live usage via `srun --jobid=<id> --overlap nvidia-smi
--query-gpu=...` — each job's cgroup remaps its assigned physical
GPU to logical index 0, so this reads only *that job's* GPU, not
a neighbor's, even on shared nodes. Captured 2026-07-10, same
sitting as the table above.

| job id | node | runtime | GPU util | mem used | temp | power |
|---|---|---|---|---|---|---|
| 22828653 | r5u09n1 | 1-05:56:22 | 100% | 24654 MiB | 66°C | 238 W |
| 22839193 | r5u37n1 | 21:08:33 | 100% | 24610 MiB | 62°C | 229 W |
| 22839178 | r5u37n1 | 21:13:35 | 98% | 24530 MiB | 67°C | 246 W |
| 22839165 | r5u35n1 | 21:18:37 | 100% | 24786 MiB | 66°C | 247 W |
| 22839107 | r5u11n1 | 21:18:38 | 100% | 24706 MiB | 66°C | 208 W |
| 22823505 | r5u35n1 | 1-13:08:08 | 100% | 24610 MiB | 64°C | 261 W |
| 22823450 | r5u03n1 | 1-13:13:09 | 100% | 26350 MiB | 64°C | 197 W |
| 22823451 | r5u03n1 | 1-13:13:09 | 100% | 27286 MiB | 65°C | 208 W |
| 22823452 | r5u35n1 | 1-13:13:09 | 98% | 24630 MiB | 67°C | 253 W |
| 22814236 | r5u09n1 | 2-03:20:20 | 100% | 25576 MiB | 63°C | 197 W |
| 22814237 | r5u11n1 | 2-03:20:20 | 99% | 26162 MiB | 69°C | 230 W |
| **22811363** | **r5u35n1** | **2-14:34:45** | **0%** | **0 MiB** | 33°C | 25 W |

All GPUs are Tesla V100S-PCIE-32GB (32768 MiB total each).

## Notes

- This is a point-in-time snapshot (`squeue`/`scontrol`/
  `nvidia-smi`, captured 2026-07-10) — re-run `squeue -u
  tnguyen9210`, `scontrol show node <name>`, and the `srun
  --overlap nvidia-smi` command above to refresh; don't treat
  this as current truth on future reads.
- `r5u35n1`'s 4 jobs are not duplication — each requests a
  distinct 1 GPU + 22 CPU slice, and together they claim all 4
  GPUs / 88 of 96 CPUs on the node. Expected packing, not a
  mistake.
- On the 4 shared nodes, the co-tenant's own GPU allocation wasn't
  individually checked (`scontrol show job <their-id>`) — assumed
  likely 2 GPUs each to round out the node's 4, but not confirmed.
- **`22811363` (r5u35n1, oldest job, 2-14:34:45 runtime) is
  idle: 0% GPU util, 0 MiB memory, 33°C.** Every other job is
  fully saturated (98-100% util, ~25-27GB used). Worth checking
  whether this jupyter session is genuinely idle (fine to keep or
  free up the GPU) vs. something that crashed/hung inside it —
  the other 11 GPUs are actively being used and this is the one
  outlier.
