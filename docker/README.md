# Docker Setup for NPU Validation

Run xPU-simulator on a real Ascend NPU server without affecting other users.

## Prerequisites

On the NPU server:
- Docker installed
- CANN toolkit installed at `/usr/local/Ascend/`
- NPU devices available (`ls /dev/davinci*`)

## Quick Start

```bash
# 1. Clone repo on the server
git clone https://github.com/rainbay001-dotcom/xPU-simulator.git
cd xPU-simulator

# 2. Build image
bash docker/run.sh build

# 3. Start container (mounts NPU devices + CANN)
bash docker/run.sh start

# 4. Enter container
bash docker/run.sh shell

# 5. Inside container — verify NPU access
npu-smi info
python3 -c "import torch; import torch_npu; print(torch.npu.is_available())"

# 6. Run profiling benchmark
python3 benchmarks/profile_on_npu.py

# 7. When done
exit
bash docker/run.sh stop
```

## What Gets Mounted

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `/usr/local/Ascend` | `/usr/local/Ascend` (ro) | CANN toolkit (shared, read-only) |
| `./` (repo) | `/workspace/xpu-simulator` | Our code (live edits) |
| `./profiling_data` | `/workspace/profiling_data` | Output data (persisted) |

## Multi-NPU

To use multiple NPUs, add more `--device` flags:
```bash
--device /dev/davinci0 --device /dev/davinci1 ... --device /dev/davinci7
```

## Notes

- Container uses `--network host` for easy access
- CANN is mounted read-only from host — no need to install inside container
- Your work is fully isolated from other users on the server
