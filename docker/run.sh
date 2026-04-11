#!/bin/bash
# Build and run Docker container with NPU access
#
# Usage:
#   cd /path/to/xPU-simulator
#   bash docker/run.sh [build|start|shell|stop]

IMAGE_NAME="xpu-simulator"
CONTAINER_NAME="xpu-sim-dev"

case "${1:-build}" in

  build)
    echo "Building Docker image..."
    docker build -t ${IMAGE_NAME} -f docker/Dockerfile .
    echo "Done. Run: bash docker/run.sh start"
    ;;

  start)
    echo "Starting container with NPU access..."
    docker run -dit \
      --name ${CONTAINER_NAME} \
      --network host \
      --device /dev/davinci0 \
      --device /dev/davinci_manager \
      --device /dev/devmm_svm \
      --device /dev/hisi_hdc \
      -v /usr/local/Ascend:/usr/local/Ascend:ro \
      -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
      -v $(pwd):/workspace/xpu-simulator \
      -v $(pwd)/profiling_data:/workspace/profiling_data \
      ${IMAGE_NAME}
    echo "Container started. Run: bash docker/run.sh shell"
    ;;

  shell)
    echo "Attaching to container..."
    docker exec -it ${CONTAINER_NAME} /bin/bash
    ;;

  stop)
    echo "Stopping container..."
    docker stop ${CONTAINER_NAME} && docker rm ${CONTAINER_NAME}
    echo "Done."
    ;;

  *)
    echo "Usage: bash docker/run.sh [build|start|shell|stop]"
    ;;

esac
