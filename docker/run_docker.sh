#!/bin/bash

# allow local connections to host system xserver
xhost +local:root >/dev/null 2>/dev/null

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TPL_DIR="$SCRIPT_DIR/.."

CONTAINER_NAME="tpl_base"

docker buildx build \
    -f "$TPL_DIR/docker/$CONTAINER_NAME.Dockerfile" \
    -t $CONTAINER_NAME \
    --build-arg ARCHITECTURE=$(uname -m) \
    $TPL_DIR

lspci | grep NVIDIA >/dev/null 2>&1
NVIDIA_GPU=$?

if [ ${NVIDIA_GPU} == 0 ] ; then
    DOCKER_ARGS=(--gpus all -v /usr/share/glvnd/egl_vendor.d/:/usr/share/glvnd/egl_vendor.d/)
else
    DOCKER_ARGS=()
fi

DOCKER_ARGS+=(
    --privileged
    --network=host
    --ipc=host
    --hostname "$(hostname)"
    --device /dev/dri
    --name "$CONTAINER_NAME"
    --workdir "/workspace/tpl"
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw
    -v $TPL_DIR:/workspace/tpl
    -v /etc/group:/etc/group:ro
    -v /etc/passwd:/etc/passwd:ro
    -v /etc/shadow:/etc/shadow:ro
    -v /etc/sudoers:/etc/sudoers:ro
    -v "$TPL_DIR/docker/docker_home_cache/":"/home"
    -e CWD=$CWD
    -e DISPLAY=$DISPLAY
    -e HOME=$HOME
    -e DOCKER_MACHINE_NAME="$CONTAINER_NAME"
    -e CONTAINER_NAME="$CONTAINER_NAME"
    -e TARGET_USER="$USER"
    "$CONTAINER_NAME"
)

if [[ $(docker ps -f "name=$CONTAINER_NAME" --format '{{.Names}}') == $CONTAINER_NAME ]]
then
  echo "Connecting to existing container."
  docker exec -it $CONTAINER_NAME "/workspace/tpl/docker/entrypoint.sh"
else
  echo "Creating a new docker container"
  docker run --rm -it ${DOCKER_ARGS[@]} "/workspace/tpl/docker/entrypoint.sh"
fi
