docker run \
    --rm \
    --runtime=nvidia \
    --gpus 1 \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v ${PWD}:/workspace \
    -w /workspace \
    -it \
    openfoam-python:latest
