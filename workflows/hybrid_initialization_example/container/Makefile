# For arch naming conventions, refer
# https://docs.docker.com/build/building/multi-platform/
# https://github.com/containerd/containerd/blob/v1.4.3/platforms/platforms.go#L86
ARCH := $(shell uname -p)

ifeq ($(ARCH), x86_64)
    TARGETPLATFORM := "linux/amd64"
else ifeq ($(ARCH), aarch64)
    TARGETPLATFORM := "linux/arm64"
else
    $(error Unknown CPU architecture ${ARCH} detected)
endif

container:
	docker build -t openfoam-python:latest --build-arg TARGETPLATFORM=${TARGETPLATFORM} -f Dockerfile .

run:
	docker run --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/workspace -w /workspace -it openfoam-python:latest
