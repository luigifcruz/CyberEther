# Docker
These are configuration files for building and running the docker image of CyberEther. These are mainly used for development and testing purposes.

## Build Docker Image

```bash
# Arch Linux Image
docker build -t cyberether-arch -f Dockerfile-arch .

# Ubuntu Image
docker build -t cyberether-ubuntu -f Dockerfile-ubuntu .
```

## Run Docker Image

```bash
docker run --rm -it --gpus all --entrypoint bash cyberether-arch
``` 
