# Docker
These are configuration files for building and running the docker image of CyberEther. These are mainly used for development and testing purposes.

## Build Docker Image

```bash
# Arch Linux Image
docker build -t cyberether-arch -f ./docker/Dockerfile-arch .

# Ubuntu Image
docker build -t cyberether-ubuntu -f ./docker/Dockerfile-ubuntu .

# Emscripten Image
docker build -t cyberether-emscripten -f ./docker/Dockerfile-emscripten .
```

## Run Docker Image

```bash
docker run --rm -it --entrypoint bash cyberether-arch
``` 
