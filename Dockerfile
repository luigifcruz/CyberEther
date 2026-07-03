ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
      git build-essential pkg-config python3-pip \
      curl ca-certificates bison flex nasm \
      spirv-cross glslang-tools \
      libudev-dev \
      libwayland-dev wayland-protocols libxkbcommon-dev \
      libglvnd-dev libx11-xcb-dev libdrm-dev \
      libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
      mesa-vulkan-drivers libvulkan-dev vulkan-validationlayers && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages meson ninja numpy mapbox_earcut

ARG ONNXRUNTIME_VERSION=1.27.0
ARG ONNXRUNTIME_CUDA_MAJOR=13
ENV PKG_CONFIG_PATH=/usr/local/lib64/pkgconfig
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    cuda=0; \
    [ ! -d /usr/local/cuda ] || cuda=1; \
    case "${arch}:${cuda}" in \
      amd64:1) \
        ort_package="onnxruntime-linux-x64-gpu_cuda${ONNXRUNTIME_CUDA_MAJOR}"; \
        ort_deps="libcudnn9-cuda-${ONNXRUNTIME_CUDA_MAJOR} libnvinfer10 libnvonnxparsers10" ;; \
      amd64:0) \
        ort_package=onnxruntime-linux-x64; \
        ort_deps= ;; \
      arm64:*) \
        ort_package=onnxruntime-linux-aarch64; \
        ort_deps= ;; \
      *) \
        printf 'Unsupported architecture: %s\n' "${arch}" >&2; \
        exit 1 ;; \
    esac; \
    if [ -n "${ort_deps}" ]; then \
      apt-get update --fix-missing; \
      apt-get install -y --no-install-recommends ${ort_deps}; \
      apt-get clean; \
      rm -rf /var/lib/apt/lists/*; \
    fi; \
    curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ort_package}-${ONNXRUNTIME_VERSION}.tgz" \
      -o /tmp/onnxruntime.tgz; \
    mkdir -p /tmp/onnxruntime /usr/local/include/onnxruntime /usr/local/lib64; \
    tar -xzf /tmp/onnxruntime.tgz -C /tmp/onnxruntime --strip-components=1; \
    cp -a /tmp/onnxruntime/include/. /usr/local/include/onnxruntime/; \
    cp -a /tmp/onnxruntime/lib/. /usr/local/lib64/; \
    printf '%s\n' /usr/local/lib64 > /etc/ld.so.conf.d/onnxruntime.conf; \
    ldconfig; \
    rm -rf /tmp/onnxruntime /tmp/onnxruntime.tgz

COPY . /workspace
WORKDIR /workspace

RUN git config --global --add safe.directory /workspace

ARG BUILD_TYPE=release
ARG PREFIX=/usr

RUN rm -rf build && \
    meson setup build \
      --force-fallback-for=libffi,openssl \
      --default-library=static \
      -Dlibffi:default_library=static \
      -Dopenssl:default_library=static \
      -Dprefix=${PREFIX} \
      -Dbuildtype=${BUILD_TYPE} \
      -Dexamples=false \
      -Dremote=enabled \
      -Dinference=enabled
RUN meson compile -C build
RUN meson install -C build
