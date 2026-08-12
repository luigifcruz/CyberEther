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
# Callout: upstream ONNX Runtime publishes Linux CUDA binaries for x86_64 but
# not arm64, so the arm64 CUDA image uses NVIDIA's archive and layout instead.
ARG ONNXRUNTIME_CUDA_AARCH64_VERSION=1.24.2
ARG ONNXRUNTIME_CUDA_AARCH64_URL=https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/onnxruntime/onnxruntime-1.24.2-cuda-13.0-aarch64.tar.gz
ARG ONNXRUNTIME_CUDA_AARCH64_SHA256=a940bcb13e600ee30df5d560956a2647dc39ef1914bde1e4eb45b4c8cb4ee1c3
ENV PKG_CONFIG_PATH=/usr/local/lib64/pkgconfig
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    cuda=0; \
    [ ! -d /usr/local/cuda ] || cuda=1; \
    ort_deps=; \
    ort_include_dir=include; \
    ort_package=; \
    ort_sha256=; \
    ort_strip_components=1; \
    ort_url=; \
    ort_version="${ONNXRUNTIME_VERSION}"; \
    case "${arch}:${cuda}" in \
      amd64:1) \
        ort_package="onnxruntime-linux-x64-gpu_cuda${ONNXRUNTIME_CUDA_MAJOR}"; \
        ort_deps="libcudnn9-cuda-${ONNXRUNTIME_CUDA_MAJOR} libnvinfer10 libnvonnxparsers10" ;; \
      amd64:0) \
        ort_package=onnxruntime-linux-x64; \
        ort_deps= ;; \
      arm64:1) \
        ort_deps="libcudnn9-cuda-${ONNXRUNTIME_CUDA_MAJOR} libnvinfer10 libnvonnxparsers10"; \
        ort_include_dir=include/onnxruntime; \
        ort_sha256="${ONNXRUNTIME_CUDA_AARCH64_SHA256}"; \
        ort_strip_components=2; \
        ort_url="${ONNXRUNTIME_CUDA_AARCH64_URL}"; \
        ort_version="${ONNXRUNTIME_CUDA_AARCH64_VERSION}" ;; \
      arm64:*) \
        ort_package=onnxruntime-linux-aarch64; \
        ort_deps= ;; \
      *) \
        printf 'Unsupported architecture: %s\n' "${arch}" >&2; \
        exit 1 ;; \
    esac; \
    if [ -z "${ort_url}" ]; then \
      ort_url="https://github.com/microsoft/onnxruntime/releases/download/v${ort_version}/${ort_package}-${ort_version}.tgz"; \
    fi; \
    if [ -n "${ort_deps}" ]; then \
      apt-get update --fix-missing; \
      apt-get install -y --no-install-recommends ${ort_deps}; \
      apt-get clean; \
      rm -rf /var/lib/apt/lists/*; \
    fi; \
    curl -fsSL "${ort_url}" -o /tmp/onnxruntime.tgz; \
    if [ -n "${ort_sha256}" ]; then \
      printf '%s  %s\n' "${ort_sha256}" /tmp/onnxruntime.tgz | sha256sum -c -; \
    fi; \
    mkdir -p /tmp/onnxruntime /usr/local/include/onnxruntime /usr/local/lib64; \
    tar -xzf /tmp/onnxruntime.tgz -C /tmp/onnxruntime --strip-components="${ort_strip_components}"; \
    cp -a "/tmp/onnxruntime/${ort_include_dir}/." /usr/local/include/onnxruntime/; \
    cp -a /tmp/onnxruntime/lib/. /usr/local/lib64/; \
    mkdir -p /usr/local/lib64/pkgconfig; \
    { \
      printf '%s\n' 'prefix=/usr/local'; \
      printf '%s\n' 'libdir=${prefix}/lib64'; \
      printf '%s\n' 'includedir=${prefix}/include/onnxruntime'; \
      printf '\n'; \
      printf '%s\n' 'Name: onnxruntime'; \
      printf '%s\n' 'Description: ONNX Runtime'; \
      printf '%s\n' "Version: ${ort_version}"; \
      printf '%s\n' 'Libs: -L${libdir} -lonnxruntime'; \
      printf '%s\n' 'Cflags: -I${includedir}'; \
    } > /usr/local/lib64/pkgconfig/libonnxruntime.pc; \
    printf '%s\n' /usr/local/lib64 > /etc/ld.so.conf.d/onnxruntime.conf; \
    ldconfig; \
    rm -rf /tmp/onnxruntime /tmp/onnxruntime.tgz

COPY . /workspace
WORKDIR /workspace

RUN git config --global --add safe.directory /workspace

ARG BUILD_TYPE=release
ARG RUN_TESTS=false
ARG PREFIX=/usr

RUN rm -rf build && \
    meson setup build \
      --force-fallback-for=libffi,openssl \
      --default-library=shared \
      -Dlibffi:default_library=static \
      -Dopenssl:default_library=static \
      -Dprefix=${PREFIX} \
      -Dbuildtype=${BUILD_TYPE} \
      -Dexamples=false \
      -Dremote=enabled \
      -Dinference=enabled && \
    meson compile -C build && \
    if [ "${RUN_TESTS}" = "true" ]; then \
      meson test -C build --print-errorlogs; \
    fi && \
    meson install -C build && \
    rm -rf build subprojects
