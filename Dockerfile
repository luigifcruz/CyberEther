ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
      git build-essential pkg-config python3-pip \
      curl ca-certificates bison flex nasm \
      spirv-cross glslang-tools \
      libudev-dev \
      libglfw3-dev zenity \
      mesa-vulkan-drivers libvulkan-dev vulkan-validationlayers \
      libsoapysdr-dev soapysdr-module-rtlsdr && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages meson ninja numpy mapbox_earcut pyyaml

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
      -Dremote=enabled
RUN meson compile -C build
RUN meson install -C build
