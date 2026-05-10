ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
      git build-essential pkg-config python3-pip \
      curl ca-certificates bison flex nasm \
      spirv-cross glslang-tools \
      libglfw3-dev zenity \
      mesa-vulkan-drivers libvulkan-dev vulkan-validationlayers \
      libsoapysdr-dev soapysdr-module-rtlsdr && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages meson ninja numpy mapbox_earcut pyyaml

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --profile minimal && \
    /root/.cargo/bin/cargo install cargo-c --locked

ENV PATH="/root/.cargo/bin:${PATH}"

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
      -Dremote=enabled
RUN meson compile -C build
RUN meson install -C build
