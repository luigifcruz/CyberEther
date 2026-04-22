ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
      git build-essential pkg-config python3-pip libssl-dev \
      spirv-cross glslang-tools \
      libglfw3-dev zenity \
      mesa-vulkan-drivers libvulkan-dev vulkan-validationlayers \
      libsoapysdr-dev soapysdr-module-rtlsdr \
      libgstreamer1.0-dev gstreamer1.0-libav gstreamer1.0-plugins-base \
      libgstreamer-plugins-bad1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev \
      gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages meson ninja numpy mapbox_earcut pyyaml

COPY . /workspace
WORKDIR /workspace

RUN git config --global --add safe.directory /workspace

ARG BUILD_TYPE=release
ARG PREFIX=/usr

RUN rm -rf build && meson setup build -Dprefix=${PREFIX} -Dbuildtype=${BUILD_TYPE}
RUN meson compile -C build
RUN meson install -C build
