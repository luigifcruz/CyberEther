ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
      git build-essential pkg-config python3-pip libpython3.12t64 \
      curl ca-certificates nasm xz-utils \
      spirv-cross glslang-tools \
      libudev-dev \
      libwayland-dev wayland-protocols libxkbcommon-dev \
      libglvnd-dev libx11-xcb-dev libdrm-dev \
      libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
      mesa-vulkan-drivers libvulkan-dev vulkan-validationlayers && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages meson ninja numpy

ARG ZIG_VERSION=0.17.0
RUN set -eux; \
    case "$(dpkg --print-architecture)" in \
      amd64) \
        zig_arch=x86_64; \
        zig_sha256=1cbe9df9f27e6b78d14ccbca43b6703a404ef79ef1c463de901d7f088d4e2026 ;; \
      arm64) \
        zig_arch=aarch64; \
        zig_sha256=9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8 ;; \
      *) \
        printf 'Unsupported Zig architecture: %s\n' "$(dpkg --print-architecture)" >&2; \
        exit 1 ;; \
    esac; \
    curl -fsSL "https://ziglang.org/download/${ZIG_VERSION}/zig-${zig_arch}-linux-${ZIG_VERSION}.tar.xz" -o /tmp/zig.tar.xz; \
    printf '%s  %s\n' "${zig_sha256}" /tmp/zig.tar.xz | sha256sum -c -; \
    mkdir -p /opt/zig; \
    tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1; \
    ln -s /opt/zig/zig /usr/local/bin/zig; \
    test "$(zig version)" = "${ZIG_VERSION}"; \
    rm /tmp/zig.tar.xz

COPY . /workspace
WORKDIR /workspace

RUN git config --global --add safe.directory /workspace

ARG BUILD_TYPE=release
ARG RUN_TESTS=false
ARG PREFIX=/usr

RUN rm -rf build && \
    meson setup build \
      --force-fallback-for=openssl \
      --default-library=shared \
      -Dopenssl:default_library=static \
      -Dprefix=${PREFIX} \
      -Dbuildtype=${BUILD_TYPE} \
      -Dexamples=false \
      -Dremote=enabled \
      -Dterminal=enabled \
      -Dinference=disabled && \
    meson compile -C build && \
    if [ "${RUN_TESTS}" = "true" ]; then \
      meson test -C build --print-errorlogs; \
    fi && \
    meson install -C build && \
    rm -rf build subprojects
