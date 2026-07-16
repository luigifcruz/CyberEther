#include "jetstream/render/base/surface.hh"

namespace Jetstream::Render {

Surface::Surface(const Config& config) : config(config) {
    const auto addBuffer = [&](const std::shared_ptr<Buffer>& buffer) {
        if (buffer) {
            dependencyBuffers.insert(buffer);
        }
    };

    const auto addTexture = [&](const std::shared_ptr<Texture>& texture) {
        if (texture) {
            dependencyTextures.insert(texture);
        }
    };

    addTexture(config.framebuffer);

    for (const auto& kernel : config.kernels) {
        if (!kernel) {
            continue;
        }
        for (const auto& [buffer, _] : kernel->getConfig().buffers) {
            addBuffer(buffer);
        }
    }

    for (const auto& program : config.programs) {
        if (!program) {
            continue;
        }

        const auto& programConfig = program->getConfig();
        for (const auto& [buffer, _] : programConfig.buffers) {
            addBuffer(buffer);
        }
        for (const auto& texture : programConfig.textures) {
            addTexture(texture);
        }

        for (const auto& draw : programConfig.draws) {
            if (!draw) {
                continue;
            }

            dependencyDraws.insert(draw);

            const auto& vertex = draw->getConfig().buffer;
            if (!vertex) {
                continue;
            }

            const auto& vertexConfig = vertex->getConfig();
            addBuffer(vertexConfig.indices);
            for (const auto& [buffer, _] : vertexConfig.vertices) {
                addBuffer(buffer);
            }
            for (const auto& [buffer, _] : vertexConfig.instances) {
                addBuffer(buffer);
            }
        }
    }
}

void Surface::clearColor(const ColorRGBA<F32>& color) {
    if (config.clearColor.r != color.r ||
        config.clearColor.g != color.g ||
        config.clearColor.b != color.b ||
        config.clearColor.a != color.a) {
        invalidate();
    }
    config.clearColor = color;
}

void Surface::invalidate() {
    dirty = true;
}

void Surface::prepareFrame() {
    for (const auto& kernel : config.kernels) {
        if (kernel) {
            kernel->prepareForFrame();
        }
    }
}

void Surface::collectTransfers(Transfer::Batch& batch) const {
    for (const auto& buffer : dependencyBuffers) {
        batch.collect(buffer);
    }
    for (const auto& draw : dependencyDraws) {
        for (const auto& buffer : draw->transferBuffers) {
            batch.collect(buffer);
        }
    }
    for (const auto& texture : dependencyTextures) {
        batch.collect(texture);
    }
}

bool Surface::affectedBy(const Transfer::Batch& batch) const {
    for (const auto& buffer : dependencyBuffers) {
        if (batch.contains(buffer)) {
            return true;
        }
    }

    for (const auto& draw : dependencyDraws) {
        for (const auto& buffer : draw->transferBuffers) {
            if (batch.contains(buffer)) {
                return true;
            }
        }
    }

    for (const auto& texture : dependencyTextures) {
        if (batch.contains(texture)) {
            return true;
        }
    }

    return false;
}

void Surface::commitDraw() {
    if (drawPending) {
        for (const auto& kernel : config.kernels) {
            if (kernel) {
                kernel->commitFrame();
            }
        }
    }
    markDrawn();
}

const Extent2D<U64>& Surface::size() const {
    if (config.framebuffer) {
        return config.framebuffer->size();
    }
    return NullSize2D;
}

bool Surface::shouldDraw(bool framebufferChanged) {
    drawPending = drawPending || !config.retained || dirty || framebufferChanged;
    return drawPending;
}

void Surface::markDrawn() {
    if (drawPending) {
        dirty = false;
        drawPending = false;
    }
}

}  // namespace Jetstream::Render
