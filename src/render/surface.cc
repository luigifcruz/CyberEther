#include "jetstream/render/base/surface.hh"

#include <algorithm>
#include <type_traits>
#include <unordered_map>

namespace Jetstream::Render {

namespace {

template<typename Visit>
void VisitDependencies(const std::vector<std::shared_ptr<Program>>& programs,
                       const std::vector<std::shared_ptr<Kernel>>& kernels,
                       Visit&& visit) {
    for (const auto& kernel : kernels) {
        if (!kernel) {
            continue;
        }
        for (const auto& [buffer, _] : kernel->getConfig().buffers) {
            visit(buffer);
        }
    }

    for (const auto& program : programs) {
        if (!program) {
            continue;
        }

        const auto& programConfig = program->getConfig();
        for (const auto& [buffer, _] : programConfig.buffers) {
            visit(buffer);
        }
        if (!programConfig.borrowedTextures) {
            for (const auto& texture : programConfig.textures) {
                visit(texture);
            }
        }

        for (const auto& draw : programConfig.draws) {
            if (!draw) {
                continue;
            }

            visit(draw);

            const auto& vertex = draw->getConfig().buffer;
            if (!vertex) {
                continue;
            }

            const auto& vertexConfig = vertex->getConfig();
            visit(vertexConfig.indices);
            for (const auto& [buffer, _] : vertexConfig.vertices) {
                visit(buffer);
            }
            for (const auto& [buffer, _] : vertexConfig.instances) {
                visit(buffer);
            }
        }
    }
}

}  // namespace

Surface::Surface(const Config& config) : config(config) {
    if (config.framebuffer) {
        dependencyTextures.insert(config.framebuffer);
        requestedSize = config.framebuffer->size();
    }
    track(config.programs, config.kernels);
}

Surface::~Surface() {
    releaseObjects();
}

Result Surface::claimObjects(const std::vector<std::shared_ptr<Program>>& programs,
                             const std::vector<std::shared_ptr<Kernel>>& kernels) {
    std::vector<std::atomic_bool*> claims;
    std::unordered_set<std::atomic_bool*> seen;
    const auto add = [&](auto& object) {
        return object && seen.insert(&object->surfaceOwned).second &&
               (claims.push_back(&object->surfaceOwned), true);
    };
    for (const auto& kernel : kernels) {
        if (!add(kernel)) {
            return Result::ERROR;
        }
    }
    for (const auto& program : programs) {
        if (!add(program)) {
            return Result::ERROR;
        }
        for (const auto& draw : program->getConfig().draws) {
            if (!add(draw)) {
                return Result::ERROR;
            }
            const auto& vertex = draw->getConfig().buffer;
            if (vertex && !add(vertex)) {
                return Result::ERROR;
            }
        }
    }
    const U64 first = objectClaims.size();
    objectClaims.reserve(first + claims.size());
    for (auto* claim : claims) {
        if (claim->exchange(true, std::memory_order_acq_rel)) {
            releaseObjects(first);
            return Result::ERROR;
        }
        objectClaims.push_back(claim);
    }
    return Result::SUCCESS;
}

void Surface::releaseObjects(U64 first) {
    while (objectClaims.size() > first) {
        objectClaims.back()->store(false, std::memory_order_release);
        objectClaims.pop_back();
    }
}

void Surface::track(const std::vector<std::shared_ptr<Program>>& programs,
                    const std::vector<std::shared_ptr<Kernel>>& kernels) {
    track(programs, kernels, dependencyBuffers, dependencyDraws, dependencyTextures, borrowedTextures);
}

void Surface::track(const std::vector<std::shared_ptr<Program>>& programs,
                    const std::vector<std::shared_ptr<Kernel>>& kernels,
                    std::unordered_set<std::shared_ptr<Buffer>>& buffers,
                    std::unordered_set<std::shared_ptr<Draw>>& draws,
                    std::unordered_set<std::shared_ptr<Texture>>& textures,
                    std::unordered_set<std::shared_ptr<Texture>>& borrowed) {
    for (const auto& program : programs) {
        if (program && program->getConfig().borrowedTextures) {
            for (const auto& texture : program->getConfig().textures) {
                if (texture) {
                    borrowed.insert(texture);
                }
            }
        }
    }
    VisitDependencies(programs, kernels, [&](const auto& dependency) {
        if (!dependency) {
            return;
        }
        using T = std::decay_t<decltype(dependency)>;
        if constexpr (std::is_same_v<T, std::shared_ptr<Buffer>>) {
            buffers.insert(dependency);
        } else if constexpr (std::is_same_v<T, std::shared_ptr<Texture>>) {
            textures.insert(dependency);
        } else {
            draws.insert(dependency);
        }
    });
}

std::vector<std::shared_ptr<WindowAttachment>> Surface::untracked(
    const std::vector<std::shared_ptr<Program>>& programs,
    const std::vector<std::shared_ptr<Kernel>>& kernels) const {
    std::vector<std::shared_ptr<WindowAttachment>> resources;
    std::unordered_set<const WindowAttachment*> seen;
    VisitDependencies(programs, kernels, [&](const auto& dependency) {
        if (!dependency) {
            return;
        }
        using T = std::decay_t<decltype(dependency)>;
        if constexpr (std::is_same_v<T, std::shared_ptr<Buffer>>) {
            if (!dependencyBuffers.contains(dependency) && seen.insert(dependency.get()).second) {
                resources.push_back(dependency);
            }
        } else if constexpr (std::is_same_v<T, std::shared_ptr<Texture>>) {
            if (!dependencyTextures.contains(dependency) && seen.insert(dependency.get()).second) {
                resources.push_back(dependency);
            }
        }
    });
    return resources;
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

Result Surface::reorder(const std::vector<std::shared_ptr<Program>>& programs) {
    if (programs == config.programs) {
        return Result::SUCCESS;
    }
    if (programs.size() != config.programs.size()) {
        return Result::ERROR;
    }
    std::unordered_map<const Program*, I64> counts;
    for (const auto& program : config.programs) {
        ++counts[program.get()];
    }
    for (const auto& program : programs) {
        if (--counts[program.get()] < 0) {
            return Result::ERROR;
        }
    }
    auto candidate = programs;
    JST_CHECK(reorderPrograms(candidate));
    config.programs.swap(candidate);
    invalidate();
    return Result::SUCCESS;
}

Result Surface::reorderPrograms(const std::vector<std::shared_ptr<Program>>&) {
    return Result::SUCCESS;
}

bool Surface::appendable() const {
    return false;
}

Result Surface::validateAppend(const std::vector<std::shared_ptr<Program>>& programs,
                               const std::vector<std::shared_ptr<Kernel>>& kernels) const {
    for (const auto& program : programs) {
        if (!program) {
            return Result::ERROR;
        }
    }
    for (const auto& kernel : kernels) {
        if (!kernel) {
            return Result::ERROR;
        }
    }
    return Result::SUCCESS;
}

Result Surface::createAppended(const std::vector<std::shared_ptr<Program>>&,
                               const std::vector<std::shared_ptr<Kernel>>&) {
    return Result::ERROR;
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
    for (const auto& texture : borrowedTextures) {
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
    for (const auto& program : config.programs) {
        if (program && program->enabled() && program->getConfig().borrowedTextures) {
            for (const auto& texture : program->getConfig().textures) {
                if (batch.contains(texture)) {
                    return true;
                }
            }
        }
    }

    return false;
}

void Surface::commitDraw() {
    if (textureBindingsResult != Result::SUCCESS) {
        return;
    }
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

const Extent2D<U64>& Surface::size(const Extent2D<U64>& size) {
    if (!config.framebuffer) {
        return NullSize2D;
    }

    requestedSize = size;
    invalidate();
    if (created() && config.framebuffer->size() != size) {
        const bool resized = framebufferResized;
        if (prepare() != Result::SUCCESS) {
            JST_ERROR("[SURFACE] Failed to resize surface framebuffer.");
        }
        framebufferResized = framebufferResized || resized;
    }

    return config.framebuffer->size();
}

Result Surface::prepare() {
    if (!config.framebuffer) {
        return Result::SUCCESS;
    }
    framebufferResized = config.framebuffer->size(requestedSize) || resizeResult != Result::SUCCESS;
    if (framebufferResized) {
        resetFramebuffer();
        resizeResult = rebuildFramebuffer();
    }
    return resizeResult;
}

bool Surface::created() const {
    return false;
}

Result Surface::rebuildFramebuffer() {
    return Result::SUCCESS;
}

Result Surface::refreshBorrowedTextures() {
    std::vector<std::tuple<const Texture*, U64, U64>> revisions;
    for (const auto& program : config.programs) {
        if (!program || !program->enabled() || !program->getConfig().borrowedTextures) {
            continue;
        }
        JST_CHECK(program->refreshTextureBindings());
        for (const auto& texture : program->getConfig().textures) {
            if (texture) {
                revisions.emplace_back(texture.get(), texture->contentRevision(), texture->allocation());
            }
        }
    }
    if (revisions != borrowedRevisions) {
        invalidate();
        borrowedRevisions = std::move(revisions);
    }
    return Result::SUCCESS;
}

Result Surface::drawResult() const {
    return resizeResult != Result::SUCCESS ? resizeResult : textureBindingsResult;
}

bool Surface::shouldDraw(bool framebufferChanged) {
    graphicsSkipped = false;
    if (framebufferChanged) {
        resetFramebuffer();
    }
    for (const auto& program : config.programs) {
        if (program && program->stateDirty) {
            invalidate();
            program->stateDirty = false;
        }
    }
    textureBindingsResult = refreshBorrowedTextures();
    if (textureBindingsResult != Result::SUCCESS) {
        invalidate();
        drawPending = true;
        return false;
    }
    drawPending = drawPending || !config.retained || dirty || framebufferChanged;
    return drawPending;
}

bool Surface::hiddenAndIdle(bool visible, bool framebufferChanged) const {
    if (visible || framebufferChanged || !config.retained || !imageInitialized) {
        return false;
    }
    return !kernelsPending();
}

bool Surface::kernelsPending() const {
    return std::any_of(config.kernels.begin(), config.kernels.end(), [](const auto& kernel) {
        return kernel && kernel->shouldEncode();
    });
}

bool Surface::shouldDrawGraphics(bool visible) {
    // Initialize new images even when hidden; only skip the graphics pass.
    graphicsSkipped = !visible && imageInitialized;
    if (!graphicsSkipped && config.framebuffer) {
        config.framebuffer->markContentChanged();
    }
    return !graphicsSkipped;
}

void Surface::resetFramebuffer() {
    imageInitialized = false;
    invalidate();
}

void Surface::markDrawn() {
    if (drawPending) {
        imageInitialized = imageInitialized || !graphicsSkipped;
        dirty = graphicsSkipped;
        drawPending = false;
    }
}

}  // namespace Jetstream::Render
