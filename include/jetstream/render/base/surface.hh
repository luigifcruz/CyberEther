#ifndef JETSTREAM_RENDER_BASE_SURFACE_HH
#define JETSTREAM_RENDER_BASE_SURFACE_HH

#include <memory>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "jetstream/types.hh"
#include "jetstream/logger.hh"
#include "jetstream/render/base/texture.hh"
#include "jetstream/render/base/program.hh"
#include "jetstream/render/base/kernel.hh"
#include "jetstream/render/types.hh"
#include "jetstream/render/base/implementations.hh"
#include "jetstream/render/base/window_attachment.hh"

namespace Jetstream::Render {

class JETSTREAM_API Surface : public WindowAttachment {
 public:
    struct Config {
        std::shared_ptr<Texture> framebuffer;
        std::vector<std::shared_ptr<Kernel>> kernels;
        std::vector<std::shared_ptr<Program>> programs;
        ColorRGBA<F32> clearColor = {0.0f, 0.0f, 0.0f, 1.0f};
        bool multisampled = false;
        bool retained = false;
        bool composite = false;
    };

    explicit Surface(const Config& config);
    virtual ~Surface();

    Type type() const override {
        return Type::Surface;
    }

    const Config& getConfig() const {
        return config;
    }

    constexpr const bool& multisampled() const {
        return config.multisampled;
    }

    constexpr const bool& retained() const {
        return config.retained;
    }

    void clearColor(const ColorRGBA<F32>& color);
    void invalidate();
    Result reorder(const std::vector<std::shared_ptr<Program>>& programs);

    void commitDraw();

    const Extent2D<U64>& size() const;
    virtual const Extent2D<U64>& size(const Extent2D<U64>& size);

    template<DeviceType D>
    static std::shared_ptr<Surface> Factory(const Config& config) {
        return std::make_shared<SurfaceImp<D>>(config);
    }

 protected:
    Config config;
    bool dirty = true;
    Extent2D<U64> requestedSize;
    bool framebufferResized = false;
    Result resizeResult = Result::SUCCESS;

    Result prepare();
    virtual bool created() const;
    virtual Result rebuildFramebuffer();

    virtual Result reorderPrograms(const std::vector<std::shared_ptr<Program>>& programs);
    virtual bool appendable() const;
    virtual Result validateAppend(const std::vector<std::shared_ptr<Program>>& programs,
                                  const std::vector<std::shared_ptr<Kernel>>& kernels) const;
    virtual Result createAppended(const std::vector<std::shared_ptr<Program>>& programs,
                                  const std::vector<std::shared_ptr<Kernel>>& kernels);

    template<DeviceType D>
    static Result validateObjects(const std::vector<std::shared_ptr<Program>>& programs,
                                  const std::vector<std::shared_ptr<Kernel>>& kernels) {
        const auto validBuffer = [](const auto& buffer) {
            return std::dynamic_pointer_cast<BufferImp<D>>(buffer) != nullptr;
        };
        for (const auto& kernel : kernels) {
            if (!std::dynamic_pointer_cast<KernelImp<D>>(kernel)) {
                return Result::ERROR;
            }
            for (const auto& [buffer, _] : kernel->getConfig().buffers) {
                if (!validBuffer(buffer)) {
                    return Result::ERROR;
                }
            }
        }
        for (const auto& program : programs) {
            if (!std::dynamic_pointer_cast<ProgramImp<D>>(program)) {
                return Result::ERROR;
            }
            for (const auto& [buffer, _] : program->getConfig().buffers) {
                if (!validBuffer(buffer)) {
                    return Result::ERROR;
                }
            }
            for (const auto& texture : program->getConfig().textures) {
                if (!std::dynamic_pointer_cast<TextureImp<D>>(texture)) {
                    return Result::ERROR;
                }
            }
            for (const auto& draw : program->getConfig().draws) {
                if (!std::dynamic_pointer_cast<DrawImp<D>>(draw)) {
                    return Result::ERROR;
                }
                const auto& vertex = draw->getConfig().buffer;
                if (!std::dynamic_pointer_cast<VertexImp<D>>(vertex)) {
                    return Result::ERROR;
                }
                const auto& vertexConfig = vertex->getConfig();
                if (vertexConfig.indices && !validBuffer(vertexConfig.indices)) {
                    return Result::ERROR;
                }
                for (const auto& [buffer, _] : vertexConfig.vertices) {
                    if (!validBuffer(buffer)) {
                        return Result::ERROR;
                    }
                }
                for (const auto& [buffer, _] : vertexConfig.instances) {
                    if (!validBuffer(buffer)) {
                        return Result::ERROR;
                    }
                }
            }
        }
        return Result::SUCCESS;
    }

    template<typename P, typename K, typename Create, typename Destroy>
    static Result appendObjects(const std::vector<std::shared_ptr<Program>>& newPrograms,
                                const std::vector<std::shared_ptr<Kernel>>& newKernels,
                                std::vector<std::shared_ptr<P>>& programs,
                                std::vector<std::shared_ptr<K>>& kernels,
                                Create&& create, Destroy&& destroy) {
        auto candidatePrograms = programs;
        auto candidateKernels = kernels;
        for (const auto& program : newPrograms) {
            auto typed = std::dynamic_pointer_cast<P>(program);
            if (!typed) {
                return Result::ERROR;
            }
            candidatePrograms.push_back(std::move(typed));
        }
        for (const auto& kernel : newKernels) {
            auto typed = std::dynamic_pointer_cast<K>(kernel);
            if (!typed) {
                return Result::ERROR;
            }
            candidateKernels.push_back(std::move(typed));
        }
        U64 createdPrograms = programs.size();
        U64 createdKernels = kernels.size();
        const auto rollback = [&]() {
            Result result = Result::SUCCESS;
            while (createdKernels > kernels.size()) {
                const Result cleanup = destroy(candidateKernels[--createdKernels]);
                if (cleanup != Result::SUCCESS && cleanup != Result::RELOAD) {
                    result = cleanup;
                }
            }
            while (createdPrograms > programs.size()) {
                const Result cleanup = destroy(candidatePrograms[--createdPrograms]);
                if (cleanup != Result::SUCCESS && cleanup != Result::RELOAD) {
                    result = cleanup;
                }
            }
            return result;
        };
        const auto attempt = [&](auto& object) {
            const Result result = create(object);
            if (result != Result::SUCCESS && result != Result::RELOAD) {
                const Result cleanup = rollback();
                return cleanup == Result::SUCCESS ? result : cleanup;
            }
            return Result::SUCCESS;
        };
        try {
            while (createdPrograms < candidatePrograms.size()) {
                const Result result = attempt(candidatePrograms[createdPrograms++]);
                if (result != Result::SUCCESS) {
                    return result;
                }
            }
            while (createdKernels < candidateKernels.size()) {
                const Result result = attempt(candidateKernels[createdKernels++]);
                if (result != Result::SUCCESS) {
                    return result;
                }
            }
        } catch (...) {
            rollback();
            throw;
        }
        programs.swap(candidatePrograms);
        kernels.swap(candidateKernels);
        return Result::SUCCESS;
    }

    bool shouldDraw(bool framebufferChanged = false);
    Result drawResult() const;
    bool hiddenAndIdle(bool visible, bool framebufferChanged) const;
    bool kernelsPending() const;
    bool shouldDrawGraphics(bool visible);
    void resetFramebuffer();
    void markDrawn();

 private:
    Result claimObjects(const std::vector<std::shared_ptr<Program>>& programs,
                        const std::vector<std::shared_ptr<Kernel>>& kernels);
    void releaseObjects(U64 first = 0);
    std::vector<std::atomic_bool*> objectClaims;

    void prepareFrame();
    void collectTransfers(Transfer::Batch& batch) const;
    bool affectedBy(const Transfer::Batch& batch) const;
    std::vector<std::shared_ptr<WindowAttachment>> untracked(const std::vector<std::shared_ptr<Program>>& programs,
                                                             const std::vector<std::shared_ptr<Kernel>>& kernels) const;
    void track(const std::vector<std::shared_ptr<Program>>& programs,
               const std::vector<std::shared_ptr<Kernel>>& kernels);
    static void track(const std::vector<std::shared_ptr<Program>>& programs,
                      const std::vector<std::shared_ptr<Kernel>>& kernels,
                      std::unordered_set<std::shared_ptr<Buffer>>& buffers,
                      std::unordered_set<std::shared_ptr<Draw>>& draws,
                      std::unordered_set<std::shared_ptr<Texture>>& textures,
                      std::unordered_set<std::shared_ptr<Texture>>& borrowed);

    Result refreshBorrowedTextures();

    bool drawPending = false;
    bool imageInitialized = false;
    bool graphicsSkipped = false;
    std::unordered_set<std::shared_ptr<Buffer>> dependencyBuffers;
    std::unordered_set<std::shared_ptr<Draw>> dependencyDraws;
    std::unordered_set<std::shared_ptr<Texture>> dependencyTextures;
    std::unordered_set<std::shared_ptr<Texture>> borrowedTextures;
    std::vector<std::tuple<const Texture*, U64, U64>> borrowedRevisions;
    Result textureBindingsResult = Result::SUCCESS;

    friend class Window;
};

}  // namespace Jetstream::Render

#endif
