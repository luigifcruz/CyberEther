#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <limits>
#include <new>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "jetstream/render/base/buffer.hh"
#include "jetstream/render/base/surface.hh"
#include "jetstream/render/base/transfer.hh"
#include "jetstream/render/base/window.hh"
#include "jetstream/viewport/adapters/generic.hh"

using namespace Jetstream;

namespace {

class TestBuffer final : public Render::Buffer {
 public:
    explicit TestBuffer(const Config& config) : Buffer(config) {}

    Result create() override {
        createCount++;
        live = true;
        return creationHandler ? creationHandler() : Result::SUCCESS;
    }
    Result destroy() override {
        destroyCount++;
        if (destructionResult == Result::SUCCESS) {
            live = false;
        }
        return destructionResult;
    }

    void nextDestructionResult(const Result result) { destructionResult = result; }

    U64 createCount = 0;
    U64 destroyCount = 0;
    bool live = false;
    std::function<Result()> creationHandler;

 private:
    Result destructionResult = Result::SUCCESS;
};

class TestTexture final : public Render::Texture {
 public:
    explicit TestTexture(const Config& config) : Texture(config) {}

    Result create() override {
        createCount++;
        if (creationResult == Result::SUCCESS) {
            markAllocated();
        }
        return creationHandler ? creationHandler() : creationResult;
    }
    Result destroy() override {
        destroyCount++;
        markDeallocated();
        return Result::SUCCESS;
    }
    uint64_t raw() const override { return 0; }

    void nextCreationResult(const Result result) { creationResult = result; }

    U64 createCount = 0;
    U64 destroyCount = 0;
    std::function<Result()> creationHandler;

 private:
    Result creationResult = Result::SUCCESS;
};

class RebindingProgram final : public Render::Program {
 public:
    explicit RebindingProgram(const Config& config) : Program(config) {}

    Result rebindTextures() override {
        rebinds++;
        return std::exchange(rebindResult, Result::SUCCESS);
    }

    U64 rebinds = 0;
    Result rebindResult = Result::SUCCESS;
};

class TestDraw final : public Render::Draw {
 public:
    explicit TestDraw(const Config& config) : Draw(config) {}

    Result updateVertexCount(U64 vertexCount) override {
        config.numberOfDraws = vertexCount;
        return Result::SUCCESS;
    }

    Result updateInstanceCount(U64 instanceCount) override {
        config.numberOfInstances = instanceCount;
        return Result::SUCCESS;
    }

    void addTransferBuffer(const std::shared_ptr<Render::Buffer>& buffer) {
        transferBuffers.push_back(buffer);
    }
};

class TestTransfer final : public Render::Transfer {
 public:
    static constexpr U64 minimumBufferSize() { return MinimumBufferSize; }

    static bool reserve(U64& used, const U64 size, const U64 alignment, U64& offset) {
        return reserveRange(used, size, alignment, offset);
    }

    static bool capacity(const U64 required, const U64 alignment, U64& result) {
        return calculateCapacity(required, alignment, result);
    }

    static bool alignedSize(const U64 size, const U64 alignment, U64& result) {
        return calculateAlignedSize(size, alignment, result);
    }

    static bool copy(U8* destination, const U8* source, const U64 size) {
        return copyBuffer(destination, source, size);
    }

    static bool copyRows(U8* destination,
                         const U8* source,
                         const U64 rowCount,
                         const U64 rowByteSize,
                         const U64 encodedRowByteSize) {
        return copyTextureRows(destination,
                               source,
                               rowCount,
                               rowByteSize,
                               encodedRowByteSize);
    }

    static bool validTexture(const TextureTransfer& transfer, U64& rowCount) {
        return validateTexture(transfer, rowCount);
    }
};

class TestKernel final : public Render::Kernel {
 public:
    explicit TestKernel(const Config& config) : Kernel(config) {}

    bool wantsDispatch() const { return shouldEncode(); }
    void scheduled() { markScheduled(); }
};

struct CreationState {
    virtual ~CreationState() = default;

    Result create() {
        ++creates;
        live = true;
        return creationResult;
    }
    Result destroy() {
        ++destroys;
        live = false;
        return Result::SUCCESS;
    }

    Result creationResult = Result::SUCCESS;
    U64 creates = 0;
    U64 destroys = 0;
    bool live = false;
};

class CreationProgram final : public Render::Program, public CreationState {
 public:
    explicit CreationProgram(const Config& config = {}) : Program(config) {}
};

class CreationKernel final : public Render::Kernel, public CreationState {
 public:
    explicit CreationKernel(const Config& config = {}) : Kernel(config) {}
};

class TestSurface final : public Render::Surface {
 public:
    explicit TestSurface(const Config& config) : Surface(config),
                                                 backendPrograms(config.programs),
                                                 backendKernels(config.kernels) {}

    Result create() override {
        backendPrograms = config.programs;
        backendKernels = config.kernels;
        resetFramebuffer();
        return creationHandler ? creationHandler() : Result::SUCCESS;
    }
    Result destroy() override {
        destructionStarted.store(true, std::memory_order_release);
        while (destructionBlocked.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        if (destructionHandler) {
            JST_CHECK(destructionHandler());
        }
        for (const auto& kernel : backendKernels) {
            destroyObject(kernel);
        }
        for (const auto& program : backendPrograms) {
            destroyObject(program);
        }
        backendKernels.clear();
        backendPrograms.clear();
        return Result::SUCCESS;
    }
    const Extent2D<U64>& size(const Extent2D<U64>&) override { return NullSize2D; }

    bool wantsDraw(const bool framebufferChanged = false) {
        return shouldDraw(framebufferChanged);
    }
    bool wantsGraphics(const bool visible) { return shouldDrawGraphics(visible); }
    Result bindingResult() const { return drawResult(); }
    bool idleWhileHidden(const bool visible, const bool changed = false) const {
        return hiddenAndIdle(visible, changed);
    }
    void recreateFramebuffer() { resetFramebuffer(); }
    void blockDestruction() { destructionBlocked.store(true, std::memory_order_release); }
    void resumeDestruction() { destructionBlocked.store(false, std::memory_order_release); }
    bool hasStartedDestruction() const {
        return destructionStarted.load(std::memory_order_acquire);
    }

    bool acceptsAppends = false;
    U64 appendCalls = 0;
    U64 firstAppendedProgram = 0;
    U64 firstAppendedKernel = 0;
    U64 reorderCalls = 0;
    Result reorderResult = Result::SUCCESS;
    bool typedAppends = false;
    std::function<Result()> creationHandler;
    std::function<Result()> destructionHandler;
    std::vector<std::shared_ptr<Render::Program>> backendPrograms;
    std::vector<std::shared_ptr<Render::Kernel>> backendKernels;

 protected:
    Result reorderPrograms(const std::vector<std::shared_ptr<Render::Program>>& ordered) override {
        ++reorderCalls;
        if (reorderResult != Result::SUCCESS) {
            return std::exchange(reorderResult, Result::SUCCESS);
        }
        backendPrograms = ordered;
        return Result::SUCCESS;
    }
    bool appendable() const override { return acceptsAppends; }
    Result validateAppend(const std::vector<std::shared_ptr<Render::Program>>& programs,
                          const std::vector<std::shared_ptr<Render::Kernel>>& kernels) const override {
        JST_CHECK(Surface::validateAppend(programs, kernels));
        if (typedAppends) {
            for (const auto& program : programs) {
                if (!std::dynamic_pointer_cast<CreationProgram>(program)) {
                    return Result::ERROR;
                }
            }
            for (const auto& kernel : kernels) {
                if (!std::dynamic_pointer_cast<CreationKernel>(kernel)) {
                    return Result::ERROR;
                }
            }
        }
        return Result::SUCCESS;
    }
    Result createAppended(const std::vector<std::shared_ptr<Render::Program>>& programs,
                          const std::vector<std::shared_ptr<Render::Kernel>>& kernels) override {
        ++appendCalls;
        firstAppendedProgram = backendPrograms.size();
        firstAppendedKernel = backendKernels.size();
        return appendObjects(programs, kernels, backendPrograms, backendKernels, [](const auto& object) {
            auto state = std::dynamic_pointer_cast<CreationState>(object);
            return state ? state->create() : Result::SUCCESS;
        }, [](const auto& object) { return destroyObject(object); });
    }

 private:
    static Result destroyObject(const auto& object) {
        auto state = std::dynamic_pointer_cast<CreationState>(object);
        return state ? state->destroy() : Result::SUCCESS;
    }

    std::atomic<bool> destructionBlocked = false;
    std::atomic<bool> destructionStarted = false;
};

class TestWindow final : public Render::Window {
 public:
    TestWindow() : Window(Config{}) {}

    constexpr DeviceType device() const override { return DeviceType::None; }
    const Stats& stats() const override { return statsData; }
    std::string info() const override { return "Test"; }

    Result collectFrameTransfers(Render::Transfer::Batch& batch) const {
        return collectTransfers(batch);
    }

    void nextSubmissionResult(const Result result) {
        submissionResult = result;
    }

    void nextCreationResult(const Result result) {
        creationResult = result;
    }

    void nextDestructionResult(const Result result) {
        destructionResult = result;
    }

    U64 destructionCount() const {
        return destructions;
    }

    std::function<Result()> synchronizeHandler;
    std::function<Result(const std::shared_ptr<Render::Surface>&)> bindHandler;
    std::function<Result(const std::shared_ptr<Render::Surface>&)> unbindHandler;
    U64 outstandingSubmissions = 0;
    U64 registeredSurfaceCount() const { return surfaces.size(); }

    void attachViewport(Viewport::Generic* viewport) {
        attachedViewport = viewport;
        updateScalingFactor(*viewport);
    }

 protected:
    Viewport::Generic* underlyingViewport() const override { return attachedViewport; }

    Result bindSurface(const std::shared_ptr<Render::Surface>& surface) override {
        surfaces.push_back(std::dynamic_pointer_cast<TestSurface>(surface));
        return bindHandler ? bindHandler(surface) : Result::SUCCESS;
    }
    Result unbindSurface(const std::shared_ptr<Render::Surface>& surface) override {
        if (unbindHandler) {
            JST_CHECK(unbindHandler(surface));
        }
        std::erase(surfaces, std::dynamic_pointer_cast<TestSurface>(surface));
        return Result::SUCCESS;
    }
    Result underlyingCreate() override { return creationResult; }
    Result underlyingDestroy() override {
        ++destructions;
        return destructionResult;
    }
    Result underlyingBegin() override { return Result::SUCCESS; }
    Result underlyingEnd() override {
        Render::Transfer::Batch batch;
        JST_CHECK(collectTransfers(batch));

        for (const auto& surface : surfaces) {
            surface->wantsDraw();
            JST_CHECK(surface->bindingResult());
        }

        if (submissionResult != Result::SUCCESS) {
            return submissionResult;
        }

        for (const auto& surface : surfaces) {
            surface->commitDraw();
        }
        batch.commit();
        ++outstandingSubmissions;
        return Result::SUCCESS;
    }
    Result underlyingSynchronize() override {
        const Result result = synchronizeHandler ? synchronizeHandler() : Result::SUCCESS;
        if (result == Result::SUCCESS || result == Result::RELOAD) {
            outstandingSubmissions = 0;
        }
        return result;
    }

 private:
    Stats statsData{};
    Result creationResult = Result::SUCCESS;
    Result destructionResult = Result::SUCCESS;
    Result submissionResult = Result::SUCCESS;
    U64 destructions = 0;
    std::vector<std::shared_ptr<TestSurface>> surfaces;
    Viewport::Generic* attachedViewport = nullptr;
};

class TestViewport final : public Viewport::Generic {
 public:
    explicit TestViewport(const Viewport::Config& config = {}) : Generic(config) {}

    std::string id() const override { return "test"; }
    std::string name() const override { return "Test"; }
    DeviceType device() const override { return DeviceType::None; }

    Result create() override { return Result::SUCCESS; }
    Result destroy() override { return Result::SUCCESS; }
    Result createImgui() override { return Result::SUCCESS; }
    Result destroyImgui() override { return Result::SUCCESS; }

    F32 scale(const F32&) const override { return 2.0f; }

    Result waitEvents() override { return Result::SUCCESS; }
    Result pollEvents() override { return Result::SUCCESS; }
    bool keepRunning() override { return true; }

    Render::TitlebarChrome titlebarChrome() const override {
        return {.integrated = true, .leadingInset = 80.0f, .height = 32.0f};
    }

    void beginWindowDrag() override { ++drags; }
    void zoomWindow() override { ++zooms; }
    void setIntegratedTitlebar(bool enabled) override {
        activations.push_back(enabled);
        Generic::setIntegratedTitlebar(enabled);
    }

    U64 drags = 0;
    U64 zooms = 0;
    std::vector<bool> activations;
};

std::shared_ptr<TestBuffer> MakeBuffer(std::array<U32, 2>& source) {
    return std::make_shared<TestBuffer>(Render::Buffer::Config{
        .size = source.size(),
        .target = Render::Buffer::Target::STORAGE,
        .elementByteSize = sizeof(U32),
        .buffer = source.data(),
    });
}

std::shared_ptr<TestSurface> MakeRetainedSurface(
    const std::shared_ptr<Render::Buffer>& buffer) {
    Render::Kernel::Config kernelConfig;
    kernelConfig.buffers.push_back({buffer, Render::Kernel::AccessMode::READ});

    Render::Surface::Config config;
    config.kernels.push_back(std::make_shared<Render::Kernel>(kernelConfig));
    config.retained = true;
    return std::make_shared<TestSurface>(config);
}

std::shared_ptr<TestSurface> MakeRetainedSurface(
    const std::shared_ptr<Render::Texture>& texture) {
    Render::Program::Config programConfig;
    programConfig.textures.push_back(texture);

    Render::Surface::Config config;
    config.programs.push_back(std::make_shared<Render::Program>(programConfig));
    config.retained = true;
    return std::make_shared<TestSurface>(config);
}

std::shared_ptr<TestSurface> MakeFramebufferSurface(const std::shared_ptr<Render::Texture>& texture) {
    Render::Surface::Config config;
    config.framebuffer = texture;
    config.retained = true;
    return std::make_shared<TestSurface>(config);
}

std::shared_ptr<TestSurface> MakeBorrowedSurface(const std::shared_ptr<Render::Texture>& texture) {
    Render::Program::Config programConfig;
    programConfig.textures.push_back(texture);
    programConfig.borrowedTextures = true;
    Render::Surface::Config config;
    config.programs.push_back(std::make_shared<RebindingProgram>(programConfig));
    config.retained = true;
    return std::make_shared<TestSurface>(config);
}

std::shared_ptr<TestSurface> MakeRetainedSurface(const std::shared_ptr<Render::Buffer>& buffer,
                                                 const std::shared_ptr<Render::Texture>& texture) {
    Render::Kernel::Config kernelConfig;
    kernelConfig.buffers.push_back({buffer, Render::Kernel::AccessMode::READ});

    Render::Program::Config programConfig;
    programConfig.textures.push_back(texture);

    Render::Surface::Config config;
    config.kernels.push_back(std::make_shared<Render::Kernel>(kernelConfig));
    config.programs.push_back(std::make_shared<Render::Program>(programConfig));
    config.retained = true;
    return std::make_shared<TestSurface>(config);
}

}  // namespace

TEST_CASE("Surface reordering preserves the exact program multiset and invalidates only changes",
          "[core][render][surface][reorder]") {
    auto first = std::make_shared<Render::Program>(Render::Program::Config{});
    auto second = std::make_shared<Render::Program>(Render::Program::Config{});
    Render::Surface::Config config;
    config.programs = {first, second, first};
    config.retained = true;
    TestSurface surface(config);
    REQUIRE(surface.wantsDraw());
    surface.commitDraw();
    REQUIRE_FALSE(surface.wantsDraw());

    REQUIRE(surface.reorder({first, second, first}) == Result::SUCCESS);
    REQUIRE(surface.reorderCalls == 0);
    REQUIRE_FALSE(surface.wantsDraw());
    REQUIRE(surface.reorder({first, second}) == Result::ERROR);
    REQUIRE(surface.reorder({first, second, second}) == Result::ERROR);
    REQUIRE(surface.getConfig().programs == config.programs);
    REQUIRE(surface.reorderCalls == 0);
    REQUIRE_FALSE(surface.wantsDraw());

    const std::vector<std::shared_ptr<Render::Program>> reordered = {second, first, first};
    REQUIRE(surface.reorder(reordered) == Result::SUCCESS);
    REQUIRE(surface.getConfig().programs == reordered);
    REQUIRE(surface.reorderCalls == 1);
    REQUIRE(surface.wantsDraw());
    surface.commitDraw();
    REQUIRE_FALSE(surface.wantsDraw());
}

TEST_CASE("Surface extension appends programs and kernels without recreating shared resources",
          "[core][render][surface][extend]") {
    TestWindow window;
    std::array<U32, 2> sharedSource = {1, 2};
    std::array<U32, 2> appendedSource = {3, 4};
    auto sharedBuffer = MakeBuffer(sharedSource);
    auto appendedBuffer = MakeBuffer(appendedSource);
    auto texture = std::make_shared<TestTexture>(Render::Texture::Config{.size = {2, 2}});
    auto surface = MakeRetainedSurface(sharedBuffer, texture);
    auto sibling = MakeRetainedSurface(sharedBuffer, texture);
    surface->acceptsAppends = true;
    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.bind(surface) == Result::SUCCESS);
    REQUIRE(window.bind(sibling) == Result::SUCCESS);
    REQUIRE(sharedBuffer->createCount == 1);
    REQUIRE(texture->createCount == 1);
    REQUIRE(surface->wantsDraw());
    surface->commitDraw();
    REQUIRE_FALSE(surface->wantsDraw());

    Render::Program::Config programConfig;
    programConfig.buffers = {{sharedBuffer, Render::Program::Target::VERTEX},
                             {appendedBuffer, Render::Program::Target::FRAGMENT}};
    programConfig.textures = {texture};
    auto program = std::make_shared<Render::Program>(programConfig);
    Render::Kernel::Config kernelConfig;
    kernelConfig.buffers = {{appendedBuffer, Render::Kernel::AccessMode::READ}};
    auto kernel = std::make_shared<TestKernel>(kernelConfig);
    REQUIRE(window.extend(surface, {program}, {kernel}) == Result::SUCCESS);
    REQUIRE(surface->appendCalls == 1);
    REQUIRE(surface->firstAppendedProgram == 1);
    REQUIRE(surface->firstAppendedKernel == 1);
    REQUIRE(surface->getConfig().programs.size() == 2);
    REQUIRE(surface->getConfig().programs.back() == program);
    REQUIRE(surface->getConfig().kernels.size() == 2);
    REQUIRE(surface->getConfig().kernels.back() == kernel);
    REQUIRE(surface->wantsDraw());
    REQUIRE(sharedBuffer->createCount == 1);
    REQUIRE(appendedBuffer->createCount == 1);
    REQUIRE(texture->createCount == 1);

    REQUIRE(appendedBuffer->update() == Result::SUCCESS);
    Render::Transfer::Batch batch;
    REQUIRE(window.collectFrameTransfers(batch) == Result::SUCCESS);
    REQUIRE(batch.contains(appendedBuffer));
    batch.commit();
    REQUIRE(window.unbind(surface) == Result::SUCCESS);
    REQUIRE(sharedBuffer->destroyCount == 0);
    REQUIRE(texture->destroyCount == 0);
    REQUIRE(window.destroy() == Result::SUCCESS);
    REQUIRE(sharedBuffer->destroyCount == 1);
    REQUIRE(appendedBuffer->destroyCount == 1);
    REQUIRE(texture->destroyCount == 1);
}

TEST_CASE("Surface extension preserves borrowed leases and falls back for ownership changes",
          "[core][render][surface][extend][borrowed]") {
    TestWindow window;
    auto texture = std::make_shared<TestTexture>(Render::Texture::Config{.size = {2, 2}});
    auto unboundTexture = std::make_shared<TestTexture>(Render::Texture::Config{.size = {2, 2}});
    auto producer = MakeFramebufferSurface(texture);
    auto consumer = MakeBorrowedSurface(texture);
    consumer->acceptsAppends = true;
    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.bind(producer) == Result::SUCCESS);
    REQUIRE(window.bind(consumer) == Result::SUCCESS);

    Render::Program::Config config;
    config.borrowedTextures = true;
    config.textures = {unboundTexture};
    REQUIRE(window.extend(consumer, {std::make_shared<Render::Program>(config)}, {}) == Result::SKIP);
    REQUIRE(unboundTexture->createCount == 0);
    config.borrowedTextures = false;
    config.textures = {texture};
    REQUIRE(window.extend(consumer, {std::make_shared<Render::Program>(config)}, {}) == Result::SKIP);
    REQUIRE(consumer->appendCalls == 0);
    REQUIRE(consumer->getConfig().programs.size() == 1);

    config.borrowedTextures = true;
    REQUIRE(window.extend(consumer, {std::make_shared<Render::Program>(config)}, {}) == Result::SUCCESS);
    REQUIRE(consumer->appendCalls == 1);
    REQUIRE(consumer->getConfig().programs.size() == 2);
    REQUIRE(texture->createCount == 1);
    REQUIRE(window.destroy() == Result::SUCCESS);
    REQUIRE(texture->destroyCount == 1);
}

TEST_CASE("Surface extension rejects null surfaces and leaves unsupported or unbound surfaces unchanged",
          "[core][render][surface][extend]") {
    TestWindow window;
    auto surface = std::make_shared<TestSurface>(Render::Surface::Config{});
    auto program = std::make_shared<Render::Program>(Render::Program::Config{});
    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.extend(nullptr, {program}, {}) == Result::ERROR);
    surface->acceptsAppends = true;
    REQUIRE(window.extend(surface, {program}, {}) == Result::SKIP);
    surface->acceptsAppends = false;
    REQUIRE(window.bind(surface) == Result::SUCCESS);
    REQUIRE(window.extend(surface, {program}, {}) == Result::SKIP);
    REQUIRE(surface->appendCalls == 0);
    REQUIRE(surface->getConfig().programs.empty());
    REQUIRE(window.destroy() == Result::SUCCESS);
}

TEST_CASE("Failed surface reorders preserve configuration and backend order until retry succeeds",
          "[core][render][surface][reorder]") {
    Result failure = Result::ERROR;
    SECTION("Error") {}
    SECTION("Skip") { failure = Result::SKIP; }
    auto first = std::make_shared<Render::Program>(Render::Program::Config{});
    auto second = std::make_shared<Render::Program>(Render::Program::Config{});
    Render::Surface::Config config;
    config.programs = {first, second};
    config.retained = true;
    TestSurface surface(config);
    REQUIRE(surface.wantsDraw());
    surface.commitDraw();
    REQUIRE_FALSE(surface.wantsDraw());
    surface.reorderResult = failure;
    REQUIRE(surface.reorder({second, first}) == failure);
    REQUIRE(surface.getConfig().programs == config.programs);
    REQUIRE(surface.backendPrograms == config.programs);
    REQUIRE(surface.reorderCalls == 1);
    REQUIRE_FALSE(surface.wantsDraw());
    REQUIRE(surface.reorder({second, first}) == Result::SUCCESS);
    REQUIRE(surface.reorderCalls == 2);
    REQUIRE(surface.getConfig().programs == std::vector<std::shared_ptr<Render::Program>>{second, first});
    REQUIRE(surface.backendPrograms == surface.getConfig().programs);
    REQUIRE(surface.wantsDraw());
}

TEST_CASE("Failed surface appends roll back partial backend creation dependencies and leases",
          "[core][render][surface][extend]") {
    U64 failedObject = 0;
    SECTION("First program") {}
    SECTION("Later program") { failedObject = 1; }
    SECTION("First kernel") { failedObject = 2; }
    SECTION("Later kernel") { failedObject = 3; }

    TestWindow window;
    std::array<U32, 2> source = {1, 2};
    auto buffer = MakeBuffer(source);
    auto texture = std::make_shared<TestTexture>(Render::Texture::Config{.size = {2, 2}});
    auto surface = std::make_shared<TestSurface>(Render::Surface::Config{.retained = true});
    surface->acceptsAppends = true;
    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.bind(surface) == Result::SUCCESS);
    auto original = std::make_shared<CreationProgram>();
    REQUIRE(window.extend(surface, {original}, {}) == Result::SUCCESS);
    REQUIRE(original->creates == 1);
    REQUIRE(surface->wantsDraw());
    surface->commitDraw();
    REQUIRE_FALSE(surface->wantsDraw());
    const auto originalConfig = surface->getConfig();

    Render::Program::Config programConfig;
    programConfig.buffers = {{buffer, Render::Program::Target::VERTEX}};
    programConfig.textures = {texture};
    auto firstProgram = std::make_shared<CreationProgram>(programConfig);
    auto secondProgram = std::make_shared<CreationProgram>(programConfig);
    Render::Kernel::Config kernelConfig;
    kernelConfig.buffers = {{buffer, Render::Kernel::AccessMode::READ}};
    auto firstKernel = std::make_shared<CreationKernel>(kernelConfig);
    auto secondKernel = std::make_shared<CreationKernel>(kernelConfig);
    std::array<CreationState*, 4> objects = {firstProgram.get(), secondProgram.get(), firstKernel.get(), secondKernel.get()};

    for (const Result failure : {Result::ERROR, Result::SKIP}) {
        const U64 attemptsBefore = objects[failedObject]->creates;
        objects[failedObject]->creationResult = failure;
        REQUIRE(window.extend(surface, {firstProgram, secondProgram}, {firstKernel, secondKernel}) == failure);
        REQUIRE(surface->getConfig().programs == originalConfig.programs);
        REQUIRE(surface->getConfig().kernels == originalConfig.kernels);
        REQUIRE(surface->backendPrograms == originalConfig.programs);
        REQUIRE(surface->backendKernels == originalConfig.kernels);
        REQUIRE_FALSE(surface->wantsDraw());
        REQUIRE(original->creates == 1);
        REQUIRE(original->destroys == 0);
        REQUIRE(original->live);
        for (U64 i = 0; i < objects.size(); ++i) {
            REQUIRE_FALSE(objects[i]->live);
            REQUIRE(objects[i]->creates == objects[i]->destroys);
            REQUIRE(objects[i]->creates == (i <= failedObject ? attemptsBefore + 1 : 0));
        }
        REQUIRE(buffer->createCount == buffer->destroyCount);
        REQUIRE(texture->createCount == texture->destroyCount);
        REQUIRE(buffer->update() == Result::SUCCESS);
        Render::Transfer::Batch batch;
        REQUIRE(window.collectFrameTransfers(batch) == Result::SUCCESS);
        REQUIRE_FALSE(batch.contains(buffer));
        REQUIRE_FALSE(batch.contains(texture));
        REQUIRE_FALSE(surface->wantsDraw());
    }

    objects[failedObject]->creationResult = Result::SUCCESS;
    REQUIRE(window.extend(surface, {firstProgram, secondProgram}, {firstKernel, secondKernel}) == Result::SUCCESS);
    REQUIRE(surface->getConfig().programs.size() == 3);
    REQUIRE(surface->getConfig().kernels.size() == 2);
    REQUIRE(surface->backendPrograms == surface->getConfig().programs);
    REQUIRE(surface->backendKernels == surface->getConfig().kernels);
    REQUIRE(surface->wantsDraw());
    REQUIRE(buffer->createCount == 3);
    REQUIRE(texture->createCount == 3);
    REQUIRE(window.destroy() == Result::SUCCESS);
    REQUIRE(buffer->destroyCount == 3);
    REQUIRE(texture->destroyCount == 3);
    REQUIRE(original->destroys == 1);
    for (auto* object : objects) {
        REQUIRE_FALSE(object->live);
        REQUIRE(object->creates == object->destroys);
    }
}

TEST_CASE("Throwing surface acquisitions release claims partial allocations and earlier leases",
          "[core][render][surface][extend][exceptions]") {
    bool initialBind = false;
    U64 failure = 1;
    bool discard = false;
    SECTION("Append first dependency") {}
    SECTION("Append later dependency") { failure = 2; }
    SECTION("Initial bind first dependency") { initialBind = true; }
    SECTION("Initial bind later dependency") { initialBind = true; failure = 2; }
    SECTION("Discard rejected append") { discard = true; failure = 2; }
    SECTION("Discard rejected initial bind") { initialBind = true; discard = true; failure = 2; }

    TestWindow window;
    std::array<U32, 2> source = {1, 2};
    auto liveBuffer = MakeBuffer(source);
    auto first = MakeBuffer(source);
    auto second = MakeBuffer(source);
    auto liveSurface = MakeRetainedSurface(liveBuffer);
    liveSurface->acceptsAppends = true;
    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.bind(liveSurface) == Result::SUCCESS);
    REQUIRE(liveSurface->wantsDraw());
    liveSurface->commitDraw();
    const auto before = liveSurface->getConfig();
    U64 attempts = 0;
    first->creationHandler = second->creationHandler = [&]() {
        if (++attempts == failure) {
            throw std::runtime_error("dependency creation");
        }
        return Result::SUCCESS;
    };
    Render::Program::Config programConfig;
    programConfig.buffers = {{liveBuffer, Render::Program::Target::VERTEX},
                             {first, Render::Program::Target::VERTEX},
                             {second, Render::Program::Target::FRAGMENT}};
    auto program = std::make_shared<CreationProgram>(programConfig);
    auto kernel = std::make_shared<CreationKernel>();
    auto target = liveSurface;
    if (initialBind) {
        target = std::make_shared<TestSurface>(Render::Surface::Config{
            .kernels = {kernel}, .programs = {program}, .retained = true,
        });
    }
    const auto request = [&]() {
        return initialBind ? window.bind(target) : window.extend(target, {program}, {kernel});
    };
    REQUIRE_THROWS_AS(request(), std::runtime_error);
    REQUIRE(attempts == failure);
    REQUIRE(program->creates == 0);
    REQUIRE(kernel->creates == 0);
    REQUIRE(first->createCount == first->destroyCount);
    REQUIRE(second->createCount == second->destroyCount);
    REQUIRE_FALSE(first->live);
    REQUIRE_FALSE(second->live);
    REQUIRE(liveBuffer->createCount == 1);
    REQUIRE(liveBuffer->destroyCount == 0);
    REQUIRE(liveBuffer->live);
    REQUIRE(liveSurface->getConfig().programs == before.programs);
    REQUIRE(liveSurface->getConfig().kernels == before.kernels);
    REQUIRE_FALSE(liveSurface->wantsDraw());
    first->creationHandler = second->creationHandler = {};
    if (discard) {
        std::weak_ptr<Render::Program> lifetime = program;
        target.reset();
        program.reset();
        kernel.reset();
        REQUIRE(lifetime.expired());
    } else {
        REQUIRE(request() == Result::SUCCESS);
        REQUIRE(first->live);
        REQUIRE(second->live);
        REQUIRE(target->getConfig().programs.back() == program);
        REQUIRE(target->getConfig().kernels.back() == kernel);
    }
    REQUIRE(window.destroy() == Result::SUCCESS);
    REQUIRE(first->createCount == first->destroyCount);
    REQUIRE(second->createCount == second->destroyCount);
    REQUIRE(liveBuffer->destroyCount == 1);
    REQUIRE_FALSE(liveBuffer->live);
}

TEST_CASE("Rejected initial surfaces retain retryable cleanup and release claims while remaining alive",
          "[core][render][surface][extend][exceptions][registration]") {
    bool registration = false;
    bool throws = true;
    bool cleanupFails = false;
    bool cleanupThrows = false;
    bool unbindFails = false;
    bool retryBind = false;
    SECTION("Initial creation throws") {}
    SECTION("Initial creation returns error") { throws = false; }
    SECTION("Registration throws after insertion") { registration = true; }
    SECTION("Registration returns error after insertion") { registration = true; throws = false; }
    SECTION("Initial cleanup returns error") { cleanupFails = true; }
    SECTION("Initial cleanup throws") { cleanupFails = true; cleanupThrows = true; }
    SECTION("Registration unbind returns error") { registration = true; cleanupFails = true; unbindFails = true; }
    SECTION("Registration unbind throws") { registration = true; cleanupFails = true; cleanupThrows = true; unbindFails = true; }
    SECTION("Retry same rejected surface") { retryBind = true; }

    TestWindow window;
    std::array<U32, 2> source = {1, 2};
    auto buffer = MakeBuffer(source);
    auto texture = std::make_shared<TestTexture>(Render::Texture::Config{.size = {2, 2}});
    auto vertex = std::make_shared<Render::Vertex>(Render::Vertex::Config{});
    auto draw = std::make_shared<TestDraw>(Render::Draw::Config{.buffer = vertex});
    Render::Program::Config programConfig;
    programConfig.buffers = {{buffer, Render::Program::Target::VERTEX}};
    programConfig.textures = {texture};
    programConfig.draws = {draw};
    auto program = std::make_shared<CreationProgram>(programConfig);
    auto kernel = std::make_shared<CreationKernel>();
    auto failed = std::make_shared<TestSurface>(Render::Surface::Config{
        .kernels = {kernel}, .programs = {program}, .retained = true,
    });
    auto target = std::make_shared<TestSurface>(Render::Surface::Config{.retained = true});
    target->acceptsAppends = true;
    const auto failure = [&]() -> Result {
        if (throws) throw std::runtime_error("initial surface");
        return Result::ERROR;
    };
    failed->creationHandler = [&]() {
        JST_CHECK(program->create());
        JST_CHECK(kernel->create());
        return registration ? Result::SUCCESS : failure();
    };
    if (registration) {
        window.bindHandler = [&](const auto& surface) {
            return surface == failed ? failure() : Result::SUCCESS;
        };
    }
    const auto cleanup = [&]() -> Result {
        if (!cleanupFails) return Result::SUCCESS;
        if (cleanupThrows) throw std::runtime_error("initial cleanup");
        return Result::ERROR;
    };
    if (unbindFails) {
        window.unbindHandler = [&](const auto& surface) {
            return surface == failed ? cleanup() : Result::SUCCESS;
        };
    } else {
        failed->destructionHandler = cleanup;
    }
    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.bind(target) == Result::SUCCESS);
    if (throws) REQUIRE_THROWS_AS(window.bind(failed), std::runtime_error);
    else REQUIRE(window.bind(failed) == Result::ERROR);
    REQUIRE(failed->getConfig().programs.front() == program);
    REQUIRE(failed->getConfig().kernels.front() == kernel);
    REQUIRE(buffer->createCount == 1);
    REQUIRE(texture->createCount == 1);
    if (cleanupFails) {
        REQUIRE(program->live);
        REQUIRE(kernel->live);
        REQUIRE(buffer->destroyCount == 0);
        REQUIRE(texture->allocated());
        REQUIRE(window.extend(target, {program}, {kernel}) == Result::ERROR);
        REQUIRE(window.bind(failed) == Result::ERROR);
        REQUIRE(window.start() == Result::SUCCESS);
        if (unbindFails) {
            REQUIRE(window.begin() == Result::ERROR);
            REQUIRE(window.registeredSurfaceCount() == 2);
            REQUIRE(program->live);
            REQUIRE(texture->allocated());
        }
        cleanupFails = false;
        REQUIRE(window.begin() == Result::SUCCESS);
        REQUIRE(window.end() == Result::SUCCESS);
        REQUIRE(window.stop() == Result::SUCCESS);
    }
    REQUIRE_FALSE(program->live);
    REQUIRE_FALSE(kernel->live);
    REQUIRE(buffer->destroyCount == 1);
    REQUIRE(texture->destroyCount == 1);
    REQUIRE_FALSE(texture->allocated());
    REQUIRE(window.registeredSurfaceCount() == 1);
    window.bindHandler = {};
    window.unbindHandler = {};
    failed->destructionHandler = {};
    if (retryBind) {
        failed->creationHandler = [&]() {
            JST_CHECK(program->create());
            return kernel->create();
        };
        REQUIRE(window.bind(failed) == Result::SUCCESS);
    } else {
        REQUIRE(window.extend(target, {program}, {kernel}) == Result::SUCCESS);
    }
    REQUIRE(failed.use_count() >= 1);
    REQUIRE(program->live);
    REQUIRE(kernel->live);
    REQUIRE(program->creates == 2);
    REQUIRE(kernel->creates == 2);
    REQUIRE(buffer->createCount == 2);
    REQUIRE(texture->createCount == 2);
    REQUIRE(window.destroy() == Result::SUCCESS);
    REQUIRE(program->destroys == 2);
    REQUIRE(kernel->destroys == 2);
    REQUIRE(buffer->destroyCount == 2);
    REQUIRE(texture->destroyCount == 2);
}

TEST_CASE("Mixed queued root failures roll back every unregistered surface and preserve earlier registrations",
          "[core][render][surface][extend][exceptions][threading][queued-bind]") {
    bool textureRoot = false;
    bool returnsError = false;
    bool badAllocation = false;
    bool surfaceCleanupFails = false;
    bool surfaceCleanupThrows = false;
    bool rootCleanupFails = false;
    bool promoteShared = false;
    SECTION("Buffer creation throws") {}
    SECTION("Texture creation throws") { textureRoot = true; }
    SECTION("Buffer allocation throws") { badAllocation = true; }
    SECTION("Texture allocation throws") { textureRoot = true; badAllocation = true; }
    SECTION("Buffer creation returns error") { returnsError = true; }
    SECTION("Texture creation returns error") { textureRoot = true; returnsError = true; }
    SECTION("Surface cleanup fails") { surfaceCleanupFails = true; }
    SECTION("Surface cleanup throws") { surfaceCleanupFails = true; surfaceCleanupThrows = true; }
    SECTION("Root cleanup fails") { rootCleanupFails = true; }
    SECTION("Earlier shared resource becomes a root") { promoteShared = true; }

    TestWindow window;
    std::array<U32, 2> source = {1, 2};
    auto shared = MakeBuffer(source);
    Render::Program::Config originalConfig;
    originalConfig.buffers = {{shared, Render::Program::Target::VERTEX}};
    auto original = std::make_shared<CreationProgram>(originalConfig);
    auto target = std::make_shared<TestSurface>(Render::Surface::Config{.programs = {original}, .retained = true});
    target->creationHandler = [&]() { return original->create(); };
    target->acceptsAppends = true;
    std::array<std::shared_ptr<TestSurface>, 2> candidates;
    std::array<std::shared_ptr<CreationProgram>, 2> programs;
    std::array<std::shared_ptr<CreationKernel>, 2> kernels;
    std::array<std::shared_ptr<TestBuffer>, 2> buffers;
    std::array<std::shared_ptr<TestTexture>, 2> textures;
    for (U64 i = 0; i < candidates.size(); ++i) {
        buffers[i] = MakeBuffer(source);
        textures[i] = std::make_shared<TestTexture>(Render::Texture::Config{.size = {2, 2}});
        auto vertex = std::make_shared<Render::Vertex>(Render::Vertex::Config{.vertices = {{buffers[i], 2}}});
        auto draw = std::make_shared<TestDraw>(Render::Draw::Config{.buffer = vertex});
        Render::Program::Config config;
        config.draws = {draw};
        config.buffers = {{shared, Render::Program::Target::VERTEX}, {buffers[i], Render::Program::Target::VERTEX}};
        config.textures = {textures[i]};
        programs[i] = std::make_shared<CreationProgram>(config);
        kernels[i] = std::make_shared<CreationKernel>();
        candidates[i] = std::make_shared<TestSurface>(Render::Surface::Config{
            .kernels = {kernels[i]}, .programs = {programs[i]}, .retained = true,
        });
        candidates[i]->creationHandler = [&, i]() {
            JST_CHECK(programs[i]->create());
            return kernels[i]->create();
        };
    }
    candidates[0]->destructionHandler = [&]() -> Result {
        if (!surfaceCleanupFails) return Result::SUCCESS;
        if (surfaceCleanupThrows) throw std::runtime_error("surface cleanup");
        return Result::ERROR;
    };
    auto rootBuffer = MakeBuffer(source);
    auto rootTexture = std::make_shared<TestTexture>(Render::Texture::Config{.size = {2, 2}});
    std::shared_ptr<Render::WindowAttachment> root = textureRoot ?
        std::static_pointer_cast<Render::WindowAttachment>(rootTexture) : rootBuffer;
    const auto createRoot = [&]() -> Result {
        if (returnsError) return Result::ERROR;
        if (badAllocation) throw std::bad_alloc();
        throw std::runtime_error("queued root creation");
    };
    rootBuffer->creationHandler = rootTexture->creationHandler = createRoot;
    if (rootCleanupFails) rootBuffer->nextDestructionResult(Result::ERROR);
    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.bind(target) == Result::SUCCESS);
    REQUIRE(window.start() == Result::SUCCESS);
    REQUIRE(window.begin() == Result::SUCCESS);
    REQUIRE(window.end() == Result::SUCCESS);
    REQUIRE_FALSE(target->wantsDraw());
    std::array<Result, 4> results{};
    std::vector<std::thread> workers;
    workers.reserve(4);
    const auto enqueue = [&](const std::shared_ptr<Render::WindowAttachment>& attachment, U64 index) {
        const auto before = attachment.use_count();
        workers.emplace_back([&, attachment, index]() { results[index] = window.bind(attachment); });
        while (attachment.use_count() < before + 2) std::this_thread::yield();
    };
    enqueue(candidates[0], 0);
    enqueue(candidates[1], 1);
    if (promoteShared) enqueue(buffers[0], 2);
    enqueue(root, promoteShared ? 3 : 2);
    std::exception_ptr exception;
    Result beginResult = Result::ERROR;
    try {
        beginResult = window.begin();
    } catch (...) {
        exception = std::current_exception();
    }
    for (auto& worker : workers) worker.join();
    if (beginResult == Result::SUCCESS) REQUIRE(window.cancel() == Result::SUCCESS);
    REQUIRE(window.stop() == Result::SUCCESS);
    if (returnsError) {
        REQUIRE_FALSE(exception);
        REQUIRE(beginResult == Result::ERROR);
    } else {
        REQUIRE(exception);
        if (badAllocation) REQUIRE_THROWS_AS(std::rethrow_exception(exception), std::bad_alloc);
        else REQUIRE_THROWS_AS(std::rethrow_exception(exception), std::runtime_error);
    }
    for (U64 i = 0; i < workers.size(); ++i) REQUIRE(results[i] == Result::SUCCESS);
    REQUIRE(window.registeredSurfaceCount() == 1);
    REQUIRE(original->live);
    REQUIRE(original->creates == 1);
    REQUIRE(original->destroys == 0);
    REQUIRE(shared->createCount == 1);
    REQUIRE(shared->destroyCount == 0);
    if (promoteShared) REQUIRE(window.bind(buffers[0]) == Result::SUCCESS);
    REQUIRE_FALSE(target->wantsDraw());
    if (rootCleanupFails) {
        Render::Program::Config config;
        config.buffers = {{rootBuffer, Render::Program::Target::VERTEX}};
        auto consumer = std::make_shared<CreationProgram>(config);
        REQUIRE(window.extend(target, {consumer}, {}) == Result::ERROR);
        REQUIRE(consumer->creates == 0);
        REQUIRE(rootBuffer->createCount == 1);
    }
    for (U64 i = 0; i < candidates.size(); ++i) {
        REQUIRE(candidates[i]->getConfig().programs.front() == programs[i]);
        REQUIRE(programs[i]->creates == 1);
        REQUIRE(kernels[i]->creates == 1);
        if (i == 0 && surfaceCleanupFails) {
            REQUIRE(programs[i]->live);
            REQUIRE(textures[i]->allocated());
            REQUIRE(window.extend(target, {programs[i]}, {kernels[i]}) == Result::ERROR);
        } else {
            REQUIRE_FALSE(programs[i]->live);
            REQUIRE(programs[i]->destroys == 1);
            REQUIRE(kernels[i]->destroys == 1);
            REQUIRE(buffers[i]->destroyCount == (promoteShared && i == 0 ? 0 : 1));
            REQUIRE_FALSE(textures[i]->allocated());
        }
    }
    if (surfaceCleanupFails || rootCleanupFails) {
        surfaceCleanupFails = false;
        rootBuffer->nextDestructionResult(Result::SUCCESS);
        REQUIRE(window.start() == Result::SUCCESS);
        REQUIRE(window.begin() == Result::SUCCESS);
        REQUIRE(window.end() == Result::SUCCESS);
        REQUIRE(window.stop() == Result::SUCCESS);
    }
    REQUIRE(window.extend(target, {programs[0], programs[1]}, {kernels[0], kernels[1]}) == Result::SUCCESS);
    rootBuffer->creationHandler = rootTexture->creationHandler = {};
    REQUIRE(window.bind(root) == Result::SUCCESS);
    REQUIRE(window.destroy() == Result::SUCCESS);
    REQUIRE(original->destroys == 1);
    REQUIRE(shared->destroyCount == 1);
    for (U64 i = 0; i < programs.size(); ++i) {
        REQUIRE(programs[i]->creates == 2);
        REQUIRE(programs[i]->destroys == 2);
        REQUIRE(kernels[i]->creates == 2);
        REQUIRE(kernels[i]->destroys == 2);
        REQUIRE(buffers[i]->createCount == (promoteShared && i == 0 ? 1 : 2));
        REQUIRE(buffers[i]->destroyCount == (promoteShared && i == 0 ? 1 : 2));
        REQUIRE(textures[i]->createCount == 2);
        REQUIRE(textures[i]->destroyCount == 2);
    }
    if (textureRoot) {
        REQUIRE(rootTexture->createCount == 2);
        REQUIRE(rootTexture->destroyCount == 2);
    } else {
        REQUIRE(rootBuffer->createCount == 2);
        REQUIRE(rootBuffer->destroyCount == (rootCleanupFails ? 3 : 2));
    }
}

TEST_CASE("Mixed binding registration rollback never revisits an already registered surface or root",
          "[core][render][surface][extend][exceptions][threading][queued-bind]") {
    U64 failedIndex = 1;
    bool throws = true;
    SECTION("Later registration throws") {}
    SECTION("First registration throws") { failedIndex = 0; }
    SECTION("Later registration returns error") { throws = false; }
    TestWindow window;
    auto target = std::make_shared<TestSurface>(Render::Surface::Config{.retained = true});
    target->acceptsAppends = true;
    std::array<std::shared_ptr<CreationProgram>, 2> programs;
    std::array<std::shared_ptr<TestSurface>, 2> candidates;
    for (U64 i = 0; i < candidates.size(); ++i) {
        programs[i] = std::make_shared<CreationProgram>();
        candidates[i] = std::make_shared<TestSurface>(Render::Surface::Config{.programs = {programs[i]}});
        candidates[i]->creationHandler = [&, i]() { return programs[i]->create(); };
    }
    std::array<U32, 2> source = {1, 2};
    auto root = MakeBuffer(source);
    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.bind(target) == Result::SUCCESS);
    REQUIRE(window.start() == Result::SUCCESS);
    REQUIRE(window.begin() == Result::SUCCESS);
    REQUIRE(window.end() == Result::SUCCESS);
    window.bindHandler = [&](const auto& surface) -> Result {
        if (surface != candidates[failedIndex]) return Result::SUCCESS;
        if (throws) throw std::runtime_error("queued registration");
        return Result::ERROR;
    };
    std::array<Result, 3> results{};
    std::vector<std::thread> workers;
    workers.reserve(3);
    const auto enqueue = [&](const std::shared_ptr<Render::WindowAttachment>& attachment, U64 index) {
        const auto before = attachment.use_count();
        workers.emplace_back([&, attachment, index]() { results[index] = window.bind(attachment); });
        while (attachment.use_count() < before + 2) std::this_thread::yield();
    };
    enqueue(candidates[0], 0);
    enqueue(candidates[1], 1);
    enqueue(root, 2);
    bool threw = false;
    Result result = Result::ERROR;
    try {
        result = window.begin();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    for (auto& worker : workers) worker.join();
    if (result == Result::SUCCESS) REQUIRE(window.cancel() == Result::SUCCESS);
    REQUIRE(window.stop() == Result::SUCCESS);
    REQUIRE(threw == throws);
    REQUIRE(result == Result::ERROR);
    REQUIRE(window.registeredSurfaceCount() == failedIndex + 1);
    REQUIRE(root->live);
    REQUIRE(root->createCount == 1);
    REQUIRE(root->destroyCount == 0);
    window.bindHandler = {};
    for (U64 i = 0; i < candidates.size(); ++i) {
        REQUIRE(programs[i]->creates == 1);
        if (i < failedIndex) {
            REQUIRE(programs[i]->live);
            REQUIRE(programs[i]->destroys == 0);
            REQUIRE(window.extend(target, {programs[i]}, {}) == Result::ERROR);
        } else {
            REQUIRE_FALSE(programs[i]->live);
            REQUIRE(programs[i]->destroys == 1);
            REQUIRE(window.extend(target, {programs[i]}, {}) == Result::SUCCESS);
        }
    }
    REQUIRE(window.bind(root) == Result::SUCCESS);
    REQUIRE(root->createCount == 1);
    REQUIRE(window.destroy() == Result::SUCCESS);
    REQUIRE(root->destroyCount == 1);
    for (U64 i = 0; i < programs.size(); ++i) {
        REQUIRE(programs[i]->destroys == (i < failedIndex ? 1 : 2));
    }
}

TEST_CASE("Surface append validation rejects null and incompatible backend objects before acquisition",
          "[core][render][surface][extend]") {
    TestWindow window;
    auto surface = std::make_shared<TestSurface>(Render::Surface::Config{.retained = true});
    surface->acceptsAppends = true;
    surface->typedAppends = true;
    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.bind(surface) == Result::SUCCESS);
    REQUIRE(surface->wantsDraw());
    surface->commitDraw();
    std::array<U32, 2> source = {1, 2};
    auto buffer = MakeBuffer(source);
    Render::Program::Config programConfig;
    programConfig.buffers = {{buffer, Render::Program::Target::VERTEX}};
    Render::Kernel::Config kernelConfig;
    kernelConfig.buffers = {{buffer, Render::Kernel::AccessMode::READ}};
    auto validProgram = std::make_shared<CreationProgram>(programConfig);
    auto validKernel = std::make_shared<CreationKernel>(kernelConfig);
    std::vector<std::shared_ptr<Render::Program>> programs = {validProgram};
    std::vector<std::shared_ptr<Render::Kernel>> kernels = {validKernel};
    SECTION("Null program") { programs.push_back(nullptr); }
    SECTION("Wrong program backend") { programs.push_back(std::make_shared<Render::Program>(programConfig)); }
    SECTION("Null kernel") { kernels.push_back(nullptr); }
    SECTION("Wrong kernel backend") { kernels.push_back(std::make_shared<TestKernel>(kernelConfig)); }
    REQUIRE(window.extend(surface, programs, kernels) == Result::ERROR);
    REQUIRE(surface->appendCalls == 0);
    REQUIRE(surface->getConfig().programs.empty());
    REQUIRE(surface->getConfig().kernels.empty());
    REQUIRE(surface->backendPrograms.empty());
    REQUIRE(surface->backendKernels.empty());
    REQUIRE(buffer->createCount == 0);
    REQUIRE_FALSE(surface->wantsDraw());
    REQUIRE(window.extend(surface, {validProgram}, {validKernel}) == Result::SUCCESS);
    REQUIRE(validProgram->creates == 1);
    REQUIRE(validKernel->creates == 1);
    REQUIRE(window.destroy() == Result::SUCCESS);
    REQUIRE(validProgram->destroys == 1);
    REQUIRE(validKernel->destroys == 1);
    REQUIRE(buffer->destroyCount == 1);
}

TEST_CASE("Surface appends reject mutable object aliases without recreating live resources",
          "[core][render][surface][extend]") {
    TestWindow window;
    TestWindow otherWindow;
    auto surface = std::make_shared<TestSurface>(Render::Surface::Config{.retained = true});
    auto sibling = std::make_shared<TestSurface>(Render::Surface::Config{.retained = true});
    surface->acceptsAppends = true;
    sibling->acceptsAppends = true;
    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(otherWindow.create() == Result::SUCCESS);
    REQUIRE(window.bind(surface) == Result::SUCCESS);
    REQUIRE(otherWindow.bind(sibling) == Result::SUCCESS);
    auto vertex = std::make_shared<Render::Vertex>(Render::Vertex::Config{});
    auto draw = std::make_shared<TestDraw>(Render::Draw::Config{.buffer = vertex});
    Render::Program::Config config;
    config.draws = {draw};
    auto program = std::make_shared<CreationProgram>(config);
    auto kernel = std::make_shared<CreationKernel>();
    REQUIRE(window.extend(surface, {program}, {kernel}) == Result::SUCCESS);
    REQUIRE(window.start() == Result::SUCCESS);
    REQUIRE(window.begin() == Result::SUCCESS);
    REQUIRE(window.end() == Result::SUCCESS);
    REQUIRE(window.outstandingSubmissions == 1);
    REQUIRE_FALSE(surface->wantsDraw());

    auto target = surface;
    TestWindow* targetWindow = &window;
    std::vector<std::shared_ptr<Render::Program>> programs;
    std::vector<std::shared_ptr<Render::Kernel>> kernels;
    SECTION("Existing program") { programs = {program}; }
    SECTION("Existing kernel") { kernels = {kernel}; }
    SECTION("Duplicate programs") {
        auto fresh = std::make_shared<CreationProgram>();
        programs = {fresh, fresh};
    }
    SECTION("Duplicate kernels") {
        auto fresh = std::make_shared<CreationKernel>();
        kernels = {fresh, fresh};
    }
    SECTION("Live draw") { programs = {std::make_shared<CreationProgram>(config)}; }
    SECTION("Live vertex") {
        config.draws = {std::make_shared<TestDraw>(Render::Draw::Config{.buffer = vertex})};
        programs = {std::make_shared<CreationProgram>(config)};
    }
    SECTION("Duplicate draws") {
        auto fresh = std::make_shared<TestDraw>(Render::Draw::Config{});
        config.draws = {fresh, fresh};
        programs = {std::make_shared<CreationProgram>(config)};
    }
    SECTION("Duplicate vertices") {
        auto fresh = std::make_shared<Render::Vertex>(Render::Vertex::Config{});
        config.draws = {std::make_shared<TestDraw>(Render::Draw::Config{.buffer = fresh}),
                        std::make_shared<TestDraw>(Render::Draw::Config{.buffer = fresh})};
        programs = {std::make_shared<CreationProgram>(config)};
    }
    SECTION("Draw shared by appended programs") {
        auto fresh = std::make_shared<TestDraw>(Render::Draw::Config{});
        config.draws = {fresh};
        programs = {std::make_shared<CreationProgram>(config), std::make_shared<CreationProgram>(config)};
    }
    SECTION("Another window") {
        target = sibling;
        targetWindow = &otherWindow;
        programs = {program};
    }
    SECTION("Retiring program") {
        REQUIRE(window.unbind(surface) == Result::SUCCESS);
        target = sibling;
        targetWindow = &otherWindow;
        programs = {program};
    }
    SECTION("Retiring kernel") {
        REQUIRE(window.unbind(surface) == Result::SUCCESS);
        target = sibling;
        targetWindow = &otherWindow;
        kernels = {kernel};
    }
    SECTION("Retiring draw") {
        REQUIRE(window.unbind(surface) == Result::SUCCESS);
        target = sibling;
        targetWindow = &otherWindow;
        programs = {std::make_shared<CreationProgram>(config)};
    }
    SECTION("Retiring vertex") {
        REQUIRE(window.unbind(surface) == Result::SUCCESS);
        target = sibling;
        targetWindow = &otherWindow;
        config.draws = {std::make_shared<TestDraw>(Render::Draw::Config{.buffer = vertex})};
        programs = {std::make_shared<CreationProgram>(config)};
    }
    const U64 appendCalls = target->appendCalls;
    const auto before = target->getConfig();
    REQUIRE(targetWindow->extend(target, programs, kernels) == Result::ERROR);
    REQUIRE(target->appendCalls == appendCalls);
    REQUIRE(target->getConfig().programs == before.programs);
    REQUIRE(target->getConfig().kernels == before.kernels);
    REQUIRE(program->creates == 1);
    REQUIRE(kernel->creates == 1);
    REQUIRE(program->destroys == 0);
    REQUIRE(kernel->destroys == 0);
    REQUIRE_FALSE(surface->wantsDraw());
    REQUIRE(window.outstandingSubmissions == 1);
    REQUIRE(window.stop() == Result::SUCCESS);
    REQUIRE(window.destroy() == Result::SUCCESS);
    REQUIRE(program->destroys == 1);
    REQUIRE(kernel->destroys == 1);
    REQUIRE(otherWindow.extend(sibling, {program}, {kernel}) == Result::SUCCESS);
    REQUIRE(program->creates == 2);
    REQUIRE(kernel->creates == 2);
    REQUIRE(otherWindow.destroy() == Result::SUCCESS);
    REQUIRE(program->destroys == 2);
    REQUIRE(kernel->destroys == 2);
}

TEST_CASE("Extending from an active graphical frame returns promptly and preserves frame completion",
          "[core][render][surface][extend][window]") {
    bool cancelFrame = false;
    SECTION("End") {}
    SECTION("Cancel") { cancelFrame = true; }
    auto window = std::make_shared<TestWindow>();
    auto surface = std::make_shared<TestSurface>(Render::Surface::Config{});
    auto program = std::make_shared<CreationProgram>();
    surface->acceptsAppends = true;
    REQUIRE(window->create() == Result::SUCCESS);
    REQUIRE(window->bind(surface) == Result::SUCCESS);
    REQUIRE(window->start() == Result::SUCCESS);
    auto result = std::make_shared<std::array<Result, 3>>();
    auto completed = std::make_shared<std::atomic_bool>(false);
    std::thread worker([=]() {
        (*result)[0] = window->begin();
        (*result)[1] = window->extend(surface, {program}, {});
        (*result)[2] = cancelFrame ? window->cancel() : window->end();
        completed->store(true, std::memory_order_release);
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!completed->load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    if (!completed->load(std::memory_order_acquire)) {
        worker.detach();
        FAIL("Active-frame extension did not return within two seconds");
    }
    worker.join();
    REQUIRE((*result)[0] == Result::SUCCESS);
    REQUIRE((*result)[1] == Result::SKIP);
    REQUIRE((*result)[2] == Result::SUCCESS);
    REQUIRE(surface->appendCalls == 0);
    REQUIRE(surface->getConfig().programs.empty());
    REQUIRE(program->creates == 0);
    REQUIRE(window->stop() == Result::SUCCESS);
    REQUIRE(window->extend(surface, {program}, {}) == Result::SUCCESS);
    REQUIRE(window->destroy() == Result::SUCCESS);
}

TEST_CASE("Retired components survive failed frames and synchronization failures",
          "[core][render][window][retirement]") {
    TestWindow window;
    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.start() == Result::SUCCESS);
    auto component = std::make_shared<Render::Components::Generic>();
    std::weak_ptr<Render::Components::Generic> lifetime = component;
    REQUIRE(window.bind(component) == Result::SUCCESS);
    for (int frame = 0; frame < 3; ++frame) {
        REQUIRE(window.begin() == Result::SUCCESS);
        REQUIRE(window.end() == Result::SUCCESS);
    }
    REQUIRE(window.outstandingSubmissions == 3);
    REQUIRE(window.unbind(component) == Result::SUCCESS);
    component.reset();

    window.nextSubmissionResult(Result::ERROR);
    for (int frame = 0; frame < 260; ++frame) {
        REQUIRE(window.begin() == Result::SUCCESS);
        REQUIRE(window.end() == Result::ERROR);
        REQUIRE_FALSE(lifetime.expired());
    }
    REQUIRE(window.outstandingSubmissions == 3);
    REQUIRE(window.begin() == Result::SUCCESS);
    REQUIRE(window.cancel() == Result::SUCCESS);
    REQUIRE_FALSE(lifetime.expired());

    window.nextSubmissionResult(Result::SUCCESS);
    window.synchronizeHandler = [&]() {
        REQUIRE_FALSE(lifetime.expired());
        return Result::ERROR;
    };
    for (int frame = 0; frame < 3; ++frame) {
        REQUIRE(window.begin() == Result::SUCCESS);
        REQUIRE(window.end() == Result::SUCCESS);
        REQUIRE_FALSE(lifetime.expired());
    }
    REQUIRE(window.begin() == Result::SUCCESS);
    REQUIRE(window.end() == Result::ERROR);
    REQUIRE_FALSE(lifetime.expired());
    window.synchronizeHandler = {};
    REQUIRE(window.begin() == Result::SUCCESS);
    REQUIRE(window.end() == Result::SUCCESS);
    REQUIRE(lifetime.expired());
    REQUIRE(window.outstandingSubmissions == 0);
    REQUIRE(window.stop() == Result::SUCCESS);
    REQUIRE(window.destroy() == Result::SUCCESS);
}

TEST_CASE("Component retirement countdowns saturate and preserve independent grace periods",
          "[core][render][window][retirement]") {
    TestWindow window;
    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.start() == Result::SUCCESS);
    const auto retire = [&]() {
        auto component = std::make_shared<Render::Components::Generic>();
        REQUIRE(window.bind(component) == Result::SUCCESS);
        REQUIRE(window.unbind(component) == Result::SUCCESS);
        return std::weak_ptr<Render::Components::Generic>(component);
    };
    const auto frame = [&]() {
        REQUIRE(window.begin() == Result::SUCCESS);
        return window.end();
    };
    auto first = retire();
    window.synchronizeHandler = []() { return Result::ERROR; };
    for (int count = 0; count < 260; ++count) {
        REQUIRE(frame() == (count < 3 ? Result::SUCCESS : Result::ERROR));
        REQUIRE_FALSE(first.expired());
    }
    auto second = retire();
    window.synchronizeHandler = {};
    REQUIRE(frame() == Result::SUCCESS);
    REQUIRE(first.expired());
    REQUIRE_FALSE(second.expired());
    for (int count = 0; count < 2; ++count) {
        REQUIRE(frame() == Result::SUCCESS);
        REQUIRE_FALSE(second.expired());
    }
    REQUIRE(frame() == Result::SUCCESS);
    REQUIRE(second.expired());
    REQUIRE(window.stop() == Result::SUCCESS);
    REQUIRE(window.destroy() == Result::SUCCESS);
}

TEST_CASE("Window destruction drains retired storage without subsequent frames and supports reuse",
          "[core][render][window][retirement]") {
    TestWindow window;
    REQUIRE(window.create() == Result::SUCCESS);
    auto component = std::make_shared<Render::Components::Generic>();
    std::weak_ptr<Render::Components::Generic> lifetime = component;
    REQUIRE(window.bind(component) == Result::SUCCESS);
    component.reset();
    window.synchronizeHandler = [&]() {
        REQUIRE_FALSE(lifetime.expired());
        return Result::ERROR;
    };
    REQUIRE(window.destroy() == Result::ERROR);
    REQUIRE_FALSE(lifetime.expired());
    REQUIRE(window.destructionCount() == 0);
    window.synchronizeHandler = {};
    window.nextDestructionResult(Result::ERROR);
    REQUIRE(window.destroy() == Result::ERROR);
    REQUIRE_FALSE(lifetime.expired());
    window.nextDestructionResult(Result::SUCCESS);
    REQUIRE(window.destroy() == Result::SUCCESS);
    REQUIRE(lifetime.expired());

    REQUIRE(window.create() == Result::SUCCESS);
    component = std::make_shared<Render::Components::Generic>();
    lifetime = component;
    REQUIRE(window.bind(component) == Result::SUCCESS);
    REQUIRE(window.unbind(component) == Result::SUCCESS);
    component.reset();
    REQUIRE(window.destroy() == Result::SUCCESS);
    REQUIRE(lifetime.expired());
}

TEST_CASE("Retired component destructors can reenter window synchronization and state queries",
          "[core][render][window][retirement]") {
    struct ReentrantComponent final : Render::Components::Generic {
        std::function<void()> onRelease;
        ~ReentrantComponent() override { onRelease(); }
    };
    TestWindow window;
    bool released = false;
    bool releaseOnDestroy = false;
    SECTION("Frame retirement") {}
    SECTION("Window destruction") { releaseOnDestroy = true; }
    REQUIRE(window.create() == Result::SUCCESS);
    auto component = std::make_shared<ReentrantComponent>();
    component->onRelease = [&]() {
        Render::Transfer::Batch batch;
        released = window.synchronize() == Result::SUCCESS &&
                   window.collectFrameTransfers(batch) == Result::SUCCESS;
    };
    REQUIRE(window.bind(component) == Result::SUCCESS);
    REQUIRE(window.unbind(component) == Result::SUCCESS);
    component.reset();
    if (!releaseOnDestroy) {
        REQUIRE(window.start() == Result::SUCCESS);
        for (int frame = 0; frame < 4; ++frame) {
            REQUIRE(window.begin() == Result::SUCCESS);
            REQUIRE(window.end() == Result::SUCCESS);
        }
        REQUIRE(released);
        REQUIRE(window.stop() == Result::SUCCESS);
    }
    REQUIRE(window.destroy() == Result::SUCCESS);
    REQUIRE(released);
}

TEST_CASE("Render transfer ranges align and reject overflow", "[core][render][transfer]") {
    U64 used = 3;
    U64 offset = 0;

    REQUIRE(TestTransfer::reserve(used, 4, 4, offset));
    REQUIRE(offset == 4);
    REQUIRE(used == 8);
    REQUIRE_FALSE(TestTransfer::reserve(used, 1, 0, offset));

    used = std::numeric_limits<U64>::max();
    REQUIRE_FALSE(TestTransfer::reserve(used, 1, 4, offset));
}

TEST_CASE("Render transfer capacities grow and align", "[core][render][transfer]") {
    U64 capacity = 0;

    REQUIRE(TestTransfer::capacity(1, 4, capacity));
    REQUIRE(capacity == TestTransfer::minimumBufferSize());

    REQUIRE(TestTransfer::capacity(TestTransfer::minimumBufferSize() + 1, 4, capacity));
    REQUIRE(capacity == 2 * TestTransfer::minimumBufferSize());

    REQUIRE_FALSE(TestTransfer::capacity(1, 0, capacity));
    REQUIRE_FALSE(TestTransfer::capacity(std::numeric_limits<U64>::max(), 4, capacity));
}

TEST_CASE("Render transfer rows are padded when copied", "[core][render][transfer]") {
    std::array<U8, 6> source = {1, 2, 3, 4, 5, 6};
    std::array<U8, 8> destination{};
    const std::array<U8, 8> expected = {1, 2, 3, 0, 4, 5, 6, 0};
    U64 rowSize = 0;

    REQUIRE(TestTransfer::alignedSize(3, 4, rowSize));
    REQUIRE(rowSize == 4);
    REQUIRE(TestTransfer::copyRows(destination.data(), source.data(), 2, 3, rowSize));
    REQUIRE(destination == expected);
}

TEST_CASE("Render buffer transfers copy checked ranges", "[core][render][transfer]") {
    std::array<U8, 4> source = {1, 2, 3, 4};
    std::array<U8, 4> destination{};

    REQUIRE(TestTransfer::copy(destination.data(), source.data(), source.size()));
    REQUIRE(destination == source);
    REQUIRE(TestTransfer::copy(nullptr, nullptr, 0));
    REQUIRE_FALSE(TestTransfer::copy(nullptr, source.data(), 1));
}

TEST_CASE("Render transfer queues expose deterministic state transitions",
          "[core][render][transfer]") {
    Render::Transfer::PendingUploadQueue queue;
    std::array<U8, 8> source = {0, 1, 2, 3, 4, 5, 6, 7};

    REQUIRE(queue.empty());
    REQUIRE(queue.queue(0, 4, source.size(), 1, source.data()) == Result::SUCCESS);
    REQUIRE_FALSE(queue.empty());
    REQUIRE(queue.queue(0, 2, source.size() / 2, 2, source.data()) == Result::ERROR);

    auto uploads = queue.take();
    REQUIRE(uploads.size() == 1);
    REQUIRE(uploads[0].data == std::vector<U8>{0, 1, 2, 3});
    REQUIRE(queue.empty());

    REQUIRE(queue.queue(4, 4, source.size(), 1, source.data()) == Result::SUCCESS);
    queue.clear();
    REQUIRE(queue.empty());
}

TEST_CASE("Render transfer updates own their source data", "[core][render][transfer]") {
    Render::Transfer::PendingUploadQueue queue;
    std::array<U8, 8> source = {0, 1, 2, 3, 4, 5, 6, 7};

    REQUIRE(queue.queue(2, 4, source.size(), 1, source.data()) == Result::SUCCESS);
    source.fill(42);

    auto uploads = queue.take();
    REQUIRE(uploads.size() == 1);
    REQUIRE(uploads[0].start == 2);
    REQUIRE(uploads[0].data == std::vector<U8>{2, 3, 4, 5});
}

TEST_CASE("Render transfer updates coalesce with last write wins",
          "[core][render][transfer]") {
    Render::Transfer::PendingUploadQueue queue;
    std::array<U8, 8> source = {0, 1, 2, 3, 4, 5, 6, 7};

    REQUIRE(queue.queue(0, 4, source.size(), 1, source.data()) == Result::SUCCESS);
    source[2] = 12;
    source[3] = 13;
    source[4] = 14;
    source[5] = 15;
    REQUIRE(queue.queue(2, 4, source.size(), 1, source.data()) == Result::SUCCESS);
    REQUIRE(queue.queue(6, 2, source.size(), 1, source.data()) == Result::SUCCESS);

    auto uploads = queue.take();
    REQUIRE(uploads.size() == 1);
    REQUIRE(uploads[0].start == 0);
    REQUIRE(uploads[0].data == std::vector<U8>{0, 1, 12, 13, 14, 15, 6, 7});
}

TEST_CASE("Render transfer updates keep wrapped ranges disjoint",
          "[core][render][transfer]") {
    Render::Transfer::PendingUploadQueue queue;
    std::array<U8, 16> source{};

    REQUIRE(queue.queue(0, 2, source.size(), 1, source.data()) == Result::SUCCESS);
    REQUIRE(queue.queue(14, 2, source.size(), 1, source.data()) == Result::SUCCESS);

    auto uploads = queue.take();
    REQUIRE(uploads.size() == 2);
    REQUIRE(uploads[0].start == 0);
    REQUIRE(uploads[1].start == 14);
}

TEST_CASE("Restored transfers preserve newer updates", "[core][render][transfer]") {
    Render::Transfer::PendingUploadQueue queue;
    std::array<U8, 4> source = {1, 1, 1, 1};

    REQUIRE(queue.queue(0, 4, source.size(), 1, source.data()) == Result::SUCCESS);
    auto pending = queue.take();

    source = {1, 9, 9, 1};
    REQUIRE(queue.queue(1, 2, source.size(), 1, source.data()) == Result::SUCCESS);
    REQUIRE(queue.restore(std::move(pending)) == Result::SUCCESS);

    auto uploads = queue.take();
    REQUIRE(uploads.size() == 1);
    REQUIRE(uploads[0].data == std::vector<U8>{1, 9, 9, 1});
}

TEST_CASE("Restored transfers do not cross generations", "[core][render][transfer]") {
    Render::Transfer::PendingUploadQueue queue;
    std::array<U8, 2> source = {1, 1};

    REQUIRE(queue.queue(0, 2, source.size(), 1, source.data(), 1) == Result::SUCCESS);
    auto oldGeneration = queue.take();

    source = {2, 2};
    REQUIRE(queue.queue(0, 2, source.size(), 1, source.data(), 2) == Result::SUCCESS);
    REQUIRE(queue.restore(std::move(oldGeneration)) == Result::SUCCESS);

    auto uploads = queue.take();
    REQUIRE(uploads.size() == 1);
    REQUIRE(uploads[0].generation == 2);
    REQUIRE(uploads[0].data == std::vector<U8>{2, 2});
}

TEST_CASE("Malformed restored transfers are rejected", "[core][render][transfer]") {
    Render::Transfer::PendingUpload empty;
    REQUIRE(empty.extent() == 0);

    Render::Transfer::PendingUploadQueue queue;
    std::array<U8, 4> source = {1, 2, 3, 4};
    REQUIRE(queue.queue(0, source.size(), source.size(), 1, source.data()) ==
            Result::SUCCESS);

    std::vector<Render::Transfer::PendingUpload> malformed(1);
    malformed[0].unitByteSize = 2;
    malformed[0].data = {5, 6, 7};
    REQUIRE(queue.restore(std::move(malformed)) == Result::ERROR);

    auto uploads = queue.take();
    REQUIRE(uploads.size() == 1);
    REQUIRE(uploads[0].data == std::vector<U8>{1, 2, 3, 4});

    Render::Transfer::PendingUpload overflowing;
    overflowing.start = std::numeric_limits<U64>::max();
    overflowing.unitByteSize = 1;
    overflowing.data = {8};
    REQUIRE(queue.restore({std::move(overflowing)}) == Result::ERROR);

    Render::Transfer::PendingUpload byteEndOverflow;
    byteEndOverflow.start = std::numeric_limits<U64>::max() / 2;
    byteEndOverflow.unitByteSize = 2;
    byteEndOverflow.data = {9, 10};
    REQUIRE(queue.restore({std::move(byteEndOverflow)}) == Result::ERROR);
}

TEST_CASE("Texture resize discards aborted uploads from the previous size",
          "[core][render][transfer]") {
    std::array<U8, 36> source{};
    auto texture = std::make_shared<TestTexture>(Render::Texture::Config{
        .size = {2, 2},
        .buffer = source.data(),
    });

    REQUIRE(texture->fill() == Result::SUCCESS);
    {
        Render::Transfer::Batch batch;
        batch.collect(texture);
        REQUIRE(batch.textures().size() == 1);
        const Extent2D<U64> originalSize{2, 2};
        REQUIRE(batch.textures()[0].destinationSize == originalSize);
        REQUIRE(texture->size({3, 3}));
        U64 rowCount = 0;
        REQUIRE_FALSE(TestTransfer::validTexture(batch.textures()[0], rowCount));
    }

    Render::Transfer::Batch batch;
    batch.collect(texture);
    REQUIRE(batch.empty());
}

TEST_CASE("Render buffer updates require four-byte alignment",
          "[core][render][transfer]") {
    std::array<U8, 8> source{};
    TestBuffer buffer({
        .size = source.size(),
        .target = Render::Buffer::Target::STORAGE,
        .elementByteSize = sizeof(U8),
        .buffer = source.data(),
    });

    REQUIRE(buffer.update() == Result::SUCCESS);
    REQUIRE(buffer.update(0, 4) == Result::SUCCESS);
    REQUIRE(buffer.update(1, 4) == Result::ERROR);
    REQUIRE(buffer.update(0, 3) == Result::ERROR);
}

TEST_CASE("Render transfer updates validate ranges", "[core][render][transfer]") {
    Render::Transfer::PendingUploadQueue queue;
    std::array<U8, 4> source{};

    REQUIRE(queue.queue(0, 0, source.size(), 1, nullptr) == Result::SUCCESS);
    REQUIRE(queue.queue(4, 1, source.size(), 1, source.data()) == Result::ERROR);
    REQUIRE(queue.queue(0, 1, source.size(), 0, source.data()) == Result::ERROR);
    REQUIRE(queue.queue(0, 1, source.size(), 1, nullptr) == Result::ERROR);
    REQUIRE(queue.queue(0,
                        1,
                        std::numeric_limits<U64>::max(),
                        2,
                        source.data()) == Result::ERROR);
}

TEST_CASE("Render transfer updates preserve their unit size", "[core][render][transfer]") {
    Render::Transfer::PendingUploadQueue queue;
    std::array<U32, 4> source = {10, 20, 30, 40};

    REQUIRE(queue.queue(1,
                        2,
                        source.size(),
                        sizeof(U32),
                        reinterpret_cast<const U8*>(source.data())) == Result::SUCCESS);

    auto uploads = queue.take();
    REQUIRE(uploads.size() == 1);
    REQUIRE(uploads[0].start == 1);
    REQUIRE(uploads[0].unitByteSize == sizeof(U32));
    REQUIRE(uploads[0].data.size() == 2 * sizeof(U32));
}

TEST_CASE("Transfer batches collect each resource once and restore aborted work",
          "[core][render][transfer]") {
    std::array<U32, 2> source = {10, 20};
    auto buffer = std::make_shared<TestBuffer>(Render::Buffer::Config{
        .size = source.size(),
        .target = Render::Buffer::Target::STORAGE,
        .elementByteSize = sizeof(U32),
        .buffer = source.data(),
    });

    REQUIRE(buffer->update() == Result::SUCCESS);
    {
        Render::Transfer::Batch batch;
        batch.collect(buffer);
        batch.collect(buffer);
        REQUIRE(batch.buffers().size() == 1);
    }

    Render::Transfer::Batch restored;
    restored.collect(buffer);
    REQUIRE(restored.buffers().size() == 1);
    restored.commit();

    Render::Transfer::Batch empty;
    empty.collect(buffer);
    REQUIRE(empty.empty());
}

TEST_CASE("Transfer batches restore work accepted after an earlier commit",
          "[core][render][transfer]") {
    std::array<U32, 2> source = {10, 20};
    auto buffer = MakeBuffer(source);

    REQUIRE(buffer->update() == Result::SUCCESS);
    {
        Render::Transfer::Batch batch;
        batch.collect(buffer);
        REQUIRE(batch.buffers().size() == 1);
        batch.commit();

        source = {30, 40};
        REQUIRE(buffer->update() == Result::SUCCESS);
        batch.collect(buffer);
        REQUIRE(batch.buffers().size() == 1);
    }

    Render::Transfer::Batch restored;
    restored.collect(buffer);
    REQUIRE(restored.buffers().size() == 1);
    const auto* bufferBytes = reinterpret_cast<const U8*>(source.data());
    const std::vector<U8> expectedBufferData(bufferBytes, bufferBytes + sizeof(source));
    REQUIRE(restored.buffers()[0].upload.data == expectedBufferData);
    restored.commit();

    std::array<U8, 16> textureSource{};
    auto texture = std::make_shared<TestTexture>(Render::Texture::Config{
        .size = {2, 2},
        .buffer = textureSource.data(),
    });

    REQUIRE(texture->fill() == Result::SUCCESS);
    {
        Render::Transfer::Batch batch;
        batch.collect(texture);
        REQUIRE(batch.textures().size() == 1);
        batch.commit();

        textureSource.fill(42);
        REQUIRE(texture->fill() == Result::SUCCESS);
        batch.collect(texture);
        REQUIRE(batch.textures().size() == 1);
    }

    Render::Transfer::Batch restoredTexture;
    restoredTexture.collect(texture);
    REQUIRE(restoredTexture.textures().size() == 1);
    REQUIRE(restoredTexture.textures()[0].upload.data ==
            std::vector<U8>(textureSource.begin(), textureSource.end()));
    restoredTexture.commit();
}

TEST_CASE("Standalone attachments do not make transfer collection order-dependent",
          "[core][render][transfer][surface]") {
    const auto checkBufferOrder = [](const bool attachmentFirst) {
        TestWindow window;
        std::array<U32, 2> source = {10, 20};
        auto buffer = MakeBuffer(source);
        auto surface = MakeRetainedSurface(buffer);

        REQUIRE(window.create() == Result::SUCCESS);
        if (attachmentFirst) {
            REQUIRE(window.bind(buffer) == Result::SUCCESS);
            REQUIRE(window.bind(surface) == Result::SUCCESS);
        } else {
            REQUIRE(window.bind(surface) == Result::SUCCESS);
            REQUIRE(window.bind(buffer) == Result::SUCCESS);
        }

        REQUIRE(surface->wantsDraw());
        surface->commitDraw();
        REQUIRE(buffer->update() == Result::SUCCESS);

        Render::Transfer::Batch batch;
        REQUIRE(window.collectFrameTransfers(batch) == Result::SUCCESS);
        REQUIRE(batch.buffers().size() == 1);
        REQUIRE(batch.contains(buffer));
        REQUIRE(surface->wantsDraw());

        surface->commitDraw();
        batch.commit();
        REQUIRE(window.destroy() == Result::SUCCESS);
    };

    const auto checkTextureOrder = [](const bool attachmentFirst) {
        TestWindow window;
        std::array<U8, 16> source{};
        auto texture = std::make_shared<TestTexture>(Render::Texture::Config{
            .size = {2, 2},
            .buffer = source.data(),
        });
        auto surface = MakeRetainedSurface(texture);

        REQUIRE(window.create() == Result::SUCCESS);
        if (attachmentFirst) {
            REQUIRE(window.bind(texture) == Result::SUCCESS);
            REQUIRE(window.bind(surface) == Result::SUCCESS);
        } else {
            REQUIRE(window.bind(surface) == Result::SUCCESS);
            REQUIRE(window.bind(texture) == Result::SUCCESS);
        }

        REQUIRE(surface->wantsDraw());
        surface->commitDraw();
        REQUIRE(texture->fill() == Result::SUCCESS);

        Render::Transfer::Batch batch;
        REQUIRE(window.collectFrameTransfers(batch) == Result::SUCCESS);
        REQUIRE(batch.textures().size() == 1);
        REQUIRE(batch.contains(texture));
        REQUIRE(surface->wantsDraw());

        surface->commitDraw();
        batch.commit();
        REQUIRE(window.destroy() == Result::SUCCESS);
    };

    SECTION("Buffer attachment before Surface") {
        checkBufferOrder(true);
    }
    SECTION("Surface before Buffer attachment") {
        checkBufferOrder(false);
    }
    SECTION("Texture attachment before Surface") {
        checkTextureOrder(true);
    }
    SECTION("Surface before Texture attachment") {
        checkTextureOrder(false);
    }
}

TEST_CASE("Failed render window creation rolls back its backend", "[core][render][window]") {
    TestWindow failed;
    TestWindow replacement;

    failed.nextCreationResult(Result::ERROR);
    REQUIRE(failed.create() == Result::ERROR);
    REQUIRE(failed.destructionCount() == 1);

    REQUIRE(replacement.create() == Result::SUCCESS);
    REQUIRE(replacement.destroy() == Result::SUCCESS);
}

TEST_CASE("Render window chrome defaults to the system frame", "[core][render][window]") {
    TestWindow window;

    const auto chrome = window.titlebarChrome();
    REQUIRE_FALSE(chrome.integrated);
    REQUIRE(chrome.leadingInset == 0.0f);
    REQUIRE(chrome.height == 0.0f);

    window.beginWindowDrag();
    window.zoomWindow();

    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE_FALSE(window.titlebarChrome().integrated);
    window.beginWindowDrag();
    window.zoomWindow();
    REQUIRE(window.destroy() == Result::SUCCESS);

    TestViewport viewport;
    const auto fallback = viewport.Viewport::Generic::titlebarChrome();
    REQUIRE_FALSE(fallback.integrated);
    REQUIRE(fallback.leadingInset == 0.0f);
    REQUIRE(fallback.height == 0.0f);
    viewport.Viewport::Generic::beginWindowDrag();
    viewport.Viewport::Generic::zoomWindow();
    REQUIRE(viewport.drags == 0);
    REQUIRE(viewport.zooms == 0);
}

TEST_CASE("Integrated titlebar activation is opt in and forwards through the viewport API",
          "[core][render][window][review17]") {
    CHECK_FALSE(Viewport::Config{}.integratedTitlebar);
    TestViewport legacy;
    CHECK_FALSE(legacy.integratedTitlebar());
    CHECK(legacy.activations.empty());
    TestViewport retained(Viewport::Config{.integratedTitlebar = true});
    CHECK(retained.integratedTitlebar());
    Viewport::Generic& viewport = legacy;
    viewport.setIntegratedTitlebar(true);
    CHECK(viewport.integratedTitlebar());
    viewport.setIntegratedTitlebar(true);
    viewport.setIntegratedTitlebar(false);
    CHECK_FALSE(viewport.integratedTitlebar());
    CHECK(legacy.activations == std::vector<bool>{true, true, false});
    CHECK_FALSE(retained.Viewport::Generic::titlebarChrome().integrated);
}

TEST_CASE("Render window chrome forwards to its viewport in logical units", "[core][render][window]") {
    TestWindow window;
    TestViewport viewport;

    REQUIRE(window.create() == Result::SUCCESS);
    window.attachViewport(&viewport);

    const auto chrome = window.titlebarChrome();
    REQUIRE(chrome.integrated);
    REQUIRE(chrome.leadingInset == 40.0f);
    REQUIRE(chrome.height == 16.0f);

    window.beginWindowDrag();
    window.zoomWindow();
    window.zoomWindow();
    REQUIRE(viewport.drags == 1);
    REQUIRE(viewport.zooms == 2);

    REQUIRE(window.destroy() == Result::SUCCESS);
}

TEST_CASE("Failed render window destruction remains retryable", "[core][render][window]") {
    TestWindow window;

    REQUIRE(window.create() == Result::SUCCESS);
    window.nextDestructionResult(Result::ERROR);
    REQUIRE(window.destroy() == Result::ERROR);
    REQUIRE(window.destructionCount() == 1);

    window.nextDestructionResult(Result::SUCCESS);
    REQUIRE(window.destroy() == Result::SUCCESS);
    REQUIRE(window.destructionCount() == 2);
}

TEST_CASE("Surfaces lease shared resources once", "[core][render][window][surface]") {
    TestWindow window;
    std::array<U32, 2> source = {10, 20};
    std::array<U8, 16> textureSource{};
    auto buffer = MakeBuffer(source);
    auto texture = std::make_shared<TestTexture>(Render::Texture::Config{
        .size = {2, 2},
        .buffer = textureSource.data(),
    });
    auto firstSurface = MakeRetainedSurface(buffer, texture);
    auto secondSurface = MakeRetainedSurface(buffer, texture);

    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.bind(firstSurface) == Result::SUCCESS);
    REQUIRE(window.bind(secondSurface) == Result::SUCCESS);
    REQUIRE(buffer->createCount == 1);
    REQUIRE(buffer->destroyCount == 0);
    REQUIRE(texture->createCount == 1);
    REQUIRE(texture->destroyCount == 0);

    REQUIRE(window.destroy() == Result::SUCCESS);
    REQUIRE(buffer->destroyCount == 1);
    REQUIRE(texture->destroyCount == 1);
}

TEST_CASE("Failed surface resource acquisition rolls back prior leases",
          "[core][render][window][surface]") {
    TestWindow failedWindow;
    TestWindow replacement;
    std::array<U32, 2> source = {10, 20};
    std::array<U8, 16> textureSource{};
    auto buffer = MakeBuffer(source);
    auto texture = std::make_shared<TestTexture>(Render::Texture::Config{
        .size = {2, 2},
        .buffer = textureSource.data(),
    });
    auto surface = MakeRetainedSurface(buffer, texture);

    texture->nextCreationResult(Result::ERROR);
    REQUIRE(failedWindow.create() == Result::SUCCESS);
    REQUIRE(failedWindow.bind(surface) == Result::ERROR);
    REQUIRE(buffer->createCount == 1);
    REQUIRE(buffer->destroyCount == 1);
    REQUIRE(texture->createCount == 1);
    REQUIRE(texture->destroyCount == 1);
    REQUIRE(failedWindow.destroy() == Result::SUCCESS);

    texture->nextCreationResult(Result::SUCCESS);
    REQUIRE(replacement.create() == Result::SUCCESS);
    REQUIRE(replacement.bind(surface) == Result::SUCCESS);
    REQUIRE(buffer->createCount == 2);
    REQUIRE(texture->createCount == 2);
    REQUIRE(replacement.destroy() == Result::SUCCESS);
    REQUIRE(buffer->destroyCount == 2);
    REQUIRE(texture->destroyCount == 2);
}

TEST_CASE("Failed resource destruction delays backend teardown and remains retryable",
          "[core][render][window]") {
    TestWindow window;
    std::array<U32, 2> source = {10, 20};
    auto buffer = MakeBuffer(source);

    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.bind(buffer) == Result::SUCCESS);
    buffer->nextDestructionResult(Result::ERROR);
    REQUIRE(window.destroy() == Result::ERROR);
    REQUIRE(buffer->destroyCount == 1);
    REQUIRE(window.destructionCount() == 0);

    buffer->nextDestructionResult(Result::SUCCESS);
    REQUIRE(window.destroy() == Result::SUCCESS);
    REQUIRE(buffer->destroyCount == 2);
    REQUIRE(window.destructionCount() == 1);
}

TEST_CASE("Surface framebuffers reject sampled sharing",
          "[core][render][window][surface]") {
    TestWindow window;
    std::array<U8, 16> textureSource{};
    auto texture = std::make_shared<TestTexture>(Render::Texture::Config{
        .size = {2, 2},
        .buffer = textureSource.data(),
    });
    auto producer = MakeFramebufferSurface(texture);
    auto consumer = MakeRetainedSurface(texture);

    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.bind(producer) == Result::SUCCESS);
    REQUIRE(window.bind(consumer) == Result::ERROR);
    REQUIRE(texture->createCount == 1);

    REQUIRE(window.destroy() == Result::SUCCESS);
    REQUIRE(texture->destroyCount == 1);
}

TEST_CASE("Surface resources can become standalone attachments",
          "[core][render][window][surface]") {
    TestWindow window;
    TestWindow replacement;
    std::array<U32, 2> source = {10, 20};
    auto buffer = MakeBuffer(source);
    auto surface = MakeRetainedSurface(buffer);

    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.bind(surface) == Result::SUCCESS);
    REQUIRE(window.bind(buffer) == Result::SUCCESS);
    REQUIRE(buffer->createCount == 1);

    REQUIRE(window.unbind(surface) == Result::SUCCESS);
    REQUIRE(window.unbind(buffer) == Result::SUCCESS);
    REQUIRE(window.destroy() == Result::SUCCESS);
    REQUIRE(buffer->destroyCount == 1);

    REQUIRE(replacement.create() == Result::SUCCESS);
    REQUIRE(replacement.bind(buffer) == Result::SUCCESS);
    REQUIRE(buffer->createCount == 2);
    REQUIRE(replacement.destroy() == Result::SUCCESS);
    REQUIRE(buffer->destroyCount == 2);
}

TEST_CASE("Standalone resource release keeps its own grace period",
          "[core][render][window][surface]") {
    TestWindow window;
    std::array<U32, 2> source = {10, 20};
    auto buffer = MakeBuffer(source);
    auto surface = MakeRetainedSurface(buffer);
    const auto renderFrame = [&]() {
        REQUIRE(window.begin() == Result::SUCCESS);
        REQUIRE(window.end() == Result::SUCCESS);
    };

    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.bind(surface) == Result::SUCCESS);
    REQUIRE(window.bind(buffer) == Result::SUCCESS);
    REQUIRE(window.start() == Result::SUCCESS);
    REQUIRE(window.unbind(surface) == Result::SUCCESS);

    renderFrame();
    renderFrame();
    renderFrame();
    REQUIRE(window.unbind(buffer) == Result::SUCCESS);

    renderFrame();
    renderFrame();
    renderFrame();
    renderFrame();
    REQUIRE(buffer->destroyCount == 0);
    renderFrame();
    REQUIRE(buffer->destroyCount == 1);

    REQUIRE(window.stop() == Result::SUCCESS);
    REQUIRE(window.destroy() == Result::SUCCESS);
}

TEST_CASE("Window destruction rejects concurrent resource binding",
          "[core][render][window][surface]") {
    TestWindow window;
    std::array<U32, 2> source = {10, 20};
    auto buffer = MakeBuffer(source);
    auto surface = MakeRetainedSurface(buffer);

    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.bind(surface) == Result::SUCCESS);
    surface->blockDestruction();

    Result destructionResult = Result::ERROR;
    std::thread destructionThread([&]() {
        destructionResult = window.destroy();
    });
    const auto destructionDeadline = std::chrono::steady_clock::now() +
                                     std::chrono::seconds(2);
    while (!surface->hasStartedDestruction() &&
           std::chrono::steady_clock::now() < destructionDeadline) {
        std::this_thread::yield();
    }

    const bool destructionStarted = surface->hasStartedDestruction();
    Result bindResult = Result::ERROR;
    if (destructionStarted) {
        bindResult = window.bind(buffer);
    }
    surface->resumeDestruction();
    destructionThread.join();
    REQUIRE(destructionStarted);
    REQUIRE(bindResult == Result::ERROR);
    REQUIRE(destructionResult == Result::SUCCESS);
    REQUIRE(buffer->destroyCount == 1);
}

TEST_CASE("Concurrent binding can snapshot retirement records while backend cleanup completes",
          "[core][render][window][retirement][threading]") {
    bool failCleanup = false;
    SECTION("Successful cleanup") {}
    SECTION("Failed cleanup followed by retry") { failCleanup = true; }
    TestWindow window;
    std::array<U32, 2> source = {10, 20};
    auto retiringBuffer = MakeBuffer(source);
    auto retiring = MakeRetainedSurface(retiringBuffer);
    constexpr U64 count = 128;
    std::array<std::shared_ptr<TestBuffer>, count> unrelated;
    Render::Kernel::Config kernelConfig;
    for (auto& buffer : unrelated) {
        buffer = MakeBuffer(source);
        kernelConfig.buffers.push_back({buffer, Render::Kernel::AccessMode::READ});
    }
    auto live = std::make_shared<TestSurface>(Render::Surface::Config{
        .kernels = {std::make_shared<Render::Kernel>(kernelConfig)}, .retained = true,
    });
    retiring->destructionHandler = [&]() { return failCleanup ? Result::ERROR : Result::SUCCESS; };
    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.bind(retiring) == Result::SUCCESS);
    REQUIRE(window.bind(live) == Result::SUCCESS);
    REQUIRE(window.unbind(retiring) == Result::SUCCESS);
    REQUIRE(window.start() == Result::SUCCESS);
    for (U64 frame = 0; frame < 4; ++frame) {
        REQUIRE(window.begin() == Result::SUCCESS);
        REQUIRE(window.end() == Result::SUCCESS);
    }
    REQUIRE_FALSE(retiring->hasStartedDestruction());
    retiring->blockDestruction();
    Result beginResult = Result::ERROR;
    Result endResult = Result::ERROR;
    std::thread renderer([&]() {
        beginResult = window.begin();
        if (beginResult == Result::SUCCESS) endResult = window.end();
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!retiring->hasStartedDestruction() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    const bool started = retiring->hasStartedDestruction();
    std::array<Result, count> bindings{};
    std::atomic<U64> completed = 0;
    std::thread binder;
    if (started) {
        binder = std::thread([&]() {
            for (U64 i = 0; i < unrelated.size(); ++i) {
                bindings[i] = window.bind(unrelated[i]);
                completed.fetch_add(1, std::memory_order_release);
            }
        });
        while (completed.load(std::memory_order_acquire) < 16 &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
    }
    const bool boundDuringCleanup = completed.load(std::memory_order_acquire) >= 16;
    retiring->resumeDestruction();
    renderer.join();
    if (binder.joinable()) binder.join();
    REQUIRE(started);
    REQUIRE(boundDuringCleanup);
    REQUIRE(beginResult == Result::SUCCESS);
    REQUIRE(endResult == (failCleanup ? Result::ERROR : Result::SUCCESS));
    REQUIRE(completed.load() == count);
    for (U64 i = 0; i < unrelated.size(); ++i) {
        REQUIRE(bindings[i] == Result::SUCCESS);
        REQUIRE(unrelated[i]->createCount == 1);
        REQUIRE(unrelated[i]->destroyCount == 0);
    }
    REQUIRE(window.registeredSurfaceCount() == 1);
    if (failCleanup) {
        REQUIRE(retiringBuffer->destroyCount == 0);
        REQUIRE(window.bind(retiring) == Result::ERROR);
        failCleanup = false;
        REQUIRE(window.begin() == Result::SUCCESS);
        REQUIRE(window.end() == Result::SUCCESS);
    }
    REQUIRE(retiringBuffer->destroyCount == 1);
    REQUIRE(window.stop() == Result::SUCCESS);
    REQUIRE(window.destroy() == Result::SUCCESS);
    for (const auto& buffer : unrelated) REQUIRE(buffer->destroyCount == 1);
}

TEST_CASE("Cancelled render frames release the window", "[core][render][window]") {
    TestWindow window;

    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.start() == Result::SUCCESS);
    REQUIRE(window.begin() == Result::SUCCESS);
    REQUIRE(window.cancel() == Result::SUCCESS);
    REQUIRE(window.begin() == Result::SUCCESS);
    REQUIRE(window.end() == Result::SUCCESS);
    REQUIRE(window.stop() == Result::SUCCESS);
    REQUIRE(window.destroy() == Result::SUCCESS);
}

TEST_CASE("Transfers invalidate affected retained surfaces only",
          "[core][render][transfer][surface]") {
    TestWindow window;
    std::array<U32, 2> affectedSource = {10, 20};
    std::array<U32, 2> unrelatedSource = {30, 40};
    auto affectedBuffer = MakeBuffer(affectedSource);
    auto unrelatedBuffer = MakeBuffer(unrelatedSource);
    auto affectedSurface = MakeRetainedSurface(affectedBuffer);
    auto unrelatedSurface = MakeRetainedSurface(unrelatedBuffer);

    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.bind(affectedSurface) == Result::SUCCESS);
    REQUIRE(window.bind(unrelatedSurface) == Result::SUCCESS);

    REQUIRE(affectedSurface->wantsDraw());
    REQUIRE(unrelatedSurface->wantsDraw());
    affectedSurface->commitDraw();
    unrelatedSurface->commitDraw();
    REQUIRE_FALSE(affectedSurface->wantsDraw());
    REQUIRE_FALSE(unrelatedSurface->wantsDraw());

    REQUIRE(affectedBuffer->update() == Result::SUCCESS);
    Render::Transfer::Batch batch;
    REQUIRE(window.collectFrameTransfers(batch) == Result::SUCCESS);
    REQUIRE(batch.buffers().size() == 1);
    REQUIRE(affectedSurface->wantsDraw());
    REQUIRE_FALSE(unrelatedSurface->wantsDraw());

    affectedSurface->commitDraw();
    batch.commit();
    REQUIRE(window.destroy() == Result::SUCCESS);
}

TEST_CASE("Retained surfaces preserve explicit invalidation until a draw commits",
          "[core][render][surface]") {
    TestSurface retained(Render::Surface::Config{.retained = true});

    REQUIRE(retained.wantsDraw());
    retained.commitDraw();
    REQUIRE_FALSE(retained.wantsDraw());

    retained.clearColor({0.0f, 0.0f, 0.0f, 1.0f});
    REQUIRE_FALSE(retained.wantsDraw());
    retained.clearColor({1.0f, 0.0f, 0.0f, 1.0f});
    REQUIRE(retained.wantsDraw());
    retained.commitDraw();
    REQUIRE_FALSE(retained.wantsDraw());

    retained.invalidate();
    retained.commitDraw();
    REQUIRE(retained.wantsDraw());
    retained.commitDraw();
    REQUIRE(retained.wantsDraw(true));
    retained.commitDraw();
    REQUIRE_FALSE(retained.wantsDraw());

    TestSurface immediate(Render::Surface::Config{});
    REQUIRE(immediate.wantsDraw());
    immediate.commitDraw();
    REQUIRE(immediate.wantsDraw());
}

TEST_CASE("Hidden surfaces initialize images and redraw after skipped graphics",
          "[core][render][surface][visibility]") {
    bool retained = true;
    SECTION("Retained surface") {}
    SECTION("Immediate surface") { retained = false; }
    TestSurface surface(Render::Surface::Config{.retained = retained});

    REQUIRE(surface.create() == Result::SUCCESS);
    REQUIRE(surface.wantsDraw());
    REQUIRE(surface.wantsGraphics(false));
    // An aborted submission must not mark the new image initialized.
    REQUIRE(surface.wantsDraw());
    REQUIRE(surface.wantsGraphics(false));
    surface.commitDraw();

    surface.invalidate();
    for (int frame = 0; frame < 2; ++frame) {
        REQUIRE(surface.wantsDraw());
        REQUIRE_FALSE(surface.wantsGraphics(false));
        surface.commitDraw();
    }

    REQUIRE(surface.wantsDraw());
    REQUIRE(surface.wantsGraphics(true));
    surface.commitDraw();
    REQUIRE(surface.wantsDraw() == !retained);
}

TEST_CASE("Hidden surfaces initialize resized and recreated framebuffers",
          "[core][render][surface][visibility]") {
    TestSurface surface(Render::Surface::Config{.retained = true});
    REQUIRE(surface.create() == Result::SUCCESS);
    REQUIRE(surface.wantsDraw());
    REQUIRE(surface.wantsGraphics(true));
    surface.commitDraw();

    bool resized = false;
    SECTION("Framebuffer resize") { resized = true; }
    SECTION("Surface recreation") {
        REQUIRE(surface.destroy() == Result::SUCCESS);
        REQUIRE(surface.create() == Result::SUCCESS);
    }
    REQUIRE(surface.wantsDraw(resized));
    REQUIRE(surface.wantsGraphics(false));
    surface.commitDraw();

    surface.invalidate();
    REQUIRE(surface.wantsDraw());
    REQUIRE_FALSE(surface.wantsGraphics(false));
    surface.commitDraw();
    REQUIRE(surface.wantsDraw());
    REQUIRE(surface.wantsGraphics(true));
    surface.commitDraw();
    REQUIRE_FALSE(surface.wantsDraw());
}

TEST_CASE("Cancelled resize frames still initialize hidden framebuffers",
          "[core][render][surface][visibility]") {
    TestWindow window;
    auto surface = std::make_shared<TestSurface>(Render::Surface::Config{.retained = true});
    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.bind(surface) == Result::SUCCESS);
    REQUIRE(window.start() == Result::SUCCESS);
    REQUIRE(window.begin() == Result::SUCCESS);
    REQUIRE(surface->wantsDraw());
    REQUIRE(surface->wantsGraphics(true));
    REQUIRE(window.end() == Result::SUCCESS);
    REQUIRE_FALSE(surface->wantsDraw());

    REQUIRE(window.begin() == Result::SUCCESS);
    // prepare() recreates the framebuffer, but cancellation prevents draw().
    surface->recreateFramebuffer();
    REQUIRE(window.cancel() == Result::SUCCESS);

    REQUIRE(window.begin() == Result::SUCCESS);
    // The following prepare() no longer reports framebufferChanged.
    REQUIRE(surface->wantsDraw(false));
    REQUIRE(surface->wantsGraphics(false));
    REQUIRE(window.end() == Result::SUCCESS);
    REQUIRE_FALSE(surface->wantsDraw());

    surface->invalidate();
    REQUIRE(surface->wantsDraw());
    REQUIRE_FALSE(surface->wantsGraphics(false));
    surface->commitDraw();
    REQUIRE(window.stop() == Result::SUCCESS);
    REQUIRE(window.destroy() == Result::SUCCESS);
}

TEST_CASE("Skipped surface graphics still commit transfers and compute updates",
          "[core][render][transfer][surface][visibility]") {
    TestWindow window;
    std::array<U32, 2> source = {10, 20};
    auto buffer = MakeBuffer(source);
    Render::Kernel::Config kernelConfig;
    kernelConfig.buffers.push_back({buffer, Render::Kernel::AccessMode::READ});
    auto kernel = std::make_shared<TestKernel>(kernelConfig);
    Render::Surface::Config config;
    config.kernels.push_back(kernel);
    config.retained = true;
    auto surface = std::make_shared<TestSurface>(config);

    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.bind(surface) == Result::SUCCESS);
    REQUIRE(surface->wantsDraw());
    REQUIRE(surface->wantsGraphics(false));
    surface->commitDraw();

    REQUIRE(buffer->update() == Result::SUCCESS);
    kernel->update();
    Render::Transfer::Batch batch;
    REQUIRE(window.collectFrameTransfers(batch) == Result::SUCCESS);
    REQUIRE(batch.buffers().size() == 1);
    REQUIRE(surface->wantsDraw());
    REQUIRE(kernel->wantsDispatch());
    kernel->scheduled();
    REQUIRE_FALSE(surface->wantsGraphics(false));
    surface->commitDraw();
    batch.commit();

    Render::Transfer::Batch nextFrame;
    REQUIRE(window.collectFrameTransfers(nextFrame) == Result::SUCCESS);
    REQUIRE(nextFrame.empty());
    REQUIRE_FALSE(kernel->wantsDispatch());
    REQUIRE(surface->wantsDraw());
    REQUIRE(surface->wantsGraphics(true));
    surface->commitDraw();
    REQUIRE_FALSE(surface->wantsDraw());
    nextFrame.commit();
    REQUIRE(window.destroy() == Result::SUCCESS);
}

TEST_CASE("Shared transfers invalidate every retained consumer",
          "[core][render][transfer][surface]") {
    TestWindow window;
    std::array<U32, 2> source = {10, 20};
    auto buffer = MakeBuffer(source);
    auto firstSurface = MakeRetainedSurface(buffer);
    auto secondSurface = MakeRetainedSurface(buffer);

    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.bind(firstSurface) == Result::SUCCESS);
    REQUIRE(window.bind(secondSurface) == Result::SUCCESS);

    REQUIRE(firstSurface->wantsDraw());
    REQUIRE(secondSurface->wantsDraw());
    firstSurface->commitDraw();
    secondSurface->commitDraw();

    REQUIRE(buffer->update() == Result::SUCCESS);
    Render::Transfer::Batch batch;
    REQUIRE(window.collectFrameTransfers(batch) == Result::SUCCESS);
    REQUIRE(batch.buffers().size() == 1);
    REQUIRE(firstSurface->wantsDraw());
    REQUIRE(secondSurface->wantsDraw());

    firstSurface->commitDraw();
    secondSurface->commitDraw();
    batch.commit();
    REQUIRE(window.destroy() == Result::SUCCESS);
}

TEST_CASE("Failed transfer submission leaves retained surfaces dirty",
          "[core][render][transfer][surface]") {
    TestWindow window;
    std::array<U32, 2> source = {10, 20};
    auto buffer = MakeBuffer(source);
    auto surface = MakeRetainedSurface(buffer);

    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.bind(surface) == Result::SUCCESS);
    REQUIRE(surface->wantsDraw());
    surface->commitDraw();

    REQUIRE(buffer->update() == Result::SUCCESS);
    REQUIRE(window.start() == Result::SUCCESS);
    window.nextSubmissionResult(Result::ERROR);
    REQUIRE(window.begin() == Result::SUCCESS);
    REQUIRE(window.end() == Result::ERROR);
    REQUIRE(surface->wantsDraw());

    window.nextSubmissionResult(Result::SUCCESS);
    REQUIRE(window.begin() == Result::SUCCESS);
    REQUIRE(window.end() == Result::SUCCESS);
    REQUIRE_FALSE(surface->wantsDraw());
    REQUIRE(window.stop() == Result::SUCCESS);
    REQUIRE(window.destroy() == Result::SUCCESS);
}

TEST_CASE("Program state invalidation preserves explicit compute invalidation",
          "[core][render][transfer][surface]") {
    TestWindow window;
    std::array<U32, 2> source = {10, 20};
    auto buffer = MakeBuffer(source);

    Render::Kernel::Config kernelConfig;
    kernelConfig.buffers.push_back({buffer, Render::Kernel::AccessMode::READ});
    auto kernel = std::make_shared<Render::Kernel>(kernelConfig);
    auto draw = std::make_shared<TestDraw>(Render::Draw::Config{});
    Render::Program::Config programConfig;
    programConfig.draws.push_back(draw);
    auto program = std::make_shared<Render::Program>(programConfig);

    Render::Surface::Config surfaceConfig;
    surfaceConfig.kernels.push_back(kernel);
    surfaceConfig.programs.push_back(program);
    surfaceConfig.retained = true;
    auto surface = std::make_shared<TestSurface>(surfaceConfig);

    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.bind(surface) == Result::SUCCESS);
    REQUIRE(surface->wantsDraw());
    surface->commitDraw();

    kernel->update();
    REQUIRE(draw->updateVertexCount(1) == Result::SUCCESS);

    Render::Transfer::Batch batch;
    REQUIRE(window.collectFrameTransfers(batch) == Result::SUCCESS);
    REQUIRE(batch.empty());
    REQUIRE_FALSE(surface->wantsDraw());

    program->scissorRect(Render::ScissorRect{0, 0, 1, 1});
    REQUIRE(surface->wantsDraw());
    surface->commitDraw();
    REQUIRE_FALSE(surface->wantsDraw());

    surface->invalidate();
    REQUIRE(surface->wantsDraw());
    surface->commitDraw();
    batch.commit();
    REQUIRE(window.destroy() == Result::SUCCESS);
}

TEST_CASE("Render program state remains local and mutable", "[core][render][program]") {
    Render::Program program(Render::Program::Config{});
    Render::Program other(Render::Program::Config{});

    REQUIRE(program.enabled());
    program.setEnabled(false);
    REQUIRE_FALSE(program.enabled());
    REQUIRE(other.enabled());
    program.setEnabled(true);
    REQUIRE(program.enabled());

    const Render::ScissorRect scissor{1, 2, 3, 4};
    program.scissorRect(scissor);
    REQUIRE(program.getConfig().scissorRect.has_value());
    REQUIRE(program.getConfig().scissorRect->x == 1);
    REQUIRE(program.getConfig().scissorRect->y == 2);
    REQUIRE(program.getConfig().scissorRect->width == 3);
    REQUIRE(program.getConfig().scissorRect->height == 4);

    program.scissorRect(std::nullopt);
    REQUIRE_FALSE(program.getConfig().scissorRect.has_value());
}

TEST_CASE("Window clear colors expire at the next frame", "[core][render][window]") {
    TestWindow window;
    bool cancelled = false;
    bool failed = false;
    SECTION("Completed frame") {}
    SECTION("Cancelled frame") { cancelled = true; }
    SECTION("Failed submission") { failed = true; }
    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.start() == Result::SUCCESS);
    REQUIRE(window.begin() == Result::SUCCESS);
    window.frameClearColor({0.2f, 0.4f, 0.6f, 1.0f});
    REQUIRE(window.frameClearColor().r == 0.2f);
    if (failed) window.nextSubmissionResult(Result::ERROR);
    if (cancelled) REQUIRE(window.cancel() == Result::SUCCESS);
    else REQUIRE(window.end() == (failed ? Result::ERROR : Result::SUCCESS));
    window.nextSubmissionResult(Result::SUCCESS);
    REQUIRE(window.begin() == Result::SUCCESS);
    REQUIRE(window.frameClearColor().r == 0.0f);
    REQUIRE(window.frameClearColor().g == 0.0f);
    REQUIRE(window.frameClearColor().b == 0.0f);
    REQUIRE(window.frameClearColor().a == 1.0f);
    REQUIRE(window.end() == Result::SUCCESS);
    REQUIRE(window.stop() == Result::SUCCESS);
    REQUIRE(window.destroy() == Result::SUCCESS);
}

TEST_CASE("Surface visibility hints combine views and expire each frame", "[core][render][surface]") {
    TestTexture texture(Render::Texture::Config{.size = {2, 2}});
    REQUIRE(texture.visibleForPresentation(1));
    texture.presentationHint(1, false);
    REQUIRE_FALSE(texture.visibleForPresentation(1));
    texture.presentationHint(1, true);
    texture.presentationHint(1, false);
    REQUIRE(texture.visibleForPresentation(1));
    REQUIRE(texture.visibleForPresentation(2));
    texture.presentationHint(2, false);
    REQUIRE_FALSE(texture.visibleForPresentation(2));
    REQUIRE(texture.visibleForPresentation(3));
}

TEST_CASE("Surfaces collect their complete transfer dependency graph",
          "[core][render][transfer][surface]") {
    TestWindow window;
    std::array<U32, 2> source = {10, 20};
    auto programBuffer = MakeBuffer(source);
    auto vertexBuffer = MakeBuffer(source);
    auto indexBuffer = MakeBuffer(source);
    auto instanceBuffer = MakeBuffer(source);
    auto indirectBuffer = MakeBuffer(source);
    auto kernelBuffer = MakeBuffer(source);

    std::array<U8, 16> textureSource{};
    auto texture = std::make_shared<TestTexture>(Render::Texture::Config{
        .size = {2, 2},
        .buffer = textureSource.data(),
    });

    auto vertex = std::make_shared<Render::Vertex>(Render::Vertex::Config{
        .vertices = {{vertexBuffer, 0}},
        .instances = {{instanceBuffer, 0}},
        .indices = indexBuffer,
    });
    auto draw = std::make_shared<TestDraw>(Render::Draw::Config{.buffer = vertex});

    Render::Program::Config programConfig;
    programConfig.draws.push_back(draw);
    programConfig.textures.push_back(texture);
    programConfig.buffers.push_back({programBuffer, Render::Program::Target::VERTEX});
    auto program = std::make_shared<Render::Program>(programConfig);

    Render::Kernel::Config kernelConfig;
    kernelConfig.buffers.push_back({kernelBuffer, Render::Kernel::AccessMode::READ});
    auto kernel = std::make_shared<Render::Kernel>(kernelConfig);

    Render::Surface::Config surfaceConfig;
    surfaceConfig.kernels.push_back(kernel);
    surfaceConfig.programs.push_back(program);
    surfaceConfig.retained = true;
    auto surface = std::make_shared<TestSurface>(surfaceConfig);

    // Device Draw implementations create indirect buffers after Surface construction.
    draw->addTransferBuffer(indirectBuffer);

    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.bind(surface) == Result::SUCCESS);
    REQUIRE(surface->wantsDraw());
    surface->commitDraw();

    const auto checkBufferDependency = [&](const std::shared_ptr<TestBuffer>& buffer) {
        REQUIRE(buffer->update() == Result::SUCCESS);
        Render::Transfer::Batch batch;
        REQUIRE(window.collectFrameTransfers(batch) == Result::SUCCESS);
        REQUIRE(batch.buffers().size() == 1);
        REQUIRE(batch.textures().empty());
        REQUIRE(surface->wantsDraw());
        surface->commitDraw();
        batch.commit();
        REQUIRE_FALSE(surface->wantsDraw());
    };

    checkBufferDependency(programBuffer);
    checkBufferDependency(vertexBuffer);
    checkBufferDependency(indexBuffer);
    checkBufferDependency(instanceBuffer);
    checkBufferDependency(indirectBuffer);
    checkBufferDependency(kernelBuffer);

    REQUIRE(texture->fill() == Result::SUCCESS);
    Render::Transfer::Batch textureBatch;
    REQUIRE(window.collectFrameTransfers(textureBatch) == Result::SUCCESS);
    REQUIRE(textureBatch.buffers().empty());
    REQUIRE(textureBatch.textures().size() == 1);
    REQUIRE(surface->wantsDraw());
    surface->commitDraw();
    textureBatch.commit();
    REQUIRE(window.destroy() == Result::SUCCESS);
}

TEST_CASE("Drawn surface graphics advance the framebuffer content revision", "[core][render][surface]") {
    auto framebuffer = std::make_shared<TestTexture>(Render::Texture::Config{.size = {2, 2}});
    auto surface = MakeFramebufferSurface(framebuffer);
    REQUIRE(surface->create() == Result::SUCCESS);
    REQUIRE(framebuffer->contentRevision() == 0);

    REQUIRE(surface->wantsDraw());
    REQUIRE(surface->wantsGraphics(false));
    REQUIRE(framebuffer->contentRevision() == 1);
    surface->commitDraw();

    REQUIRE_FALSE(surface->wantsDraw());
    REQUIRE(framebuffer->contentRevision() == 1);

    surface->invalidate();
    REQUIRE(surface->wantsDraw());
    REQUIRE_FALSE(surface->wantsGraphics(false));
    surface->commitDraw();
    REQUIRE(framebuffer->contentRevision() == 1);

    REQUIRE(surface->wantsDraw());
    REQUIRE(surface->wantsGraphics(true));
    surface->commitDraw();
    REQUIRE(framebuffer->contentRevision() == 2);

    REQUIRE(surface->wantsDraw(true));
    REQUIRE(surface->wantsGraphics(false));
    surface->commitDraw();
    REQUIRE(framebuffer->contentRevision() == 3);
}

TEST_CASE("Retained surfaces redraw when borrowed textures change content", "[core][render][surface]") {
    auto producedTexture = std::make_shared<TestTexture>(Render::Texture::Config{.size = {2, 2}});
    auto otherTexture = std::make_shared<TestTexture>(Render::Texture::Config{.size = {2, 2}});
    REQUIRE(producedTexture->create() == Result::SUCCESS);
    REQUIRE(otherTexture->create() == Result::SUCCESS);
    auto producer = MakeFramebufferSurface(producedTexture);
    REQUIRE(producer->create() == Result::SUCCESS);

    Render::Program::Config programConfig;
    programConfig.textures.push_back(producedTexture);
    programConfig.textures.push_back(otherTexture);
    programConfig.borrowedTextures = true;

    Render::Surface::Config consumerConfig;
    consumerConfig.programs.push_back(std::make_shared<Render::Program>(programConfig));
    consumerConfig.retained = true;
    auto consumer = std::make_shared<TestSurface>(consumerConfig);

    REQUIRE(consumer->wantsDraw());
    consumer->commitDraw();
    REQUIRE_FALSE(consumer->wantsDraw());

    REQUIRE(producer->wantsDraw());
    REQUIRE(producer->wantsGraphics(true));
    producer->commitDraw();
    REQUIRE(consumer->wantsDraw());
    consumer->commitDraw();
    REQUIRE_FALSE(consumer->wantsDraw());

    producer->invalidate();
    REQUIRE(producer->wantsDraw());
    REQUIRE_FALSE(producer->wantsGraphics(false));
    producer->commitDraw();
    REQUIRE_FALSE(consumer->wantsDraw());

    REQUIRE(producer->wantsDraw());
    REQUIRE(producer->wantsGraphics(true));
    producer->commitDraw();
    REQUIRE(consumer->wantsDraw());
    REQUIRE(consumer->wantsDraw());
    consumer->commitDraw();
    REQUIRE_FALSE(consumer->wantsDraw());

    otherTexture->markContentChanged();
    REQUIRE(consumer->wantsDraw());
    consumer->commitDraw();
    REQUIRE_FALSE(consumer->wantsDraw());
}

TEST_CASE("Sampled textures that are not borrowed do not trigger content redraws", "[core][render][surface]") {
    auto texture = std::make_shared<TestTexture>(Render::Texture::Config{.size = {2, 2}});
    auto borrowedTexture = std::make_shared<TestTexture>(Render::Texture::Config{.size = {2, 2}});
    REQUIRE(borrowedTexture->create() == Result::SUCCESS);

    auto sampled = MakeRetainedSurface(std::static_pointer_cast<Render::Texture>(texture));
    REQUIRE(sampled->wantsDraw());
    sampled->commitDraw();
    texture->markContentChanged();
    REQUIRE_FALSE(sampled->wantsDraw());

    Render::Program::Config borrowedConfig;
    borrowedConfig.textures.push_back(borrowedTexture);
    borrowedConfig.borrowedTextures = true;

    Render::Program::Config ownedConfig;
    ownedConfig.textures.push_back(texture);

    Render::Surface::Config config;
    config.programs.push_back(std::make_shared<Render::Program>(ownedConfig));
    config.programs.push_back(std::make_shared<Render::Program>(borrowedConfig));
    config.retained = true;
    auto mixed = std::make_shared<TestSurface>(config);

    REQUIRE(mixed->wantsDraw());
    mixed->commitDraw();
    texture->markContentChanged();
    REQUIRE_FALSE(mixed->wantsDraw());
    borrowedTexture->markContentChanged();
    REQUIRE(mixed->wantsDraw());
    mixed->commitDraw();
    REQUIRE_FALSE(mixed->wantsDraw());
}

TEST_CASE("Disabled borrowed programs do not redraw their surface when the producer draws",
          "[core][render][surface][visibility]") {
    auto hiddenTexture = std::make_shared<TestTexture>(Render::Texture::Config{.size = {2, 2}});
    auto shownTexture = std::make_shared<TestTexture>(Render::Texture::Config{.size = {2, 2}});
    REQUIRE(hiddenTexture->create() == Result::SUCCESS);
    REQUIRE(shownTexture->create() == Result::SUCCESS);
    auto hiddenProducer = MakeFramebufferSurface(hiddenTexture);
    auto shownProducer = MakeFramebufferSurface(shownTexture);
    REQUIRE(hiddenProducer->create() == Result::SUCCESS);
    REQUIRE(shownProducer->create() == Result::SUCCESS);
    const auto produce = [](const std::shared_ptr<TestSurface>& producer) {
        producer->invalidate();
        REQUIRE(producer->wantsDraw());
        REQUIRE(producer->wantsGraphics(true));
        producer->commitDraw();
    };

    Render::Program::Config hiddenConfig;
    hiddenConfig.textures.push_back(hiddenTexture);
    hiddenConfig.borrowedTextures = true;
    auto hiddenProgram = std::make_shared<Render::Program>(hiddenConfig);

    Render::Program::Config shownConfig;
    shownConfig.textures.push_back(shownTexture);
    shownConfig.borrowedTextures = true;
    auto shownProgram = std::make_shared<Render::Program>(shownConfig);

    Render::Surface::Config consumerConfig;
    consumerConfig.programs.push_back(hiddenProgram);
    consumerConfig.programs.push_back(shownProgram);
    consumerConfig.retained = true;
    auto consumer = std::make_shared<TestSurface>(consumerConfig);

    REQUIRE(consumer->wantsDraw());
    consumer->commitDraw();
    REQUIRE_FALSE(consumer->wantsDraw());

    hiddenProgram->setEnabled(false);
    REQUIRE(consumer->wantsDraw());
    consumer->commitDraw();
    REQUIRE_FALSE(consumer->wantsDraw());

    for (int frame = 0; frame < 3; ++frame) {
        produce(hiddenProducer);
        REQUIRE_FALSE(consumer->wantsDraw());
    }

    produce(shownProducer);
    REQUIRE(consumer->wantsDraw());
    consumer->commitDraw();
    REQUIRE_FALSE(consumer->wantsDraw());

    hiddenProgram->setEnabled(true);
    REQUIRE(consumer->wantsDraw());
    consumer->commitDraw();
    REQUIRE_FALSE(consumer->wantsDraw());

    produce(hiddenProducer);
    REQUIRE(consumer->wantsDraw());
    consumer->commitDraw();
    REQUIRE_FALSE(consumer->wantsDraw());
}

TEST_CASE("Hidden retained surfaces are idle only once initialized and without pending work",
          "[core][render][surface][visibility]") {
    TestWindow window;
    std::array<U32, 2> source = {10, 20};
    auto buffer = MakeBuffer(source);
    Render::Kernel::Config kernelConfig;
    kernelConfig.buffers.push_back({buffer, Render::Kernel::AccessMode::READ});
    auto kernel = std::make_shared<TestKernel>(kernelConfig);
    Render::Surface::Config config;
    config.kernels.push_back(kernel);
    config.retained = true;
    auto surface = std::make_shared<TestSurface>(config);

    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.bind(surface) == Result::SUCCESS);
    REQUIRE_FALSE(surface->idleWhileHidden(false));
    REQUIRE(surface->wantsDraw());
    REQUIRE(surface->wantsGraphics(false));
    surface->commitDraw();

    REQUIRE(surface->idleWhileHidden(false));
    REQUIRE_FALSE(surface->idleWhileHidden(true));
    REQUIRE_FALSE(surface->idleWhileHidden(false, true));

    surface->invalidate();
    REQUIRE(surface->wantsDraw());
    REQUIRE_FALSE(surface->wantsGraphics(false));
    surface->commitDraw();
    REQUIRE(surface->idleWhileHidden(false));

    REQUIRE(buffer->update() == Result::SUCCESS);
    kernel->update();
    Render::Transfer::Batch batch;
    REQUIRE(window.collectFrameTransfers(batch) == Result::SUCCESS);
    REQUIRE(kernel->wantsDispatch());
    REQUIRE_FALSE(surface->idleWhileHidden(false));
    REQUIRE(surface->wantsDraw());
    kernel->scheduled();
    REQUIRE_FALSE(surface->wantsGraphics(false));
    surface->commitDraw();
    batch.commit();

    Render::Transfer::Batch nextFrame;
    REQUIRE(window.collectFrameTransfers(nextFrame) == Result::SUCCESS);
    REQUIRE_FALSE(kernel->wantsDispatch());
    REQUIRE(surface->idleWhileHidden(false));
    nextFrame.commit();

    surface->recreateFramebuffer();
    REQUIRE_FALSE(surface->idleWhileHidden(false));
    REQUIRE(surface->wantsDraw());
    REQUIRE(surface->wantsGraphics(false));
    surface->commitDraw();
    REQUIRE(surface->idleWhileHidden(false));
    REQUIRE(window.destroy() == Result::SUCCESS);

    TestSurface immediate(Render::Surface::Config{});
    REQUIRE(immediate.wantsDraw());
    REQUIRE(immediate.wantsGraphics(false));
    immediate.commitDraw();
    REQUIRE_FALSE(immediate.idleWhileHidden(false));
}

TEST_CASE("Borrowed programs rebind once after their texture is reallocated, even at the same size or while hidden",
          "[core][render][surface]") {
    auto texture = std::make_shared<TestTexture>(Render::Texture::Config{.size = {64, 32}});
    REQUIRE(texture->create() == Result::SUCCESS);

    Render::Program::Config programConfig;
    programConfig.textures = {texture};
    programConfig.borrowedTextures = true;
    auto program = std::make_shared<RebindingProgram>(programConfig);

    Render::Surface::Config config;
    config.retained = true;
    config.programs.push_back(program);
    TestSurface surface(config);
    REQUIRE(surface.create() == Result::SUCCESS);

    REQUIRE(surface.wantsDraw());
    REQUIRE(program->rebinds == 0);
    surface.commitDraw();
    REQUIRE_FALSE(surface.wantsDraw());

    REQUIRE(texture->destroy() == Result::SUCCESS);
    REQUIRE(texture->create() == Result::SUCCESS);
    REQUIRE(surface.wantsDraw());
    REQUIRE(program->rebinds == 1);
    surface.commitDraw();
    REQUIRE_FALSE(surface.wantsDraw());
    REQUIRE(program->rebinds == 1);

    program->setEnabled(false);
    REQUIRE(surface.wantsDraw());
    surface.commitDraw();
    REQUIRE_FALSE(surface.wantsDraw());
    REQUIRE(texture->destroy() == Result::SUCCESS);
    REQUIRE(texture->create() == Result::SUCCESS);
    REQUIRE_FALSE(surface.wantsDraw());
    REQUIRE(program->rebinds == 1);
    program->setEnabled(true);
    REQUIRE(surface.wantsDraw());
    REQUIRE(program->rebinds == 2);
    surface.commitDraw();
    REQUIRE_FALSE(surface.wantsDraw());
}

TEST_CASE("Borrowed programs rebind when their texture is reallocated before the first draw or while disabled",
          "[core][render][surface]") {
    auto texture = std::make_shared<TestTexture>(Render::Texture::Config{.size = {64, 32}});
    REQUIRE(texture->create() == Result::SUCCESS);

    Render::Program::Config programConfig;
    programConfig.textures = {texture};
    programConfig.borrowedTextures = true;
    auto program = std::make_shared<RebindingProgram>(programConfig);
    program->setEnabled(false);

    Render::Surface::Config config;
    config.retained = true;
    config.programs.push_back(program);
    TestSurface surface(config);
    REQUIRE(surface.create() == Result::SUCCESS);

    REQUIRE(texture->create() == Result::SUCCESS);
    surface.wantsDraw();
    REQUIRE(program->rebinds == 0);

    program->setEnabled(true);
    surface.wantsDraw();
    REQUIRE(program->rebinds == 1);
    surface.wantsDraw();
    REQUIRE(program->rebinds == 1);
}

TEST_CASE("Borrowers preserve texture allocations after the producer is unbound",
          "[core][render][window][surface]") {
    TestWindow window;
    auto texture = std::make_shared<TestTexture>(Render::Texture::Config{.size = {2, 2}});
    auto producer = MakeFramebufferSurface(texture);
    bool standalone = false;
    SECTION("Framebuffer producer") {}
    SECTION("Standalone producer") { standalone = true; }
    auto borrower = MakeBorrowedSurface(texture);
    const std::shared_ptr<Render::WindowAttachment> source = standalone ?
        std::static_pointer_cast<Render::WindowAttachment>(texture) : producer;
    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.bind(source) == Result::SUCCESS);
    REQUIRE(window.bind(borrower) == Result::SUCCESS);
    REQUIRE(texture->createCount == 1);
    REQUIRE(window.start() == Result::SUCCESS);
    REQUIRE(window.unbind(source) == Result::SUCCESS);
    for (int frame = 0; frame < 8; ++frame) {
        REQUIRE(window.begin() == Result::SUCCESS);
        REQUIRE(window.end() == Result::SUCCESS);
    }
    REQUIRE(texture->destroyCount == 0);
    REQUIRE(texture->allocated());
    texture->markContentChanged();
    REQUIRE(borrower->wantsDraw());
    borrower->commitDraw();
    REQUIRE_FALSE(borrower->wantsDraw());
    REQUIRE(window.unbind(borrower) == Result::SUCCESS);
    REQUIRE(window.stop() == Result::SUCCESS);
    REQUIRE(window.destroy() == Result::SUCCESS);
    REQUIRE(texture->destroyCount == 1);
    REQUIRE_FALSE(texture->allocated());
}

TEST_CASE("Borrowers reject unbound foreign and retiring texture sources without allocation",
          "[core][render][window][surface]") {
    TestWindow window;
    TestWindow foreign;
    auto texture = std::make_shared<TestTexture>(Render::Texture::Config{.size = {2, 2}});
    auto borrower = MakeBorrowedSurface(texture);
    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(foreign.create() == Result::SUCCESS);
    U64 expectedCreations = 0;
    SECTION("Unbound source") {}
    SECTION("Foreign source") {
        REQUIRE(foreign.bind(texture) == Result::SUCCESS);
        expectedCreations = 1;
    }
    SECTION("Retiring source") {
        REQUIRE(window.bind(texture) == Result::SUCCESS);
        REQUIRE(window.unbind(texture) == Result::SUCCESS);
        expectedCreations = 1;
    }
    REQUIRE(window.bind(borrower) == Result::ERROR);
    REQUIRE(texture->createCount == expectedCreations);
    REQUIRE(window.destroy() == Result::SUCCESS);
    REQUIRE(foreign.destroy() == Result::SUCCESS);
    REQUIRE(texture->destroyCount == expectedCreations);
}

TEST_CASE("Failed borrowed texture refresh aborts the frame without acknowledging invalidation",
          "[core][render][surface][window]") {
    TestWindow window;
    auto texture = std::make_shared<TestTexture>(Render::Texture::Config{.size = {2, 2}});
    auto borrower = MakeBorrowedSurface(texture);
    auto program = std::dynamic_pointer_cast<RebindingProgram>(borrower->getConfig().programs.front());
    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.bind(texture) == Result::SUCCESS);
    REQUIRE(window.bind(borrower) == Result::SUCCESS);
    REQUIRE(window.start() == Result::SUCCESS);
    REQUIRE(window.begin() == Result::SUCCESS);
    REQUIRE(window.end() == Result::SUCCESS);
    REQUIRE_FALSE(borrower->wantsDraw());
    const U64 rebinds = program->rebinds;
    REQUIRE(texture->destroy() == Result::SUCCESS);
    REQUIRE(texture->create() == Result::SUCCESS);
    program->rebindResult = Result::ERROR;
    REQUIRE(window.begin() == Result::SUCCESS);
    REQUIRE(window.end() == Result::ERROR);
    REQUIRE(borrower->bindingResult() == Result::ERROR);
    borrower->commitDraw();
    REQUIRE(program->rebinds == rebinds + 1);
    REQUIRE(borrower->wantsDraw());
    REQUIRE(program->rebinds == rebinds + 2);
    REQUIRE(borrower->bindingResult() == Result::SUCCESS);
    REQUIRE(borrower->wantsDraw());
    REQUIRE(window.begin() == Result::SUCCESS);
    REQUIRE(window.end() == Result::SUCCESS);
    REQUIRE_FALSE(borrower->wantsDraw());
    REQUIRE(window.stop() == Result::SUCCESS);
    REQUIRE(window.destroy() == Result::SUCCESS);
}

TEST_CASE("Deallocated borrowed textures cannot acknowledge or encode stale bindings",
          "[core][render][surface]") {
    auto texture = std::make_shared<TestTexture>(Render::Texture::Config{.size = {2, 2}});
    REQUIRE(texture->create() == Result::SUCCESS);
    auto borrower = MakeBorrowedSurface(texture);
    REQUIRE(borrower->wantsDraw());
    borrower->commitDraw();
    REQUIRE_FALSE(borrower->wantsDraw());
    const U64 allocation = texture->allocation();
    REQUIRE(texture->destroy() == Result::SUCCESS);
    REQUIRE(texture->allocation() != allocation);
    REQUIRE_FALSE(borrower->wantsDraw());
    REQUIRE(borrower->bindingResult() == Result::ERROR);
    borrower->commitDraw();
    REQUIRE(texture->create() == Result::SUCCESS);
    REQUIRE(borrower->wantsDraw());
    borrower->commitDraw();
    REQUIRE_FALSE(borrower->wantsDraw());
}

TEST_CASE("Borrowed texture uploads invalidate consumers only when collected or committed",
          "[core][render][transfer][surface]") {
    TestWindow window;
    std::array<U8, 16> source{};
    auto texture = std::make_shared<TestTexture>(Render::Texture::Config{
        .size = {2, 2}, .buffer = source.data(),
    });
    auto borrower = MakeBorrowedSurface(texture);
    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.bind(texture) == Result::SUCCESS);
    REQUIRE(window.bind(borrower) == Result::SUCCESS);
    REQUIRE(borrower->wantsDraw());
    borrower->commitDraw();
    REQUIRE_FALSE(borrower->wantsDraw());
    REQUIRE(texture->fill() == Result::SUCCESS);
    REQUIRE(texture->contentRevision() == 0);
    {
        Render::Transfer::Batch aborted;
        REQUIRE(window.collectFrameTransfers(aborted) == Result::SUCCESS);
        REQUIRE(aborted.textures().size() == 1);
        REQUIRE(borrower->wantsDraw());
    }
    REQUIRE(texture->contentRevision() == 0);
    Render::Transfer::Batch retry;
    REQUIRE(window.collectFrameTransfers(retry) == Result::SUCCESS);
    REQUIRE(retry.textures().size() == 1);
    REQUIRE(borrower->wantsDraw());
    borrower->commitDraw();
    REQUIRE_FALSE(borrower->wantsDraw());
    source.fill(42);
    REQUIRE(texture->fillRow(1, 1) == Result::SUCCESS);
    retry.commit();
    REQUIRE(texture->contentRevision() == 1);
    REQUIRE(borrower->wantsDraw());
    borrower->commitDraw();
    REQUIRE_FALSE(borrower->wantsDraw());
    Render::Transfer::Batch next;
    REQUIRE(window.collectFrameTransfers(next) == Result::SUCCESS);
    REQUIRE(next.textures().size() == 1);
    REQUIRE(next.textures().front().destinationRow == 1);
    REQUIRE(borrower->wantsDraw());
    next.commit();
    REQUIRE(texture->contentRevision() == 2);
    REQUIRE(borrower->wantsDraw());
    borrower->commitDraw();
    REQUIRE_FALSE(borrower->wantsDraw());
    REQUIRE(window.destroy() == Result::SUCCESS);
}

TEST_CASE("Failed texture upload submission preserves borrowed redraw and content revisions",
          "[core][render][transfer][surface]") {
    TestWindow window;
    std::array<U8, 16> source{};
    auto texture = std::make_shared<TestTexture>(Render::Texture::Config{
        .size = {2, 2}, .buffer = source.data(),
    });
    auto borrower = MakeBorrowedSurface(texture);
    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(window.bind(texture) == Result::SUCCESS);
    REQUIRE(window.bind(borrower) == Result::SUCCESS);
    REQUIRE(window.start() == Result::SUCCESS);
    REQUIRE(window.begin() == Result::SUCCESS);
    REQUIRE(window.end() == Result::SUCCESS);
    REQUIRE_FALSE(borrower->wantsDraw());
    REQUIRE(texture->fill() == Result::SUCCESS);
    window.nextSubmissionResult(Result::ERROR);
    REQUIRE(window.begin() == Result::SUCCESS);
    REQUIRE(window.end() == Result::ERROR);
    REQUIRE(texture->contentRevision() == 0);
    REQUIRE(borrower->wantsDraw());
    window.nextSubmissionResult(Result::SUCCESS);
    REQUIRE(window.begin() == Result::SUCCESS);
    REQUIRE(window.end() == Result::SUCCESS);
    REQUIRE(texture->contentRevision() == 1);
    REQUIRE(borrower->wantsDraw());
    borrower->commitDraw();
    REQUIRE_FALSE(borrower->wantsDraw());
    REQUIRE(window.stop() == Result::SUCCESS);
    REQUIRE(window.destroy() == Result::SUCCESS);
}

TEST_CASE("Texture revisions can be sampled while their producer reallocates",
          "[core][render][program][threading]") {
    auto texture = std::make_shared<TestTexture>(Render::Texture::Config{.size = {2, 2}});
    Render::Program::Config config;
    config.textures.push_back(texture);
    std::atomic<bool> finished = false;
    std::thread producer([&] {
        for (int iteration = 0; iteration < 10000; ++iteration) {
            texture->create();
            texture->markContentChanged();
            texture->destroy();
        }
        finished.store(true, std::memory_order_release);
    });
    U64 previousAllocation = 0;
    U64 previousContent = 0;
    bool monotonic = true;
    do {
        Render::Program program(config);
        const U64 allocation = program.textureAllocations().front();
        const U64 content = texture->contentRevision();
        monotonic &= allocation >= previousAllocation && content >= previousContent;
        previousAllocation = allocation;
        previousContent = content;
    } while (!finished.load(std::memory_order_acquire));
    producer.join();
    REQUIRE(monotonic);
    REQUIRE(texture->allocation() == 20000);
    REQUIRE(texture->contentRevision() == 10000);
}
