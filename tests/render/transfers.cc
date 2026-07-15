#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

#include "jetstream/render/base/buffer.hh"
#include "jetstream/render/base/surface.hh"
#include "jetstream/render/base/transfer.hh"
#include "jetstream/render/base/window.hh"

using namespace Jetstream;

namespace {

class TestBuffer final : public Render::Buffer {
 public:
    explicit TestBuffer(const Config& config) : Buffer(config) {}

    Result create() override { return Result::SUCCESS; }
    Result destroy() override { return Result::SUCCESS; }

};

class TestTexture final : public Render::Texture {
 public:
    explicit TestTexture(const Config& config) : Texture(config) {}

    Result create() override { return Result::SUCCESS; }
    Result destroy() override { return Result::SUCCESS; }
    uint64_t raw() const override { return 0; }
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

class TestSurface final : public Render::Surface {
 public:
    explicit TestSurface(const Config& config) : Surface(config) {}

    Result create() override { return Result::SUCCESS; }
    Result destroy() override { return Result::SUCCESS; }
    const Extent2D<U64>& size(const Extent2D<U64>&) override { return NullSize2D; }

    bool wantsDraw() { return shouldDraw(); }
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

    U64 destructionCount() const {
        return destructions;
    }

 protected:
    Result bindSurface(const std::shared_ptr<Render::Surface>& surface) override {
        surfaces.push_back(std::dynamic_pointer_cast<TestSurface>(surface));
        return Result::SUCCESS;
    }
    Result unbindSurface(const std::shared_ptr<Render::Surface>& surface) override {
        std::erase(surfaces, std::dynamic_pointer_cast<TestSurface>(surface));
        return Result::SUCCESS;
    }
    Result underlyingCreate() override { return creationResult; }
    Result underlyingDestroy() override {
        ++destructions;
        return Result::SUCCESS;
    }
    Result underlyingBegin() override { return Result::SUCCESS; }
    Result underlyingEnd() override {
        Render::Transfer::Batch batch;
        JST_CHECK(collectTransfers(batch));

        for (const auto& surface : surfaces) {
            surface->wantsDraw();
        }

        if (submissionResult != Result::SUCCESS) {
            return submissionResult;
        }

        for (const auto& surface : surfaces) {
            surface->commitDraw();
        }
        batch.commit();
        return Result::SUCCESS;
    }
    Result underlyingSynchronize() override { return Result::SUCCESS; }

 private:
    Stats statsData{};
    Result creationResult = Result::SUCCESS;
    Result submissionResult = Result::SUCCESS;
    U64 destructions = 0;
    std::vector<std::shared_ptr<TestSurface>> surfaces;
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

}  // namespace

TEST_CASE("Render transfer ranges align and reject overflow", "[render][transfer]") {
    U64 used = 3;
    U64 offset = 0;

    REQUIRE(TestTransfer::reserve(used, 4, 4, offset));
    REQUIRE(offset == 4);
    REQUIRE(used == 8);
    REQUIRE_FALSE(TestTransfer::reserve(used, 1, 0, offset));

    used = std::numeric_limits<U64>::max();
    REQUIRE_FALSE(TestTransfer::reserve(used, 1, 4, offset));
}

TEST_CASE("Render transfer capacities grow and align", "[render][transfer]") {
    U64 capacity = 0;

    REQUIRE(TestTransfer::capacity(1, 4, capacity));
    REQUIRE(capacity == TestTransfer::minimumBufferSize());

    REQUIRE(TestTransfer::capacity(TestTransfer::minimumBufferSize() + 1, 4, capacity));
    REQUIRE(capacity == 2 * TestTransfer::minimumBufferSize());

    REQUIRE_FALSE(TestTransfer::capacity(1, 0, capacity));
    REQUIRE_FALSE(TestTransfer::capacity(std::numeric_limits<U64>::max(), 4, capacity));
}

TEST_CASE("Render transfer rows are padded when copied", "[render][transfer]") {
    std::array<U8, 6> source = {1, 2, 3, 4, 5, 6};
    std::array<U8, 8> destination{};
    const std::array<U8, 8> expected = {1, 2, 3, 0, 4, 5, 6, 0};
    U64 rowSize = 0;

    REQUIRE(TestTransfer::alignedSize(3, 4, rowSize));
    REQUIRE(rowSize == 4);
    REQUIRE(TestTransfer::copyRows(destination.data(), source.data(), 2, 3, rowSize));
    REQUIRE(destination == expected);
}

TEST_CASE("Render buffer transfers copy checked ranges", "[render][transfer]") {
    std::array<U8, 4> source = {1, 2, 3, 4};
    std::array<U8, 4> destination{};

    REQUIRE(TestTransfer::copy(destination.data(), source.data(), source.size()));
    REQUIRE(destination == source);
    REQUIRE(TestTransfer::copy(nullptr, nullptr, 0));
    REQUIRE_FALSE(TestTransfer::copy(nullptr, source.data(), 1));
}

TEST_CASE("Render transfer updates own their source data", "[render][transfer]") {
    Render::Transfer::PendingUploadQueue queue;
    std::array<U8, 8> source = {0, 1, 2, 3, 4, 5, 6, 7};

    REQUIRE(queue.queue(2, 4, source.size(), 1, source.data()) == Result::SUCCESS);
    source.fill(42);

    auto uploads = queue.take();
    REQUIRE(uploads.size() == 1);
    REQUIRE(uploads[0].start == 2);
    REQUIRE(uploads[0].data == std::vector<U8>{2, 3, 4, 5});
}

TEST_CASE("Render transfer updates coalesce with last write wins", "[render][transfer]") {
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

TEST_CASE("Render transfer updates keep wrapped ranges disjoint", "[render][transfer]") {
    Render::Transfer::PendingUploadQueue queue;
    std::array<U8, 16> source{};

    REQUIRE(queue.queue(0, 2, source.size(), 1, source.data()) == Result::SUCCESS);
    REQUIRE(queue.queue(14, 2, source.size(), 1, source.data()) == Result::SUCCESS);

    auto uploads = queue.take();
    REQUIRE(uploads.size() == 2);
    REQUIRE(uploads[0].start == 0);
    REQUIRE(uploads[1].start == 14);
}

TEST_CASE("Restored transfers preserve newer updates", "[render][transfer]") {
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

TEST_CASE("Restored transfers do not cross generations", "[render][transfer]") {
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

TEST_CASE("Malformed restored transfers are rejected", "[render][transfer]") {
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
          "[render][transfer]") {
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

TEST_CASE("Render buffer updates require four-byte alignment", "[render][transfer]") {
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

TEST_CASE("Render transfer updates validate ranges", "[render][transfer]") {
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

TEST_CASE("Render transfer updates preserve their unit size", "[render][transfer]") {
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
          "[render][transfer]") {
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

TEST_CASE("Standalone attachments do not make transfer collection order-dependent",
          "[render][transfer][surface]") {
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

TEST_CASE("Failed render window creation rolls back its backend", "[render][window]") {
    TestWindow failed;
    TestWindow replacement;

    failed.nextCreationResult(Result::ERROR);
    REQUIRE(failed.create() == Result::ERROR);
    REQUIRE(failed.destructionCount() == 1);

    REQUIRE(replacement.create() == Result::SUCCESS);
    REQUIRE(replacement.destroy() == Result::SUCCESS);
}

TEST_CASE("Cancelled render frames release the window", "[render][window]") {
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
          "[render][transfer][surface]") {
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

TEST_CASE("Shared transfers invalidate every retained consumer",
          "[render][transfer][surface]") {
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
          "[render][transfer][surface]") {
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

TEST_CASE("Semantic changes still require explicit retained invalidation",
          "[render][transfer][surface]") {
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
    program->scissorRect(Render::ScissorRect{0, 0, 1, 1});

    Render::Transfer::Batch batch;
    REQUIRE(window.collectFrameTransfers(batch) == Result::SUCCESS);
    REQUIRE(batch.empty());
    REQUIRE_FALSE(surface->wantsDraw());

    surface->invalidate();
    REQUIRE(surface->wantsDraw());
    surface->commitDraw();
    batch.commit();
    REQUIRE(window.destroy() == Result::SUCCESS);
}

TEST_CASE("Render programs can be enabled and disabled", "[render][program]") {
    Render::Program program(Render::Program::Config{});

    REQUIRE(program.enabled());
    program.setEnabled(false);
    REQUIRE_FALSE(program.enabled());
    program.setEnabled(true);
    REQUIRE(program.enabled());
}

TEST_CASE("Surfaces collect their complete transfer dependency graph",
          "[render][transfer][surface]") {
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

int main(int argc, char* argv[]) {
    return Catch::Session().run(argc, argv);
}
