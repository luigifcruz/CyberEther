#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <new>
#include <utility>

#include "jetstream/backend/base.hh"
#include "jetstream/render/components/curves.hh"
#include "jetstream/render/components/image.hh"
#include "jetstream/render/components/mesh.hh"
#include "jetstream/render/components/text.hh"
#include "../../../resources/fonts/compressed_jbmm.hh"
#include "component_test_support.hh"

using namespace Jetstream;
using namespace Jetstream::Render::Components;

namespace {

thread_local int allocationCountdown = -1;

Result FailAllocation(int countdown, auto&& operation) {
    allocationCountdown = countdown;
    try {
        const Result result = operation();
        allocationCountdown = -1;
        return result;
    } catch (...) {
        allocationCountdown = -1;
        throw;
    }
}

class UnsupportedWindow final : public Render::Window {
 public:
    UnsupportedWindow() : Window(Config{}) {}
    std::string info() const override { return "UnsupportedWindow"; }
    DeviceType device() const override {
        ++deviceQueries;
        return DeviceType::None;
    }
    mutable U64 deviceQueries = 0;

 protected:
    Result bindSurface(const std::shared_ptr<Render::Surface>&) override { return Result::ERROR; }
    Result unbindSurface(const std::shared_ptr<Render::Surface>&) override { return Result::ERROR; }
    Result underlyingCreate() override { return Result::SUCCESS; }
    Result underlyingDestroy() override { return Result::SUCCESS; }
    Result underlyingBegin() override { return Result::SUCCESS; }
    Result underlyingEnd() override { return Result::SUCCESS; }
    Result underlyingSynchronize() override { return Result::SUCCESS; }

};

class UnallocatedTexture final : public Render::Texture {
 public:
    UnallocatedTexture() : Texture(Config{.size = {1, 1}}) {}
    Result create() override { return Result::ERROR; }
    Result destroy() override { return Result::SUCCESS; }
    U64 raw() const override { return 0; }
};

class StateSurface final : public Render::Surface {
 public:
    using Surface::Surface;
    Result create() override { return Result::SUCCESS; }
    Result destroy() override { return Result::SUCCESS; }
    const Extent2D<U64>& size(const Extent2D<U64>&) override { return NullSize2D; }
    bool draw(bool visible = true) {
        if (!shouldDraw()) return false;
        const bool graphics = shouldDrawGraphics(visible);
        commitDraw();
        return graphics;
    }
};

F32 Pixel(const std::vector<F32>& pixels, U64 x, U64 y, U64 channel = 3) {
    return pixels[(y * 64 + x) * 4 + channel];
}

template<DeviceType D>
auto Frame() {
    return std::make_shared<ComponentTexture<D>>(Render::Texture::Config{
        .size = {64, 64}, .ptype = Render::Texture::PixelType::F32,
    });
}

template<DeviceType D>
auto Attach(ComponentWindow<D>& window, auto& component,
            const std::shared_ptr<ComponentTexture<D>>& texture,
            ColorRGBA<F32> clear = {0, 0, 0, 0}) {
    Render::Surface::Config config;
    config.framebuffer = texture;
    config.clearColor = clear;
    config.retained = true;
    REQUIRE(component.surface(config) == Result::SUCCESS);
    auto surface = std::make_shared<ComponentSurface<D>>(config);
    REQUIRE(window.bind(surface) == Result::SUCCESS);
    return surface;
}

template<DeviceType D>
void NoUploads(ComponentWindow<D>& window) {
    Render::Transfer::Batch batch;
    REQUIRE(window.collect(batch) == Result::SUCCESS);
    CHECK(batch.empty());
}

template<DeviceType D>
void RetryUploads(ComponentWindow<D>& window, ComponentDriver<D>& driver,
                  auto& component, int failures, U64 buffers, auto&& update) {
    for (int allocation = 0; allocation < failures; ++allocation) {
        CAPTURE(allocation);
        update(allocation);
        const Result result = FailAllocation(allocation, [&] { return component.present(); });
        REQUIRE(result == Result::ERROR);
        REQUIRE(component.present() == Result::SUCCESS);
        {
            Render::Transfer::Batch batch;
            REQUIRE(window.collect(batch) == Result::SUCCESS);
            CHECK(batch.buffers().size() == buffers);
        }
        driver.render(window);
        REQUIRE(component.present() == Result::SUCCESS);
        NoUploads(window);
    }
}

void FillMesh(Mesh& mesh) {
    mesh.positions()[0] = {8, 8};
    mesh.positions()[1] = {56, 8};
    mesh.positions()[2] = {8, 56};
    mesh.positions()[3] = {32, 32};
    mesh.positions()[4] = {56, 32};
    mesh.positions()[5] = {32, 56};
    for (auto& color : mesh.colors()) color = {1, 0, 0, 1};
    for (U32 i = 0; i < 6; ++i) mesh.indices()[i] = i;
}

void FillCurves(Curves& curves) {
    auto& instance = curves.instances()[0];
    instance.p0 = {8, 32};
    instance.p1 = {24, 32};
    instance.p2 = {40, 32};
    instance.p3 = {56, 32};
    instance.color = {0, 1, 0, 1};
    instance.thickness = 4;
    instance.segments = 4;
    curves.instances()[1] = instance;
    auto& second = curves.instances()[1];
    second.p0.y = second.p1.y = second.p2.y = second.p3.y = 48;
}

template<DeviceType D>
void CheckMesh(ComponentDriver<D>& driver) {
    ComponentWindow<D> window;
    REQUIRE(window.create() == Result::SUCCESS);
    Mesh mesh({.vertexCapacity = 6, .indexCapacity = 6, .framebufferSize = {64, 64}});
    REQUIRE(mesh.create(&window) == Result::SUCCESS);
    FillMesh(mesh);
    REQUIRE(mesh.update(3, 3) == Result::SUCCESS);
    auto texture = Frame<D>();
    auto surface = Attach(window, mesh, texture);
    REQUIRE(mesh.present() == Result::SUCCESS);
    driver.render(window);
    CHECK(Pixel(driver.pixels(texture), 12, 12) > 0.99f);
    const auto revision = texture->contentRevision();
    driver.render(window);
    CHECK(texture->contentRevision() == revision);

    REQUIRE(mesh.update(0, 0) == Result::SUCCESS);
    REQUIRE(mesh.present() == Result::SUCCESS);
    NoUploads(window);
    driver.render(window);
    CHECK(Pixel(driver.pixels(texture), 12, 12) < 0.001f);
    REQUIRE(mesh.update(3, 3) == Result::SUCCESS);
    REQUIRE(mesh.present() == Result::SUCCESS);
    driver.render(window);
    mesh.enabled(false);
    NoUploads(window);
    driver.render(window);
    CHECK(Pixel(driver.pixels(texture), 12, 12) < 0.001f);
    mesh.enabled(true);
    NoUploads(window);
    driver.render(window);
    CHECK(Pixel(driver.pixels(texture), 12, 12) > 0.99f);
    REQUIRE(mesh.updateScissorRect(Render::ScissorRect{0, 0, 1, 1}) == Result::SUCCESS);
    NoUploads(window);
    driver.render(window, surface.get());
    CHECK(Pixel(driver.pixels(texture), 12, 12) > 0.99f);
    driver.render(window);
    CHECK(Pixel(driver.pixels(texture), 12, 12) < 0.001f);
    REQUIRE(mesh.updateScissorRect(std::nullopt) == Result::SUCCESS);
    NoUploads(window);
    driver.render(window);
    CHECK(Pixel(driver.pixels(texture), 12, 12) > 0.99f);

    RetryUploads(window, driver, mesh, 2, 1, [&](int i) {
        REQUIRE(mesh.updateTransform({64, 64}, {static_cast<F32>(i + 1), 0}, 1) == Result::SUCCESS);
    });
    REQUIRE(mesh.updateTransform({64, 64}, {0, 0}, 1) == Result::SUCCESS);
    REQUIRE(mesh.present() == Result::SUCCESS);
    driver.render(window);
    RetryUploads(window, driver, mesh, 8, 4, [&](int i) {
        REQUIRE(mesh.update(6, i % 2 ? 3 : 6) == Result::SUCCESS);
    });
    REQUIRE(mesh.update(3, 3) == Result::SUCCESS);
    REQUIRE(mesh.present() == Result::SUCCESS);
    driver.render(window);
    REQUIRE(window.destroy() == Result::SUCCESS);
    REQUIRE(mesh.destroy(&window) == Result::SUCCESS);
    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(mesh.create(&window) == Result::SUCCESS);
    FillMesh(mesh);
    REQUIRE(mesh.update(3, 3) == Result::SUCCESS);
    texture = Frame<D>();
    surface = Attach(window, mesh, texture);
    REQUIRE(mesh.present() == Result::SUCCESS);
    driver.render(window);
    CHECK(Pixel(driver.pixels(texture), 12, 12) > 0.99f);
    CHECK(Pixel(driver.pixels(texture), 36, 36) < 0.001f);
    REQUIRE(window.destroy() == Result::SUCCESS);
}

template<DeviceType D>
void CheckMeshIndexValidation() {
    if (!std::getenv("JETSTREAM_TEST_NATIVE_COMPONENTS")) {
        SKIP("Set JETSTREAM_TEST_NATIVE_COMPONENTS=1 to run headless GPU regressions.");
    }
    REQUIRE(Backend::Initialize<D>(Backend::Config{.validationEnabled = false, .headless = true}) == Result::SUCCESS);
    ComponentDriver<D> driver;
    ComponentWindow<D> window;
    REQUIRE(window.create() == Result::SUCCESS);
    Mesh mesh({.vertexCapacity = 6, .indexCapacity = 6, .framebufferSize = {64, 64}});
    REQUIRE(mesh.create(&window) == Result::SUCCESS);
    FillMesh(mesh);
    REQUIRE(mesh.update(6, 6) == Result::SUCCESS);
    auto texture = Frame<D>();
    auto surface = Attach(window, mesh, texture);
    REQUIRE(mesh.present() == Result::SUCCESS);
    driver.render(window);
    const auto baseline = driver.pixels(texture);
    REQUIRE(Pixel(baseline, 12, 12) > 0.99f);
    REQUIRE(Pixel(baseline, 36, 36) > 0.99f);

    struct Request {
        U64 vertexCount;
        U32 invalidIndex;
    };
    const std::array<Request, 4> requests = {{{3, 6}, {3, 3}, {3, std::numeric_limits<U32>::max()}, {0, 0}}};
    for (const auto& request : requests) {
        for (U64 offset = 0; offset < 3; ++offset) {
            CAPTURE(request.vertexCount, request.invalidIndex, offset);
            FillMesh(mesh);
            REQUIRE(mesh.update(6, 6) == Result::SUCCESS);
            REQUIRE(mesh.updateTransform({64, 64}, {0, 0}, 1) == Result::SUCCESS);
            REQUIRE(mesh.present() == Result::SUCCESS);
            driver.render(window);
            REQUIRE(driver.pixels(texture) == baseline);

            REQUIRE(mesh.update(request.vertexCount, 3) == Result::SUCCESS);
            mesh.indices()[offset] = request.invalidIndex;
            mesh.positions()[0] = {36, 8};
            mesh.positions()[1] = {56, 8};
            mesh.positions()[2] = {36, 24};
            for (U64 i = 0; i < 3; ++i) mesh.colors()[i] = {0, 1, 0, 1};
            REQUIRE(mesh.updateTransform({64, 64}, {4, 0}, 1) == Result::SUCCESS);
            REQUIRE(mesh.present() == Result::ERROR);
            {
                Render::Transfer::Batch batch;
                REQUIRE(window.collect(batch) == Result::SUCCESS);
                REQUIRE(batch.empty());
            }
            surface->invalidate();
            driver.render(window);
            CHECK(driver.pixels(texture) == baseline);
            REQUIRE(mesh.present() == Result::ERROR);
            NoUploads(window);

            mesh.indices()[offset] = static_cast<U32>(offset);
            if (request.vertexCount == 0) REQUIRE(mesh.update(3, 3) == Result::SUCCESS);
            REQUIRE(mesh.indices()[2] == 2);
            REQUIRE(mesh.present() == Result::SUCCESS);
            {
                Render::Transfer::Batch batch;
                REQUIRE(window.collect(batch) == Result::SUCCESS);
                REQUIRE(batch.buffers().size() == 5);
                const auto count = [&](Render::Buffer::Target target) {
                    return std::count_if(batch.buffers().begin(), batch.buffers().end(), [&](const auto& upload) {
                        return upload.destination->getConfig().target == target;
                    });
                };
                CHECK(count(Render::Buffer::Target::UNIFORM) == 1);
                CHECK(count(Render::Buffer::Target::VERTEX) == 2);
                CHECK(count(Render::Buffer::Target::VERTEX_INDICES) == 1);
                CHECK(count(Render::Buffer::Target::INDIRECT) == 1);
            }
            driver.render(window);
            const auto corrected = driver.pixels(texture);
            CHECK(Pixel(corrected, 44, 12) > 0.99f);
            CHECK(Pixel(corrected, 44, 12, 1) > 0.99f);
            CHECK(Pixel(corrected, 44, 12, 0) < 0.001f);
            CHECK(Pixel(corrected, 12, 12) < 0.001f);
            CHECK(Pixel(corrected, 36, 36) < 0.001f);
            REQUIRE(mesh.present() == Result::SUCCESS);
            NoUploads(window);
        }
    }
    REQUIRE(window.destroy() == Result::SUCCESS);
}

template<DeviceType D>
void ReleaseComponentSurface(ComponentWindow<D>& window, const std::shared_ptr<ComponentSurface<D>>& surface) {
    REQUIRE(window.unbind(surface) == Result::SUCCESS);
    REQUIRE(window.surfaces.empty());
    REQUIRE(window.start() == Result::SUCCESS);
    for (U64 frame = 0; frame < 5; ++frame) {
        REQUIRE(window.begin() == Result::SUCCESS);
        REQUIRE(window.end() == Result::SUCCESS);
    }
    REQUIRE(window.stop() == Result::SUCCESS);
}

void CheckMeshIndexUpload(const Render::Transfer::Batch& batch,
                          const std::array<U32, 6>& expected, U64 vertexLimit, U64 count = 6) {
    const auto& uploads = batch.buffers();
    const auto isIndex = [](const auto& upload) {
        return upload.destination->getConfig().target == Render::Buffer::Target::VERTEX_INDICES;
    };
    REQUIRE(std::count_if(uploads.begin(), uploads.end(), isIndex) == 1);
    const auto& upload = *std::find_if(uploads.begin(), uploads.end(), isIndex);
    REQUIRE(upload.destinationOffset == 0);
    REQUIRE(upload.upload.start == 0);
    REQUIRE(upload.upload.unitByteSize == sizeof(U32));
    REQUIRE(count <= expected.size());
    REQUIRE(upload.upload.data.size() == count * sizeof(U32));
    std::vector<U32> values(count);
    std::memcpy(values.data(), upload.upload.data.data(), count * sizeof(U32));
    REQUIRE(std::all_of(values.begin(), values.end(), [&](U32 index) { return index < vertexLimit; }));
    REQUIRE(std::equal(values.begin(), values.end(), expected.begin()));
    REQUIRE_FALSE(upload.destination->getConfig().enableZeroCopy);
}

template<DeviceType D>
void CheckMeshIndexSnapshotBindings() {
    if (!std::getenv("JETSTREAM_TEST_NATIVE_COMPONENTS")) {
        SKIP("Set JETSTREAM_TEST_NATIVE_COMPONENTS=1 to run headless GPU regressions.");
    }
    REQUIRE(Backend::Initialize<D>(Backend::Config{.validationEnabled = false, .headless = true}) == Result::SUCCESS);
    ComponentDriver<D> driver;
    ComponentWindow<D> window;
    REQUIRE(window.create() == Result::SUCCESS);
    Mesh mesh({.vertexCapacity = 6, .indexCapacity = 6, .framebufferSize = {64, 64}});
    REQUIRE(mesh.create(&window) == Result::SUCCESS);
    FillMesh(mesh);
    mesh.indices()[2] = 5000;
    REQUIRE(mesh.update(3, 3) == Result::SUCCESS);
    auto texture = Frame<D>();
    auto surface = Attach(window, mesh, texture);
    {
        Render::Transfer::Batch binding;
        REQUIRE(window.collect(binding) == Result::SUCCESS);
        REQUIRE(binding.buffers().size() == 5);
        CheckMeshIndexUpload(binding, {0, 0, 0, 0, 0, 0}, 3);
        REQUIRE(mesh.updateTransform({64, 64}, {4, 0}, 1) == Result::SUCCESS);
        REQUIRE(mesh.present() == Result::ERROR);
        NoUploads(window);
    }
    driver.render(window);
    const auto empty = driver.pixels(texture);
    REQUIRE(std::all_of(empty.begin(), empty.end(), [](F32 value) { return value == 0.0f; }));

    mesh.indices()[2] = 2;
    REQUIRE(mesh.present() == Result::SUCCESS);
    {
        Render::Transfer::Batch retry;
        REQUIRE(window.collect(retry) == Result::SUCCESS);
        REQUIRE(retry.buffers().size() == 5);
        CheckMeshIndexUpload(retry, {0, 1, 2, 0, 0, 0}, 3, 3);
    }
    driver.render(window);
    CHECK(Pixel(driver.pixels(texture), 16, 12) > 0.99f);

    REQUIRE(mesh.update(6, 6) == Result::SUCCESS);
    REQUIRE(mesh.present() == Result::SUCCESS);
    {
        Render::Transfer::Batch full;
        REQUIRE(window.collect(full) == Result::SUCCESS);
        CheckMeshIndexUpload(full, {0, 1, 2, 3, 4, 5}, 6);
    }
    driver.render(window);
    REQUIRE(Pixel(driver.pixels(texture), 40, 36) > 0.99f);
    REQUIRE(mesh.update(3, 3) == Result::SUCCESS);
    std::fill(mesh.indices().begin() + 3, mesh.indices().end(), std::numeric_limits<U32>::max());
    REQUIRE(mesh.updateTransform({64, 64}, {0, 0}, 1) == Result::SUCCESS);
    REQUIRE(mesh.present() == Result::SUCCESS);
    {
        Render::Transfer::Batch shrink;
        REQUIRE(window.collect(shrink) == Result::SUCCESS);
        REQUIRE(shrink.buffers().size() == 5);
        CheckMeshIndexUpload(shrink, {0, 1, 2, 0, 0, 0}, 3);
    }
    driver.render(window);
    REQUIRE(Pixel(driver.pixels(texture), 36, 36) < 0.001f);

    auto indexBuffer = surface->getConfig().programs.front()->getConfig().draws.front()->getConfig().buffer->getConfig().indices;
    REQUIRE(indexBuffer->getConfig().buffer != mesh.indices().data());
    REQUIRE(window.bind(indexBuffer) == Result::SUCCESS);
    ReleaseComponentSurface(window, surface);
    texture = Frame<D>();
    surface = Attach(window, mesh, texture);
    {
        Render::Transfer::Batch pinned;
        REQUIRE(window.collect(pinned) == Result::SUCCESS);
        REQUIRE(std::none_of(pinned.buffers().begin(), pinned.buffers().end(), [&](const auto& upload) {
            return upload.destination == indexBuffer;
        }));
    }
    driver.render(window);
    CHECK(Pixel(driver.pixels(texture), 12, 12) > 0.99f);
    CHECK(Pixel(driver.pixels(texture), 36, 36) < 0.001f);
    REQUIRE(window.unbind(indexBuffer) == Result::SUCCESS);

    for (bool update : {false, true}) {
        CAPTURE(update);
        const auto baseline = driver.pixels(texture);
        mesh.indices()[0] = 5000;
        if (update) REQUIRE(mesh.update(3, 3) == Result::SUCCESS);
        ReleaseComponentSurface(window, surface);
        texture = Frame<D>();
        surface = Attach(window, mesh, texture);
        {
            Render::Transfer::Batch rebound;
            REQUIRE(window.collect(rebound) == Result::SUCCESS);
            CheckMeshIndexUpload(rebound, {0, 1, 2, 0, 0, 0}, 3);
            REQUIRE(mesh.present() == (update ? Result::ERROR : Result::SUCCESS));
            NoUploads(window);
        }
        driver.render(window);
        CHECK(driver.pixels(texture) == baseline);
        mesh.indices()[0] = 0;
        REQUIRE(mesh.present() == Result::SUCCESS);
        if (update) {
            Render::Transfer::Batch corrected;
            REQUIRE(window.collect(corrected) == Result::SUCCESS);
            REQUIRE(corrected.buffers().size() == 3);
            CheckMeshIndexUpload(corrected, {0, 1, 2, 0, 0, 0}, 3, 3);
        }
        driver.render(window);
        CHECK(driver.pixels(texture) == baseline);
    }

    REQUIRE(mesh.update(0, 0) == Result::SUCCESS);
    REQUIRE(mesh.present() == Result::SUCCESS);
    NoUploads(window);
    std::fill(mesh.indices().begin(), mesh.indices().end(), std::numeric_limits<U32>::max());
    ReleaseComponentSurface(window, surface);
    texture = Frame<D>();
    surface = Attach(window, mesh, texture);
    {
        Render::Transfer::Batch zero;
        REQUIRE(window.collect(zero) == Result::SUCCESS);
        CheckMeshIndexUpload(zero, {0, 0, 0, 0, 0, 0}, 1);
    }
    driver.render(window);
    CHECK(Pixel(driver.pixels(texture), 12, 12) < 0.001f);
    REQUIRE(mesh.update(0, 3) == Result::SUCCESS);
    REQUIRE(mesh.present() == Result::ERROR);
    NoUploads(window);
    driver.render(window);
    CHECK(Pixel(driver.pixels(texture), 12, 12) < 0.001f);
    REQUIRE(window.destroy() == Result::SUCCESS);
}

template<DeviceType D>
void CheckMeshUploadRanges() {
    if (!std::getenv("JETSTREAM_TEST_NATIVE_COMPONENTS")) {
        SKIP("Set JETSTREAM_TEST_NATIVE_COMPONENTS=1 to run headless GPU regressions.");
    }
    REQUIRE(Backend::Initialize<D>(Backend::Config{.validationEnabled = false, .headless = true}) == Result::SUCCESS);
    ComponentDriver<D> driver;
    ComponentWindow<D> window;
    REQUIRE(window.create() == Result::SUCCESS);
    Mesh mesh({.vertexCapacity = 6, .indexCapacity = 196608, .framebufferSize = {64, 64}});
    REQUIRE(mesh.create(&window) == Result::SUCCESS);
    FillMesh(mesh);
    REQUIRE(mesh.update(3, 3) == Result::SUCCESS);
    auto texture = Frame<D>();
    auto surface = Attach(window, mesh, texture);
    REQUIRE(mesh.present() == Result::SUCCESS);
    driver.render(window);
    const auto baseline = driver.pixels(texture);

    for (int frame = 0; frame < 60; ++frame) {
        REQUIRE(mesh.update(3, 3) == Result::SUCCESS);
        REQUIRE(mesh.present() == Result::SUCCESS);
        {
            Render::Transfer::Batch batch;
            REQUIRE(window.collect(batch) == Result::SUCCESS);
            CheckMeshIndexUpload(batch, {0, 1, 2, 0, 0, 0}, 3, 3);
        }
        driver.render(window);
    }
    CHECK(driver.pixels(texture) == baseline);

    auto indexBuffer = surface->getConfig().programs.front()->getConfig().draws.front()->getConfig().buffer->getConfig().indices;
    auto& indexConfig = const_cast<Render::Buffer::Config&>(indexBuffer->getConfig());
    void* indexSource = indexConfig.buffer;
    REQUIRE(mesh.update(6, 6) == Result::SUCCESS);
    indexConfig.buffer = nullptr;
    const Result rejected = mesh.present();
    indexConfig.buffer = indexSource;
    REQUIRE(rejected == Result::ERROR);
    REQUIRE(mesh.update(3, 3) == Result::SUCCESS);
    REQUIRE(mesh.present() == Result::SUCCESS);
    {
        Render::Transfer::Batch retry;
        REQUIRE(window.collect(retry) == Result::SUCCESS);
        CheckMeshIndexUpload(retry, {0, 1, 2, 0, 0, 0}, 3);
    }
    driver.render(window);
    CHECK(driver.pixels(texture) == baseline);

    REQUIRE(mesh.update(6, 6) == Result::SUCCESS);
    REQUIRE(mesh.present() == Result::SUCCESS);
    driver.render(window);
    CHECK(Pixel(driver.pixels(texture), 36, 36) > 0.99f);
    REQUIRE(mesh.update(0, 0) == Result::SUCCESS);
    REQUIRE(mesh.present() == Result::SUCCESS);
    NoUploads(window);
    REQUIRE(mesh.update(3, 3) == Result::SUCCESS);
    REQUIRE(mesh.present() == Result::SUCCESS);
    {
        Render::Transfer::Batch batch;
        REQUIRE(window.collect(batch) == Result::SUCCESS);
        CheckMeshIndexUpload(batch, {0, 1, 2, 0, 0, 0}, 3);
    }
    driver.render(window);
    CHECK(driver.pixels(texture) == baseline);
    ReleaseComponentSurface(window, surface);
    surface = Attach(window, mesh, texture);
    driver.render(window);
    CHECK(driver.pixels(texture) == baseline);
    REQUIRE(window.destroy() == Result::SUCCESS);
}

template<DeviceType D>
void CheckCurves(ComponentDriver<D>& driver) {
    ComponentWindow<D> window;
    REQUIRE(window.create() == Result::SUCCESS);
    Curves curves({.capacity = 2, .maxSegments = 4, .framebufferSize = {64, 64}});
    REQUIRE(curves.create(&window) == Result::SUCCESS);
    FillCurves(curves);
    REQUIRE(curves.update(1) == Result::SUCCESS);
    auto texture = Frame<D>();
    auto surface = Attach(window, curves, texture);
    REQUIRE(curves.present() == Result::SUCCESS);
    driver.render(window);
    for (F32 segments : {8.0f, 2.5f, 0.5f, 4.0f}) {
        curves.instances()[0].segments = segments;
        REQUIRE(curves.update(1) == Result::SUCCESS);
        REQUIRE(curves.present() == Result::SUCCESS);
        driver.render(window);
        const auto pixels = driver.pixels(texture);
        CHECK(Pixel(pixels, 8, 32) > 0.99f);
        CHECK(Pixel(pixels, 54, 32) > 0.99f);
        CHECK(Pixel(pixels, 57, 32) < 0.001f);
    }
    REQUIRE(curves.update(0) == Result::SUCCESS);
    REQUIRE(curves.present() == Result::SUCCESS);
    NoUploads(window);
    driver.render(window);
    CHECK(Pixel(driver.pixels(texture), 48, 32) < 0.001f);
    REQUIRE(curves.update(1) == Result::SUCCESS);
    REQUIRE(curves.present() == Result::SUCCESS);
    driver.render(window);
    curves.enabled(false);
    NoUploads(window);
    driver.render(window);
    CHECK(Pixel(driver.pixels(texture), 48, 32) < 0.001f);
    curves.enabled(true);
    NoUploads(window);
    driver.render(window);
    CHECK(Pixel(driver.pixels(texture), 48, 32) > 0.99f);
    REQUIRE(curves.updateScissorRect(Render::ScissorRect{0, 0, 1, 1}) == Result::SUCCESS);
    NoUploads(window);
    driver.render(window, surface.get());
    CHECK(Pixel(driver.pixels(texture), 48, 32) > 0.99f);
    driver.render(window);
    CHECK(Pixel(driver.pixels(texture), 48, 32) < 0.001f);
    REQUIRE(curves.updateScissorRect(std::nullopt) == Result::SUCCESS);
    NoUploads(window);
    driver.render(window);
    CHECK(Pixel(driver.pixels(texture), 48, 32) > 0.99f);
    RetryUploads(window, driver, curves, 2, 1, [&](int i) {
        REQUIRE(curves.updateTransform({64, 64}, {0, 0}, 1, 1.25f + i) == Result::SUCCESS);
    });
    RetryUploads(window, driver, curves, 4, 2, [&](int i) {
        REQUIRE(curves.update(i % 2 ? 1 : 2) == Result::SUCCESS);
    });
    REQUIRE(curves.update(2) == Result::SUCCESS);
    REQUIRE(curves.present() == Result::SUCCESS);
    driver.render(window);
    REQUIRE(window.destroy() == Result::SUCCESS);
    REQUIRE(curves.destroy(&window) == Result::SUCCESS);
    REQUIRE(window.create() == Result::SUCCESS);
    REQUIRE(curves.create(&window) == Result::SUCCESS);
    FillCurves(curves);
    REQUIRE(curves.update(2) == Result::SUCCESS);
    texture = Frame<D>();
    surface = Attach(window, curves, texture);
    REQUIRE(curves.present() == Result::SUCCESS);
    driver.render(window);
    CHECK(Pixel(driver.pixels(texture), 48, 32) > 0.99f);
    CHECK(Pixel(driver.pixels(texture), 48, 48) > 0.99f);
    REQUIRE(window.destroy() == Result::SUCCESS);
}

template<DeviceType D>
void CheckImages(ComponentDriver<D>& driver) {
    ComponentWindow<D> window;
    REQUIRE(window.create() == Result::SUCCESS);
    std::array<U8, 4> bytes = {255, 0, 0, 128};
    auto source = std::make_shared<ComponentTexture<D>>(Render::Texture::Config{
        .size = {1, 1}, .buffer = bytes.data(),
    });
    REQUIRE(window.bind(source) == Result::SUCCESS);
    window.eraseSupported = false;
    Image unsupported({.texture = source, .punch = true});
    REQUIRE(unsupported.create(&window) == Result::ERROR);
    window.eraseSupported = true;
    Image image({.texture = source});
    REQUIRE(image.create(&window) == Result::SUCCESS);
    REQUIRE(image.update({64, 64}, {0, 0}, {8, 8, 48, 48}, 12, 0.5f) == Result::SUCCESS);
    auto texture = Frame<D>();
    auto surface = Attach(window, image, texture);
    REQUIRE(image.present() == Result::SUCCESS);
    driver.render(window);
    const auto straight = driver.pixels(texture);
    CHECK(std::abs(Pixel(straight, 32, 32) - 128.0f / 510) < 0.002f);
    CHECK(std::abs(Pixel(straight, 32, 32, 0) - Pixel(straight, 32, 32)) < 0.002f);
    CHECK(Pixel(straight, 8, 8) < 0.001f);
    REQUIRE(image.update({64, 64}, {0, 0}, {0, 0, 64, 64}, 0, 0.5f) == Result::SUCCESS);
    REQUIRE(image.present() == Result::SUCCESS);
    driver.render(window);

    Image consumer({.texture = texture, .sourceAlpha = Image::SourceAlpha::PREMULTIPLIED});
    REQUIRE(consumer.create(&window) == Result::SUCCESS);
    REQUIRE(consumer.update({64, 64}, {0, 0}, {0, 0, 64, 64}, 12, 0.5f) == Result::SUCCESS);
    auto destination = Frame<D>();
    auto consumerSurface = Attach(window, consumer, destination);
    REQUIRE(consumer.present() == Result::SUCCESS);
    driver.render(window);
    const auto premultiplied = driver.pixels(destination);
    for (U64 y = 0; y < 64; ++y) {
        for (U64 x = 0; x < 64; ++x) {
            const F32 qx = std::abs(static_cast<F32>(x) + 0.5f - 32) - 20;
            const F32 qy = std::abs(static_cast<F32>(y) + 0.5f - 32) - 20;
            const F32 distance = std::hypot(std::max(qx, 0.0f), std::max(qy, 0.0f)) +
                                 std::min(std::max(qx, qy), 0.0f) - 12;
            const F32 expected = 128.0f / 1020 * std::clamp(0.5f - distance, 0.0f, 1.0f);
            CHECK(std::isfinite(Pixel(premultiplied, x, y, 0)));
            CHECK(std::abs(Pixel(premultiplied, x, y, 0) - expected) < 0.002f);
            CHECK(std::abs(Pixel(premultiplied, x, y) - expected) < 0.002f);
        }
    }
    REQUIRE(consumer.update({64, 64}, {0, 0}, {0, 0, 64, 64}, 0, 1) == Result::SUCCESS);
    REQUIRE(consumer.present() == Result::SUCCESS);
    driver.render(window);
    CHECK(std::abs(Pixel(driver.pixels(destination), 32, 32, 0) - Pixel(straight, 32, 32, 0)) < 0.002f);
    image.enabled(false);
    NoUploads(window);
    driver.render(window);
    CHECK(Pixel(driver.pixels(texture), 32, 32) < 0.001f);
    const auto transparent = driver.pixels(destination);
    CHECK(std::isfinite(Pixel(transparent, 32, 32, 0)));
    CHECK(Pixel(transparent, 32, 32, 0) == 0.0f);
    CHECK(Pixel(transparent, 32, 32) == 0.0f);
    image.enabled(true);
    NoUploads(window);
    driver.render(window);
    CHECK(Pixel(driver.pixels(texture), 32, 32) > 0.25f);
    REQUIRE(image.updateScissorRect(Render::ScissorRect{0, 0, 1, 1}) == Result::SUCCESS);
    NoUploads(window);
    driver.render(window, surface.get());
    CHECK(Pixel(driver.pixels(texture), 32, 32) > 0.25f);
    driver.render(window);
    CHECK(Pixel(driver.pixels(texture), 32, 32) < 0.001f);
    REQUIRE(image.updateScissorRect(std::nullopt) == Result::SUCCESS);
    NoUploads(window);
    driver.render(window);
    CHECK(Pixel(driver.pixels(texture), 32, 32) > 0.25f);
    RetryUploads(window, driver, image, 2, 1, [&](int i) {
        REQUIRE(image.update({64, 64}, {0, 0}, {8, 8, 48, 48}, 12, 0.6f + i * 0.1f) == Result::SUCCESS);
    });
    REQUIRE(window.destroy() == Result::SUCCESS);

    REQUIRE(window.create() == Result::SUCCESS);
    Image punch({.punch = true});
    REQUIRE(punch.create(&window) == Result::SUCCESS);
    REQUIRE(punch.update({64, 64}, {0, 0}, {8, 8, 48, 48}, 12, 0.5f) == Result::SUCCESS);
    texture = Frame<D>();
    surface = Attach(window, punch, texture, {0.4f, 0.3f, 0.2f, 0.8f});
    REQUIRE(punch.present() == Result::SUCCESS);
    driver.render(window);
    const auto punched = driver.pixels(texture);
    CHECK(std::abs(Pixel(punched, 32, 32) - 0.4f) < 0.002f);
    CHECK(std::abs(Pixel(punched, 32, 32, 0) - 0.2f) < 0.002f);
    CHECK(std::abs(Pixel(punched, 8, 8) - 0.8f) < 0.002f);
    punch.enabled(false);
    NoUploads(window);
    driver.render(window);
    CHECK(std::abs(Pixel(driver.pixels(texture), 32, 32) - 0.8f) < 0.002f);
    punch.enabled(true);
    NoUploads(window);
    driver.render(window);
    CHECK(std::abs(Pixel(driver.pixels(texture), 32, 32) - 0.4f) < 0.002f);
    REQUIRE(window.destroy() == Result::SUCCESS);
}

template<DeviceType D>
void CheckTextStyles(ComponentDriver<D>& driver) {
    ComponentWindow<D> window;
    REQUIRE(window.create() == Result::SUCCESS);
    auto font = std::make_shared<Font>(Font::Config{.size = 32.0f, .data = jbmm_compressed_data});
    REQUIRE(font->create(&window) == Result::SUCCESS);
    Text text({.maxCharacters = 4, .font = font, .pixelSize = {2.0f / 64, 2.0f / 64},
               .elements = {{"label", {.position = {0, 0}, .alignment = {1, 1}, .fill = "A"}}}});
    REQUIRE(text.create(&window) == Result::SUCCESS);
    auto texture = Frame<D>();
    auto surface = Attach(window, text, texture);
    REQUIRE(text.present() == Result::SUCCESS);
    driver.render(window);
    const auto before = driver.pixels(texture);
    Render::Surface::Config programsBefore;
    REQUIRE(text.surface(programsBefore) == Result::SUCCESS);

    REQUIRE(text.updateStyle(0.3f, 0.75f, 0.6f, 0.9f) == Result::SUCCESS);
    REQUIRE(text.present() == Result::SUCCESS);
    {
        Render::Transfer::Batch batch;
        REQUIRE(window.collect(batch) == Result::SUCCESS);
        REQUIRE(batch.buffers().size() == 1);
        CHECK(batch.buffers().front().upload.data.size() == 32);
        CHECK(batch.textures().empty());
    }
    driver.render(window);
    const auto after = driver.pixels(texture);
    bool changed = false;
    for (U64 i = 0; i < before.size(); ++i) changed |= std::abs(before[i] - after[i]) > 0.001f;
    CHECK(changed);
    Render::Surface::Config programsAfter;
    REQUIRE(text.surface(programsAfter) == Result::SUCCESS);
    CHECK(programsBefore.programs == programsAfter.programs);
    REQUIRE(text.updateStyle(0.3f, 0.75f, 0.6f, 0.9f) == Result::SUCCESS);
    REQUIRE(text.present() == Result::SUCCESS);
    NoUploads(window);
    REQUIRE(window.destroy() == Result::SUCCESS);
}

template<DeviceType D>
void CheckNativeTextStyle() {
    if (!std::getenv("JETSTREAM_TEST_NATIVE_COMPONENTS")) {
        SKIP("Set JETSTREAM_TEST_NATIVE_COMPONENTS=1 to run headless GPU regressions.");
    }
    REQUIRE(Backend::Initialize<D>(Backend::Config{.validationEnabled = false, .headless = true}) == Result::SUCCESS);
    ComponentDriver<D> driver;
    CheckTextStyles(driver);
}

template<DeviceType D>
void CheckNativeComponents() {
    if (!std::getenv("JETSTREAM_TEST_NATIVE_COMPONENTS")) {
        SKIP("Set JETSTREAM_TEST_NATIVE_COMPONENTS=1 to run headless GPU regressions.");
    }
    REQUIRE(Backend::Initialize<D>(Backend::Config{.validationEnabled = false, .headless = true}) == Result::SUCCESS);
    ComponentDriver<D> driver;
    CheckMesh(driver);
    CheckCurves(driver);
    CheckImages(driver);
}

}  // namespace

void* operator new(std::size_t size) {
    if (allocationCountdown >= 0 && allocationCountdown-- == 0) {
        allocationCountdown = -1;
        throw std::bad_alloc();
    }
    if (auto memory = std::malloc(size ? size : 1)) return memory;
    throw std::bad_alloc();
}

void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }

TEST_CASE("Component capacities reject overflow before allocation", "[core][render][components][overflow]") {
    for (U64 segments : {(U64{1} << 63) + 1, std::numeric_limits<U64>::max(),
                         U64{std::numeric_limits<U32>::max()} / 18 + 1,
                         (U64{1} << std::numeric_limits<F32>::digits) + 1}) {
        Curves curves({.capacity = 1, .maxSegments = segments});
        CHECK(curves.create(nullptr) == Result::ERROR);
    }
    for (U64 capacity : {std::numeric_limits<U64>::max(), U64{std::numeric_limits<U32>::max()} + 1}) {
        Curves curves({.capacity = capacity, .maxSegments = 1});
        CHECK(curves.create(nullptr) == Result::ERROR);
        Mesh vertices({.vertexCapacity = capacity, .indexCapacity = 3});
        CHECK(vertices.create(nullptr) == Result::ERROR);
        Mesh indices({.vertexCapacity = 3, .indexCapacity = capacity + (capacity != std::numeric_limits<U64>::max() ? 2 : 0)});
        CHECK(indices.create(nullptr) == Result::ERROR);
    }
}

TEST_CASE("Image rejects unsupported erase blending before construction", "[core][render][components][punch]") {
    UnsupportedWindow window;
    Image image({.texture = std::make_shared<UnallocatedTexture>(), .punch = true});
    REQUIRE(image.create(&window) == Result::ERROR);
    CHECK(window.deviceQueries == 0);
    Render::Surface::Config config;
    REQUIRE(image.surface(config) == Result::SUCCESS);
    REQUIRE(config.programs.size() == 1);
    CHECK_FALSE(config.programs.front());
}

TEST_CASE("Program state changes invalidate retained surfaces without transfers", "[core][render][components][retained]") {
    auto program = std::make_shared<Render::Program>(Render::Program::Config{});
    StateSurface surface({.programs = {program}, .retained = true});
    REQUIRE(surface.draw());
    CHECK_FALSE(surface.draw());
    program->setEnabled(false);
    CHECK(surface.draw());
    CHECK_FALSE(surface.draw());
    program->setEnabled(false);
    CHECK_FALSE(surface.draw());
    program->setEnabled(true);
    CHECK(surface.draw());
    program->scissorRect(Render::ScissorRect{1, 2, 3, 4});
    CHECK_FALSE(surface.draw(false));
    CHECK(surface.draw());
    CHECK_FALSE(surface.draw());
    program->scissorRect(Render::ScissorRect{1, 2, 3, 4});
    CHECK_FALSE(surface.draw());
    program->scissorRect(std::nullopt);
    CHECK(surface.draw());
    CHECK_FALSE(surface.draw());
}

#ifdef JETSTREAM_RENDER_METAL_AVAILABLE
TEST_CASE("Metal text style changes update uniforms without rebuilding programs", "[core][render][native][metal][text-style]") {
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    CheckNativeTextStyle<DeviceType::Metal>();
}

TEST_CASE("Metal components retain state and retry failed uploads", "[core][render][components][native][metal]") {
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    CheckNativeComponents<DeviceType::Metal>();
}

TEST_CASE("Metal mesh rejects invalid indices before uploads", "[core][render][components][native][metal][mesh-indices]") {
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    CheckMeshIndexValidation<DeviceType::Metal>();
}

TEST_CASE("Metal mesh binds only validated index snapshots", "[core][render][components][native][metal][mesh-bind-indices]") {
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    CheckMeshIndexSnapshotBindings<DeviceType::Metal>();
}

TEST_CASE("Metal mesh uploads active index ranges without changing pixels", "[core][render][components][native][metal][mesh-upload-ranges]") {
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    CheckMeshUploadRanges<DeviceType::Metal>();
}
#endif

#ifdef JETSTREAM_RENDER_VULKAN_AVAILABLE
TEST_CASE("Vulkan text style changes update uniforms without rebuilding programs", "[core][render][native][vulkan][text-style]") {
    CheckNativeTextStyle<DeviceType::Vulkan>();
}

TEST_CASE("Vulkan components retain state and retry failed uploads", "[core][render][components][native][vulkan]") {
    CheckNativeComponents<DeviceType::Vulkan>();
}

TEST_CASE("Vulkan mesh rejects invalid indices before uploads", "[core][render][components][native][vulkan][mesh-indices]") {
    CheckMeshIndexValidation<DeviceType::Vulkan>();
}

TEST_CASE("Vulkan mesh binds only validated index snapshots", "[core][render][components][native][vulkan][mesh-bind-indices]") {
    CheckMeshIndexSnapshotBindings<DeviceType::Vulkan>();
}

TEST_CASE("Vulkan mesh uploads active index ranges without changing pixels", "[core][render][components][native][vulkan][mesh-upload-ranges]") {
    CheckMeshUploadRanges<DeviceType::Vulkan>();
}
#endif
