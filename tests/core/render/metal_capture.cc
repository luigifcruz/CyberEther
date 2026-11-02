#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "jetstream/render/devices/metal/window.hh"
#include "jetstream/viewport/capture/metal.hh"
#include "jetstream/viewport/platforms/headless/metal.hh"

#include <future>

using namespace Jetstream;
using namespace std::chrono_literals;

namespace {

struct Capture {
    std::shared_ptr<Viewport::Adapter<DeviceType::Metal>> viewport;
    std::unique_ptr<Render::WindowImp<DeviceType::Metal>> window;
    Viewport::FrameCaptureMetal capture;

    explicit Capture(std::shared_ptr<Viewport::Adapter<DeviceType::Metal>> viewport = {}) : viewport(viewport) {
        REQUIRE(Backend::Initialize<DeviceType::Metal>({.headless = true}) == Result::SUCCESS);
        if (!this->viewport) {
            this->viewport = std::make_shared<Viewport::Headless<DeviceType::Metal>>(
                Viewport::Config{.size = {65, 8}, .framerate = 1000});
        }
        REQUIRE(this->viewport->create() == Result::SUCCESS);
        window = std::make_unique<Render::WindowImp<DeviceType::Metal>>(Render::Window::Config{}, this->viewport);
        REQUIRE(window->create() == Result::SUCCESS);
        REQUIRE(window->start() == Result::SUCCESS);
        REQUIRE(capture.create(this->viewport.get()) == Result::SUCCESS);
    }

    void render(const ColorRGBA<F32>& color) {
        REQUIRE(capture.captureFrame() == Result::SUCCESS);
        REQUIRE(window->begin() == Result::SUCCESS);
        window->frameClearColor(color);
        REQUIRE(window->end() == Result::SUCCESS);
        REQUIRE(window->synchronize() == Result::SUCCESS);
    }

    Tensor read() {
        Tensor frame;
        auto pending = std::async(std::launch::async, [&] { return capture.getFrameData(frame); });
        const bool ready = pending.wait_for(3s) == std::future_status::ready;
        if (!ready) capture.stop();
        REQUIRE(ready);
        REQUIRE(pending.get() == Result::SUCCESS);
        return frame;
    }

    ~Capture() {
        capture.destroy();
        window->stop();
        window->destroy();
        viewport->destroy();
    }
};

void CheckColor(const Tensor& frame, U8 b, U8 g, U8 r) {
    REQUIRE(frame.shape() == Shape{8, 65, 4});
    REQUIRE(frame.device() == DeviceType::CPU);
    REQUIRE(frame.nativeDevice() == DeviceType::Metal);
    REQUIRE(frame.contiguous());
    for (U64 y : {0, 7}) {
        for (U64 x : {0, 64}) {
            CHECK(frame.at<U8>(y, x, 0) == b);
            CHECK(frame.at<U8>(y, x, 1) == g);
            CHECK(frame.at<U8>(y, x, 2) == r);
            CHECK(frame.at<U8>(y, x, 3) == 255);
        }
    }
}

}  // namespace

TEST_CASE("Metal remote capture preserves BGRA pixels and held frames under backpressure",
          "[core][render][metal][capture]") {
    Capture fixture;
    fixture.render({1, 0, 0, 1});
    auto first = fixture.read();
    CheckColor(first, 0, 0, 255);

    fixture.render({0, 1, 0, 1});
    fixture.render({0, 0, 1, 1});
    CheckColor(first, 0, 0, 255);
    REQUIRE(fixture.capture.releaseFrame() == Result::SUCCESS);
    auto second = fixture.read();
    CheckColor(second, 0, 255, 0);
    REQUIRE(fixture.capture.releaseFrame() == Result::SUCCESS);

    fixture.render({0, 0, 1, 1});
    CheckColor(fixture.read(), 255, 0, 0);
    REQUIRE(fixture.capture.releaseFrame() == Result::SUCCESS);
}

TEST_CASE("Metal remote capture copies a fixed extent from larger render targets",
          "[core][render][metal][capture]") {
    Capture fixture;
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto* device = Backend::State<DeviceType::Metal>()->getDevice();
    auto* descriptor = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatBGRA8Unorm, 130, 16, false);
    descriptor->setStorageMode(MTL::StorageModeShared);
    auto texture = NS::TransferPtr(device->newTexture(descriptor));
    REQUIRE(texture);
    std::vector<U8> pixels(130 * 16 * 4);
    for (U64 y = 0; y < 16; ++y) {
        for (U64 x = 0; x < 130; ++x) {
            pixels[(y * 130 + x) * 4] = x;
            pixels[(y * 130 + x) * 4 + 1] = y;
            pixels[(y * 130 + x) * 4 + 3] = 255;
        }
    }
    texture->replaceRegion(MTL::Region(0, 0, 130, 16), 0, pixels.data(), 130 * 4);
    auto queue = NS::TransferPtr(device->newCommandQueue());
    auto* command = queue->commandBuffer();
    REQUIRE(fixture.capture.captureFrame() == Result::SUCCESS);
    REQUIRE(fixture.viewport->encodeCapture(command, texture.get()) == Result::SUCCESS);
    command->commit();
    auto frame = fixture.read();
    REQUIRE(frame.shape() == Shape{8, 65, 4});
    CHECK(frame.at<U8>(7, 64, 0) == 64);
    CHECK(frame.at<U8>(7, 64, 1) == 7);
    CHECK(frame.at<U8>(7, 64, 3) == 255);
    REQUIRE(fixture.capture.releaseFrame() == Result::SUCCESS);
}

TEST_CASE("Metal remote capture rejects unsupported render target formats",
          "[core][render][metal][capture]") {
    Capture fixture;
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto* device = Backend::State<DeviceType::Metal>()->getDevice();
    auto* descriptor = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatRGBA8Unorm, 65, 8, false);
    auto texture = NS::TransferPtr(device->newTexture(descriptor));
    REQUIRE(texture);
    auto queue = NS::TransferPtr(device->newCommandQueue());
    auto* command = queue->commandBuffer();
    REQUIRE(fixture.capture.captureFrame() == Result::SUCCESS);
    REQUIRE(fixture.viewport->encodeCapture(command, texture.get()) == Result::ERROR);

    fixture.render({1, 0, 0, 1});
    CheckColor(fixture.read(), 0, 0, 255);
    REQUIRE(fixture.capture.releaseFrame() == Result::SUCCESS);
}

TEST_CASE("Metal rendering survives a smaller drawable and capture resumes when restored",
          "[core][render][metal][capture]") {
    const bool shrinkWidth = GENERATE(false, true);
    struct Viewport : Jetstream::Viewport::Headless<DeviceType::Metal> {
        Viewport() : Headless({.size = {65, 8}, .framerate = 1000}) {}

        MTL::Texture* nextDrawable(CA::MetalDrawable*& drawable) {
            auto* texture = Headless::nextDrawable(drawable);
            return smallerTexture ? smallerTexture.get() : texture;
        }

        NS::SharedPtr<MTL::Texture> smallerTexture;
    };

    auto viewport = std::make_shared<Viewport>();
    Capture fixture(viewport);
    fixture.render({1, 0, 0, 1});
    CheckColor(fixture.read(), 0, 0, 255);
    REQUIRE(fixture.capture.releaseFrame() == Result::SUCCESS);

    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto* device = Backend::State<DeviceType::Metal>()->getDevice();
    auto* descriptor = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatBGRA8Unorm, shrinkWidth ? 64 : 65, shrinkWidth ? 8 : 7, false);
    descriptor->setStorageMode(MTL::StorageModePrivate);
    descriptor->setUsage(MTL::TextureUsageRenderTarget);
    viewport->smallerTexture = NS::TransferPtr(device->newTexture(descriptor));
    REQUIRE(viewport->smallerTexture);
    for (unsigned i = 0; i < 3; ++i) {
        fixture.render({0, 0, 1, 1});
    }

    viewport->smallerTexture = {};
    fixture.render({0, 1, 0, 1});
    CheckColor(fixture.read(), 0, 255, 0);
    REQUIRE(fixture.capture.releaseFrame() == Result::SUCCESS);
}

TEST_CASE("Metal remote capture wakes on stop and can restart with GPU work in flight",
          "[core][render][metal][capture]") {
    Capture fixture;
    Tensor frame;
    auto pending = std::async(std::launch::async, [&] { return fixture.capture.getFrameData(frame); });
    REQUIRE(pending.wait_for(20ms) == std::future_status::timeout);
    REQUIRE(fixture.capture.stop() == Result::SUCCESS);
    REQUIRE(pending.wait_for(3s) == std::future_status::ready);
    REQUIRE(pending.get() == Result::ERROR);

    REQUIRE(fixture.capture.destroy() == Result::SUCCESS);
    REQUIRE(fixture.capture.create(fixture.viewport.get()) == Result::SUCCESS);
    REQUIRE(fixture.capture.captureFrame() == Result::SUCCESS);
    REQUIRE(fixture.window->begin() == Result::SUCCESS);
    REQUIRE(fixture.window->end() == Result::SUCCESS);
    REQUIRE(fixture.capture.destroy() == Result::SUCCESS);
    REQUIRE(fixture.capture.destroy() == Result::SUCCESS);
    REQUIRE(fixture.capture.create(fixture.viewport.get()) == Result::SUCCESS);
    fixture.render({1, 0, 1, 1});
    CheckColor(fixture.read(), 255, 0, 255);
    REQUIRE(fixture.capture.releaseFrame() == Result::SUCCESS);
}

#ifdef JETSTREAM_VIEWPORT_GLFW_AVAILABLE
TEST_CASE("Metal windowed capture enables readback only while streaming",
          "[.][core][render][metal][capture][glfw]") {
    struct Viewport : Jetstream::Viewport::GLFW<DeviceType::Metal> {
        Viewport() : GLFW({.size = {65, 8}}) {}

        MTL::Texture* nextDrawable(CA::MetalDrawable*& drawable) {
            auto* texture = GLFW::nextDrawable(drawable);
            if (texture) framebufferOnly = texture->framebufferOnly();
            return texture;
        }

        bool framebufferOnly = false;
    };

    auto viewport = std::make_shared<Viewport>();
    Capture fixture(viewport);
    REQUIRE(fixture.capture.destroy() == Result::SUCCESS);
    fixture.render({0, 0, 0, 1});
    REQUIRE(viewport->framebufferOnly);

    REQUIRE(fixture.capture.create(viewport.get()) == Result::SUCCESS);
    fixture.render({1, 0, 0, 1});
    REQUIRE_FALSE(viewport->framebufferOnly);
    auto frame = fixture.read();
    CHECK(frame.at<U8>(0, 0, 0) == 0);
    CHECK(frame.at<U8>(0, 0, 2) == 255);
    REQUIRE(fixture.capture.releaseFrame() == Result::SUCCESS);

    REQUIRE(fixture.capture.destroy() == Result::SUCCESS);
    fixture.render({0, 0, 0, 1});
    REQUIRE(viewport->framebufferOnly);
}
#endif
