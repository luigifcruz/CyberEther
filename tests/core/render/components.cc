#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <new>
#include <utility>

#include "jetstream/backend/base.hh"
#include "jetstream/render/components/image.hh"
#include "jetstream/render/components/shapes.hh"
#include "jetstream/render/components/text.hh"
#include "../../../resources/fonts/compressed_jbmm.hh"
#include "component_test_support.hh"

using namespace Jetstream;
using namespace Jetstream::Render::Components;

namespace {

thread_local int allocationCountdown = -1;

std::pair<Result, bool> FailAllocation(int countdown, auto&& operation) {
    allocationCountdown = countdown;
    try {
        const Result result = operation();
        const bool injected = allocationCountdown < 0;
        allocationCountdown = -1;
        return {result, injected};
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
    int errors = 0;
    bool exhausted = false;
    for (int allocation = 0; allocation < 128; ++allocation) {
        CAPTURE(allocation);
        update(allocation);
        const auto [result, injected] = FailAllocation(allocation, [&] { return component.present(); });
        if (result == Result::ERROR) {
            REQUIRE(injected);
            ++errors;
            REQUIRE(component.present() == Result::SUCCESS);
        } else {
            REQUIRE(result == Result::SUCCESS);
        }
        // Successful notification-allocation recovery must retain every upload
        // without a retry that could conceal lost work.
        {
            Render::Transfer::Batch batch;
            REQUIRE(window.collect(batch) == Result::SUCCESS);
            CHECK(batch.buffers().size() == buffers);
        }
        driver.render(window);
        REQUIRE(component.present() == Result::SUCCESS);
        NoUploads(window);
        if (!injected) {
            exhausted = true;
            break;
        }
    }
    REQUIRE(exhausted);
    CHECK(errors == failures);
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
void CheckNativeComponents() {
    if (!std::getenv("JETSTREAM_TEST_NATIVE_COMPONENTS")) {
        SKIP("Set JETSTREAM_TEST_NATIVE_COMPONENTS=1 to run headless GPU regressions.");
    }
    REQUIRE(Backend::Initialize<D>(Backend::Config{.validationEnabled = false, .headless = true}) == Result::SUCCESS);
    ComponentDriver<D> driver;
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
TEST_CASE("Metal components retain state and retry failed uploads", "[core][render][components][native][metal]") {
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    CheckNativeComponents<DeviceType::Metal>();
}

#endif

#ifdef JETSTREAM_RENDER_VULKAN_AVAILABLE
TEST_CASE("Vulkan surfaces report clears and redraws but not retained idle frames",
          "[core][render][native][vulkan][idle-submit]") {
    if (!std::getenv("JETSTREAM_TEST_NATIVE_COMPONENTS")) {
        SKIP("Set JETSTREAM_TEST_NATIVE_COMPONENTS=1 to run headless GPU regressions.");
    }
    constexpr auto device = DeviceType::Vulkan;
    REQUIRE(Backend::Initialize<device>(Backend::Config{
        .validationEnabled = false, .headless = true}) == Result::SUCCESS);
    ComponentDriver<device> driver;
    ComponentWindow<device> window;
    REQUIRE(window.create() == Result::SUCCESS);
    auto texture = Frame<device>();
    auto surface = std::make_shared<ComponentSurface<device>>(Render::Surface::Config{
        .framebuffer = texture, .clearColor = {1, 0, 0, 1}, .retained = true,
    });
    REQUIRE(window.bind(surface) == Result::SUCCESS);

    driver.render(window);
    CHECK(surface->hasCommands());
    CHECK(Pixel(driver.pixels(texture), 32, 32, 0) > 0.99f);
    for (U64 frame = 0; frame < 8; ++frame) {
        driver.render(window);
        CHECK_FALSE(surface->hasCommands());
    }

    surface->clearColor({0, 1, 0, 1});
    for (U64 frame = 0; frame < 8; ++frame) {
        driver.render(window, surface.get());
        CHECK_FALSE(surface->hasCommands());
    }
    CHECK(Pixel(driver.pixels(texture), 32, 32, 0) > 0.99f);
    driver.render(window);
    CHECK(surface->hasCommands());
    CHECK(Pixel(driver.pixels(texture), 32, 32, 1) > 0.99f);
    driver.render(window);
    CHECK_FALSE(surface->hasCommands());

    surface->size({32, 32});
    driver.render(window, surface.get());
    CHECK(surface->hasCommands());
    driver.render(window);
    CHECK_FALSE(surface->hasCommands());
    ReleaseComponentSurface(window, surface);
    REQUIRE(window.destroy() == Result::SUCCESS);
}

TEST_CASE("Vulkan hidden compute work still reports encoded commands",
          "[core][render][native][vulkan][idle-submit]") {
    if (!std::getenv("JETSTREAM_TEST_NATIVE_COMPONENTS")) {
        SKIP("Set JETSTREAM_TEST_NATIVE_COMPONENTS=1 to run headless GPU regressions.");
    }
    constexpr auto device = DeviceType::Vulkan;
    REQUIRE(Backend::Initialize<device>(Backend::Config{
        .validationEnabled = false, .headless = true}) == Result::SUCCESS);
    ComponentDriver<device> driver;
    ComponentWindow<device> window;
    REQUIRE(window.create() == Result::SUCCESS);
    Shapes::Config config{.pixelSize = {2.0f / 64, 2.0f / 64}};
    config.elements["rect"] = {
        .color = {1, 0, 0, 1}, .size = {32, 32},
    };
    Shapes shapes(config);
    REQUIRE(shapes.create(&window) == Result::SUCCESS);
    auto texture = Frame<device>();
    auto surface = Attach(window, shapes, texture);
    REQUIRE(shapes.present() == Result::SUCCESS);
    driver.render(window);
    CHECK(surface->hasCommands());
    const auto baseline = driver.pixels(texture);
    REQUIRE(Pixel(baseline, 32, 32, 0) > 0.99f);
    driver.render(window);
    CHECK_FALSE(surface->hasCommands());

    std::span<ColorRGBA<F32>> colors;
    REQUIRE(shapes.getColors("rect", colors) == Result::SUCCESS);
    colors[0] = {0, 1, 0, 1};
    REQUIRE(shapes.updateColors("rect") == Result::SUCCESS);
    REQUIRE(shapes.present() == Result::SUCCESS);
    driver.render(window, surface.get());
    CHECK(surface->hasCommands());
    CHECK(driver.pixels(texture) == baseline);
    driver.render(window, surface.get());
    CHECK_FALSE(surface->hasCommands());
    driver.render(window);
    CHECK(surface->hasCommands());
    CHECK(driver.pixels(texture) != baseline);
    ReleaseComponentSurface(window, surface);
    REQUIRE(shapes.destroy(&window) == Result::SUCCESS);
    REQUIRE(window.destroy() == Result::SUCCESS);
}

TEST_CASE("Vulkan components retain state and retry failed uploads", "[core][render][components][native][vulkan]") {
    CheckNativeComponents<DeviceType::Vulkan>();
}

#endif
