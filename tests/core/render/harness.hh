#ifndef JETSTREAM_TESTS_CORE_RENDER_HARNESS_HH
#define JETSTREAM_TESTS_CORE_RENDER_HARNESS_HH

#include "jetstream/render/base/input.hh"
#include "jetstream/render/base/window.hh"
#include "jetstream/viewport/adapters/generic.hh"

#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace RenderTest {

class TestPlatform final : public Jetstream::Viewport::Generic::Platform {
 public:
    Jetstream::F64 time() const override { return clock; }
    void poll() override {}
    void setCursor(Jetstream::SurfaceCursor next) override { cursor = next; }
    std::string clipboardText() const override { return clipboard; }
    void setClipboardText(const std::string& text) override { clipboard = text; }

    Jetstream::F64 clock = 1.0;
    Jetstream::SurfaceCursor cursor = Jetstream::SurfaceCursor::Default;
    std::string clipboard;
};

class Viewport final : public Jetstream::Viewport::Generic {
 public:
    Viewport() : Generic(Jetstream::Viewport::Config{.size = {0, 0}}) {
        auto platform = std::make_unique<TestPlatform>();
        state = platform.get();
        attachPlatform(std::move(platform));
    }

    std::string id() const override { return "test"; }
    std::string name() const override { return "Test"; }
    Jetstream::DeviceType device() const override { return Jetstream::DeviceType::None; }

    Jetstream::Result create() override { return Jetstream::Result::SUCCESS; }
    Jetstream::Result destroy() override { return Jetstream::Result::SUCCESS; }
    Jetstream::Result waitEvents() override { return Jetstream::Result::SUCCESS; }
    Jetstream::Result pollEvents() override { return Jetstream::Result::SUCCESS; }
    bool keepRunning() override { return true; }

    Jetstream::Extent2D<Jetstream::F32> displaySize() const override { return display; }
    Jetstream::Extent2D<Jetstream::F32> framebufferSize() const override {
        return {display.x * scale, display.y * scale};
    }
    Jetstream::Extent2D<Jetstream::F32> contentScale() const override { return content; }
    Jetstream::Render::WindowMetrics::Policy scalePolicy() const override { return policy; }
    bool macOSBehaviors() const override { return macOS; }
    bool retainedPresent() const override { return retained; }
    void waitIdle(Jetstream::F64) override { ++idleWaits; }

    Jetstream::Render::TitlebarChrome titlebarChrome() const override {
        return chrome.value_or(Generic::titlebarChrome());
    }
    void beginWindowDrag() override { ++drags; }
    void zoomWindow() override { ++zooms; }
    bool setDarkAppearance(bool dark) override {
        if (appearances.empty() || appearances.back() != dark) {
            appearances.push_back(dark);
        }
        return true;
    }

    TestPlatform& platform() { return *state; }

    Jetstream::Extent2D<Jetstream::F32> display = {0.0f, 0.0f};
    Jetstream::F32 scale = 1.0f;
    Jetstream::Extent2D<Jetstream::F32> content = {1.0f, 1.0f};
    Jetstream::Render::WindowMetrics::Policy policy = Jetstream::Render::WindowMetrics::Policy::Interface;
    std::optional<Jetstream::Render::TitlebarChrome> chrome;
    bool macOS = false;
    bool retained = true;
    Jetstream::U64 idleWaits = 0;
    Jetstream::U64 drags = 0;
    Jetstream::U64 zooms = 0;
    std::vector<bool> appearances;

 private:
    TestPlatform* state = nullptr;
};

class Window : public Jetstream::Render::Window {
 public:
    using Acquire = Jetstream::Render::Window::Acquire;
    using Jetstream::Render::Window::textureVisibleForPresentation;

    explicit Window(Jetstream::F32 scale = 1.0f) : Jetstream::Render::Window(Config{scale}) {}

    ~Window() override {
        close();
    }

    Jetstream::Result open() {
        if (const auto result = create(); result != Jetstream::Result::SUCCESS) {
            return result;
        }
        if (const auto result = start(); result != Jetstream::Result::SUCCESS) {
            return result;
        }
        started = true;
        return Jetstream::Result::SUCCESS;
    }

    void close() {
        if (started) {
            (void)stop();
            started = false;
        }
        if (opened) {
            (void)destroy();
            opened = false;
        }
    }

    Jetstream::Result frame() {
        if (!opened) {
            if (const auto result = open(); result != Jetstream::Result::SUCCESS) {
                return result;
            }
        }
        if (const auto result = begin(); result != Jetstream::Result::SUCCESS) {
            return result;
        }
        return end();
    }

    template<typename Body>
    void frame(Body body) {
        (void)begin();
        body();
        (void)end();
    }

    Jetstream::Result inject(const std::vector<Jetstream::Render::InputQueue::Event>& events,
                             Jetstream::F64 step = 1.0 / 60.0) {
        for (const auto& event : events) {
            viewport.input().push(event);
        }
        viewport.platform().clock += step;
        return frame();
    }

    std::string info() const override { return "RenderTest"; }
    constexpr Jetstream::DeviceType device() const override { return Jetstream::DeviceType::None; }

    mutable Viewport viewport;
    bool attached = true;
    std::deque<Acquire> acquisitions;
    Jetstream::U64 recreates = 0;
    Jetstream::U64 cancels = 0;
    std::optional<Jetstream::Result> failBegin;
    std::optional<Jetstream::Result> failEnd;
    bool throwEnd = false;

 protected:
    Jetstream::Viewport::Generic* underlyingViewport() const override { return attached ? &viewport : nullptr; }

    Jetstream::Result bindSurface(const std::shared_ptr<Jetstream::Render::Surface>&) override {
        return Jetstream::Result::SUCCESS;
    }
    Jetstream::Result unbindSurface(const std::shared_ptr<Jetstream::Render::Surface>&) override {
        return Jetstream::Result::SUCCESS;
    }

    Jetstream::Result underlyingCreate() override {
        opened = true;
        if (attached) {
            updateMetrics(viewport);
        }
        return Jetstream::Result::SUCCESS;
    }

    Jetstream::Result underlyingDestroy() override {
        opened = false;
        return Jetstream::Result::SUCCESS;
    }

    Jetstream::Result underlyingBegin() override {
        if (failBegin) {
            return std::exchange(failBegin, std::nullopt).value();
        }
        if (attached) {
            updateMetrics(viewport);
            updateInput(viewport);
        }
        return Jetstream::Result::SUCCESS;
    }

    Jetstream::Result underlyingEnd() override {
        if (std::exchange(throwEnd, false)) {
            throw Jetstream::Result::ERROR;
        }
        if (failEnd) {
            return std::exchange(failEnd, std::nullopt).value();
        }
        Jetstream::Render::Transfer::Batch transfers;
        if (const auto result = collectTransfers(transfers); result != Jetstream::Result::SUCCESS) {
            return result;
        }
        auto pass = beginPresent(transfers);
        if (pass.presenting && !acquisitions.empty()) {
            acquired(pass, acquisitions.front());
            acquisitions.pop_front();
        }
        transfers.commit();
        return finishPresent(pass);
    }

    Jetstream::Result underlyingRecreate() override {
        ++recreates;
        return Jetstream::Result::SUCCESS;
    }

    Jetstream::Result underlyingCancel() override {
        ++cancels;
        return underlyingRecreate();
    }

    Jetstream::Result underlyingSynchronize() override { return Jetstream::Result::SUCCESS; }

 private:
    bool opened = false;
    bool started = false;
};

template<typename Type, typename Events>
std::vector<Type> Only(const Events& events) {
    std::vector<Type> result;
    for (const auto& event : events) {
        if (const auto* typed = std::get_if<Type>(&event)) {
            result.push_back(*typed);
        }
    }
    return result;
}

}  // namespace RenderTest

#endif  // JETSTREAM_TESTS_CORE_RENDER_HARNESS_HH
