#include <jetstream/render/sakura/debug_window.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct DebugWindow::Impl {
    Config config;
    F32 positionX = 0.0f;
    F32 directionX = 1.0f;
};

DebugWindow::DebugWindow() {
    this->impl = std::make_unique<Impl>();
}

DebugWindow::~DebugWindow() = default;
DebugWindow::DebugWindow(DebugWindow&&) noexcept = default;
DebugWindow& DebugWindow::operator=(DebugWindow&&) noexcept = default;

bool DebugWindow::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void DebugWindow::render(const Context& ctx) {
    auto& impl = *this->impl;
    const auto& config = impl.config;

    if (!config.visible) {
        return;
    }

    const ImGuiIO& io = ImGui::GetIO();
    const Extent2D<F32> size = Scale(ctx, config.size);
    impl.positionX += impl.directionX;

    if (impl.positionX > io.DisplaySize.x - size.x) {
        impl.directionX = -impl.directionX;
    }
    if (impl.positionX < 0.0f) {
        impl.directionX = -impl.directionX;
    }

    const std::string windowId = config.title + "###" + config.id;
    ImGui::SetNextWindowPos(ImVec2(impl.positionX, (io.DisplaySize.y * config.verticalRatio) - (size.y * 0.5f)),
                            ImGuiCond_Always);
    ImGui::SetNextWindowSize(Private::ToImVec2(size), ImGuiCond_Always);

    if (ImGui::Begin(windowId.c_str(), nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse)) {
        const U64 ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        ImGui::TextFormatted("Time: {} ms", ms);
    }
    ImGui::End();
}

}  // namespace Jetstream::Sakura
