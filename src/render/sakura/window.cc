#include <jetstream/render/sakura/window.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct Window::Impl {
    Config config;
    bool open = false;
    std::string windowId;
};

Window::Window() {
    this->impl = std::make_unique<Impl>();
}

Window::~Window() = default;
Window::Window(Window&&) noexcept = default;
Window& Window::operator=(Window&&) noexcept = default;

bool Window::update(Config config) {
    this->impl->windowId = config.title + "###" + config.id;
    this->impl->config = std::move(config);
    return true;
}

void Window::render(const Context& ctx, Child content) {
    auto& impl = *this->impl;
    const auto& config = impl.config;

    if (config.dockId.has_value() && *config.dockId != 0) {
        ImGui::SetNextWindowDockID(static_cast<ImGuiID>(*config.dockId), ImGuiCond_FirstUseEver);
    }

    ImGui::SetNextWindowSize(Private::ToImVec2(Scale(ctx, config.size)), ImGuiCond_FirstUseEver);

    I32 styleVarCount = 0;
    if (config.padding.has_value()) {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, Private::ToImVec2(Scale(ctx, *config.padding)));
        ++styleVarCount;
    }

    bool nextOpen = true;
    const bool expanded = ImGui::Begin(impl.windowId.c_str(), &nextOpen);
    if (styleVarCount > 0) {
        ImGui::PopStyleVar(styleVarCount);
    }
    if (!nextOpen) {
        ImGui::End();
        if (impl.open) {
            if (config.onClose) {
                config.onClose();
            }
            impl.open = false;
        }
        return;
    }

    if (!impl.open) {
        if (config.onOpen) {
            config.onOpen();
        }
        impl.open = true;
    }
    if (expanded && content) {
        content(ctx);
    }
    ImGui::End();
}

}  // namespace Jetstream::Sakura
