#include <jetstream/render/sakura/menu/context_menu.hh>

#include "../base.hh"

namespace Jetstream::Sakura {

struct ContextMenu::Impl {
    Config config;
    bool opening = false;
    bool visible = false;
};

ContextMenu::ContextMenu() {
    this->impl = std::make_unique<Impl>();
}

ContextMenu::~ContextMenu() = default;
ContextMenu::ContextMenu(ContextMenu&&) noexcept = default;
ContextMenu& ContextMenu::operator=(ContextMenu&&) noexcept = default;

bool ContextMenu::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void ContextMenu::render(const Context& ctx, Child child) {
    auto& impl = *this->impl;
    const auto& config = impl.config;

    if (!impl.opening) {
        ImGui::OpenPopup(config.id.c_str());
        ImGui::SetNextWindowPos(ImGui::GetMousePos(), ImGuiCond_Always);
        impl.opening = true;
    }

    if (config.size.has_value()) {
        ImGui::SetNextWindowSize(Private::ToImVec2(Scale(ctx, *config.size)), ImGuiCond_Always);
    }

    int styleVarCount = 0;
    if (config.padding.has_value()) {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, Private::ToImVec2(Scale(ctx, *config.padding)));
        ++styleVarCount;
    }
    if (config.rounding.has_value()) {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, Scale(ctx, *config.rounding));
        ++styleVarCount;
    }
    if (config.borderSize.has_value()) {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, Scale(ctx, *config.borderSize));
        ++styleVarCount;
    }
    const bool open = ImGui::BeginPopup(config.id.c_str());
    if (styleVarCount > 0) {
        ImGui::PopStyleVar(styleVarCount);
    }

    if (open) {
        impl.visible = true;
        if (child) {
            child(ctx);
        }
        ImGui::EndPopup();
    } else if (impl.visible) {
        impl.visible = false;
        impl.opening = false;
        if (config.onClose) {
            config.onClose();
        }
    } else {
        impl.opening = false;
    }
}

}  // namespace Jetstream::Sakura
