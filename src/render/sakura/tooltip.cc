#include <jetstream/render/sakura/tooltip.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct Tooltip::Impl {
    Config config;
};

Tooltip::Tooltip() {
    this->impl = std::make_unique<Impl>();
}

Tooltip::~Tooltip() = default;
Tooltip::Tooltip(Tooltip&&) noexcept = default;
Tooltip& Tooltip::operator=(Tooltip&&) noexcept = default;

bool Tooltip::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void Tooltip::render(const Context& ctx, Child child) const {
    const auto& config = this->impl->config;

    if (!child) {
        return;
    }

    const ImGuiHoveredFlags flags = config.delayed ? ImGuiHoveredFlags_DelayShort : ImGuiHoveredFlags_None;
    if (!config.visible && !ImGui::IsItemHovered(flags)) {
        return;
    }

    ImGui::PushID(config.id.c_str());
    ImGui::BeginTooltip();
    if (config.wrapWidth > 0.0f) {
        ImGui::PushTextWrapPos(Scale(ctx, config.wrapWidth));
    }
    child(ctx);
    if (config.wrapWidth > 0.0f) {
        ImGui::PopTextWrapPos();
    }
    ImGui::EndTooltip();
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
