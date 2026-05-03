#include <jetstream/render/sakura/notifications.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct Notifications::Impl {
    Config config;
};

Notifications::Notifications() {
    this->impl = std::make_unique<Impl>();
}

Notifications::~Notifications() = default;
Notifications::Notifications(Notifications&&) noexcept = default;
Notifications& Notifications::operator=(Notifications&&) noexcept = default;

bool Notifications::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void Notifications::render(const Context& ctx) const {
    const auto& config = this->impl->config;

    ImGui::PushID(config.id.c_str());
    ImGui::PushStyleColor(ImGuiCol_WindowBg, Private::ImColor(ctx, config.backgroundColorKey));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, Scale(ctx, config.rounding));
    ImGui::RenderNotifications();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
