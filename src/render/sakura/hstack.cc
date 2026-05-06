#include <jetstream/render/sakura/hstack.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct HStack::Impl {
    Config config;
};

HStack::HStack() {
    this->impl = std::make_unique<Impl>();
}

HStack::~HStack() = default;
HStack::HStack(HStack&&) noexcept = default;
HStack& HStack::operator=(HStack&&) noexcept = default;

bool HStack::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void HStack::render(const Context& ctx, Children children) const {
    const auto& config = this->impl->config;

    ImGui::PushID(config.id.c_str());

    for (U64 i = 0; i < children.size(); ++i) {
        children[i](ctx);

        if (i + 1 < children.size()) {
            ImGui::SameLine();
            if (config.spacing > 0.0f) {
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + Scale(ctx, config.spacing));
            }
        }
    }
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
