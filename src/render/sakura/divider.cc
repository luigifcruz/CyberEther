#include <jetstream/render/sakura/divider.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct Divider::Impl {
    Config config;
};

Divider::Divider() {
    this->impl = std::make_unique<Impl>();
}

Divider::~Divider() = default;
Divider::Divider(Divider&&) noexcept = default;
Divider& Divider::operator=(Divider&&) noexcept = default;

bool Divider::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void Divider::render(const Context& ctx) const {
    const auto& config = this->impl->config;

    ImGui::PushID(config.id.c_str());

    if (config.spacing < 0.0f) {
        ImGui::Spacing();
    } else if (config.spacing > 0.0f) {
        ImGui::Dummy(Private::ToImVec2({0.0f, Scale(ctx, config.spacing)}));
    }

    if (config.separator) {
        ImGui::Separator();
    }

    if (config.spacing < 0.0f) {
        ImGui::Spacing();
    } else if (config.spacing > 0.0f) {
        ImGui::Dummy(Private::ToImVec2({0.0f, Scale(ctx, config.spacing)}));
    }

    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
