#include <jetstream/render/sakura/vstack.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct VStack::Impl {
    Config config;
};

VStack::VStack() {
    this->impl = std::make_unique<Impl>();
}

VStack::~VStack() = default;
VStack::VStack(VStack&&) noexcept = default;
VStack& VStack::operator=(VStack&&) noexcept = default;

bool VStack::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void VStack::render(const Context& ctx, Children children) const {
    const auto& config = this->impl->config;

    ImGui::PushID(config.id.c_str());

    for (U64 i = 0; i < children.size(); ++i) {
        children[i](ctx);

        if (config.spacing > 0.0f && i + 1 < children.size()) {
            ImGui::Dummy(Private::ToImVec2({0.0f, Scale(ctx, config.spacing)}));
        }
    }

    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
