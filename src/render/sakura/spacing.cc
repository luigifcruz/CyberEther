#include <jetstream/render/sakura/spacing.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct Spacing::Impl {
    Config config;
};

Spacing::Spacing() {
    this->impl = std::make_unique<Impl>();
}

Spacing::~Spacing() = default;
Spacing::Spacing(Spacing&&) noexcept = default;
Spacing& Spacing::operator=(Spacing&&) noexcept = default;

bool Spacing::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void Spacing::render(const Context& ctx) const {
    (void)ctx;
    const auto& config = this->impl->config;

    for (U64 i = 0; i < config.lines; ++i) {
        ImGui::Spacing();
    }
}

}  // namespace Jetstream::Sakura
