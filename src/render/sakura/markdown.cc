#include <jetstream/render/sakura/markdown.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct Markdown::Impl {
    Config config;
};

Markdown::Markdown() {
    this->impl = std::make_unique<Impl>();
}

Markdown::~Markdown() = default;
Markdown::Markdown(Markdown&&) noexcept = default;
Markdown& Markdown::operator=(Markdown&&) noexcept = default;

bool Markdown::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void Markdown::render(const Context& ctx) const {
    const auto& config = this->impl->config;

    ImGui::PushID(config.id.c_str());
    const auto* markdownConfig = Private::NativeMarkdownConfig(ctx.markdownConfig);
    if (markdownConfig) {
        ImGui::Markdown(config.value.c_str(), config.value.length(), *markdownConfig);
    } else {
        ImGui::TextWrapped("%s", config.value.c_str());
    }
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
