#include <jetstream/render/sakura/progress_bar.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct ProgressBar::Impl {
    Config config;
};

ProgressBar::ProgressBar() {
    this->impl = std::make_unique<Impl>();
}

ProgressBar::~ProgressBar() = default;
ProgressBar::ProgressBar(ProgressBar&&) noexcept = default;
ProgressBar& ProgressBar::operator=(ProgressBar&&) noexcept = default;

bool ProgressBar::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void ProgressBar::render(const Context& ctx) const {
    const auto& config = this->impl->config;

    ImGui::PushID(config.id.c_str());
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, Private::ImColor(ctx, config.colorKey));
    ImGui::ProgressBar(config.value, Private::ToImVec2(Scale(ctx, config.size)), config.overlay.c_str());
    ImGui::PopStyleColor();
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
