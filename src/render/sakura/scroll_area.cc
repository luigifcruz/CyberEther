#include <jetstream/render/sakura/scroll_area.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct ScrollArea::Impl {
    Config config;
};

ScrollArea::ScrollArea() {
    this->impl = std::make_unique<Impl>();
}

ScrollArea::~ScrollArea() = default;
ScrollArea::ScrollArea(ScrollArea&&) noexcept = default;
ScrollArea& ScrollArea::operator=(ScrollArea&&) noexcept = default;

bool ScrollArea::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void ScrollArea::render(const Context& ctx, Child child) const {
    const auto& config = this->impl->config;

    ImGui::PushID(config.id.c_str());
    if (ImGui::BeginChild(config.id.c_str(),
                          Private::ToImVec2({0.0f, Scale(ctx, config.height)}),
                          ImGuiChildFlags_Borders,
                          ImGuiWindowFlags_AlwaysVerticalScrollbar)) {
        if (child) {
            child(ctx);
        }
    }
    ImGui::EndChild();
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
