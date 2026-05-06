#include <jetstream/render/sakura/node/loading_bar.hh>

#include <jetstream/render/sakura/spacing.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct NodeLoadingBar::Impl {
    Config config;
    Spacing spacing;
};

NodeLoadingBar::NodeLoadingBar() {
    this->impl = std::make_unique<Impl>();
}

NodeLoadingBar::~NodeLoadingBar() = default;
NodeLoadingBar::NodeLoadingBar(NodeLoadingBar&&) noexcept = default;
NodeLoadingBar& NodeLoadingBar::operator=(NodeLoadingBar&&) noexcept = default;

bool NodeLoadingBar::update(Config config) {
    this->impl->config = std::move(config);
    this->impl->spacing.update({
        .id = this->impl->config.id + "Spacing",
    });
    return true;
}

void NodeLoadingBar::render(const Context& ctx) const {
    const auto& impl = *this->impl;
    const auto& config = impl.config;

    ImGui::PushID(config.id.c_str());
    impl.spacing.render(ctx);

    const F32 width = ImGui::GetContentRegionAvail().x;
    const F32 height = Scale(ctx, config.height);
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const ImVec4 color = Private::ImColor(ctx, "node_outline_pending");
    const float time = static_cast<float>(ImGui::GetTime());
    const float t = (sinf(time * 3.0f) + 1.0f) * 0.5f;
    const float glowWidth = width * 0.4f;
    const float centerX = pos.x + t * width;
    ImDrawList* drawList = ImGui::GetWindowDrawList();

    for (float x = pos.x; x < pos.x + width; x += 1.0f) {
        const float dist = fabsf(x - centerX);
        const float alpha = fmaxf(0.0f, 1.0f - (dist / (glowWidth * 0.5f)));
        const float glow = alpha * alpha * alpha;
        if (glow > 0.01f) {
            drawList->AddLine(ImVec2(x, pos.y),
                              ImVec2(x, pos.y + height),
                              IM_COL32(static_cast<int>(color.x * 255),
                                       static_cast<int>(color.y * 255),
                                       static_cast<int>(color.z * 255),
                                       static_cast<int>(glow * 255)));
        }
    }
    ImGui::Dummy(ImVec2(width, height));
    impl.spacing.render(ctx);
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
