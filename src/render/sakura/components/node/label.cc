#include <jetstream/render/sakura/components/node/label.hh>

#include <jetstream/render/sakura/components/text.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct NodeLabel::Impl {
    Config config;
    Text label;
};

NodeLabel::NodeLabel() {
    this->impl = std::make_unique<Impl>();
}

NodeLabel::~NodeLabel() = default;
NodeLabel::NodeLabel(NodeLabel&&) noexcept = default;
NodeLabel& NodeLabel::operator=(NodeLabel&&) noexcept = default;

bool NodeLabel::update(Config config) {
    impl->config = std::move(config);
    impl->label.update({
        .id = impl->config.id,
        .str = impl->config.str,
        .font = impl->config.font,
        .tone = impl->config.tone,
        .align = impl->config.align,
        .wrapped = impl->config.wrapped,
        .clipped = impl->config.clipped,
        .scale = impl->config.scale,
    });
    return true;
}

void NodeLabel::render(const Context& ctx) const {
    if (!impl->config.boxed) {
        impl->label.render(ctx);
        return;
    }

    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const F32 width = std::max(0.0f, ImGui::GetContentRegionAvail().x);
    const F32 padding = Scale(ctx, 8.0f);
    const F32 paddingX = std::min(padding, width * 0.5f);
    const F32 contentRight = origin.x + width - paddingX;
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    ImDrawList* drawList = ImGui::GetWindowDrawList();

    // Keep the frame and text in the node's current canvas channel. A child
    // window would submit its text after all nodes, bypassing their depth order.
    ImDrawListSplitter splitter;
    splitter.Split(drawList, 2);
    splitter.SetCurrentChannel(drawList, 1);

    const F32 contentMaxX = window->ContentRegionRect.Max.x;
    const F32 workMaxX = window->WorkRect.Max.x;
    window->ContentRegionRect.Max.x = contentRight;
    window->WorkRect.Max.x = contentRight;
    ImGui::SetCursorScreenPos(ImVec2(origin.x + paddingX, origin.y + padding));
    ImGui::BeginGroup();
    ImGui::PushClipRect(ImVec2(origin.x + paddingX, window->ClipRect.Min.y),
                        ImVec2(contentRight, window->ClipRect.Max.y), true);
    impl->label.render(ctx);
    ImGui::PopClipRect();
    ImGui::EndGroup();
    const F32 height = ImGui::GetItemRectSize().y + 2.0f * padding;
    window->ContentRegionRect.Max.x = contentMaxX;
    window->WorkRect.Max.x = workMaxX;

    splitter.SetCurrentChannel(drawList, 0);
    const ImVec2 end(origin.x + width, origin.y + height);
    drawList->AddRectFilled(origin, end,
                            ImGui::ColorConvertFloat4ToU32(Private::ImColor(ctx, "card")),
                            Scale(ctx, 6.0f));
    drawList->AddRect(origin, end,
                      ImGui::ColorConvertFloat4ToU32(Private::ImColor(ctx, "table_border_strong")),
                      Scale(ctx, 6.0f));
    splitter.Merge(drawList);

    ImGui::SetCursorScreenPos(origin);
    ImGui::Dummy(ImVec2(width, height));
}

}  // namespace Jetstream::Sakura
