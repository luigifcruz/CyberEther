#include <jetstream/render/sakura/split_view.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct SplitView::Impl {
    Config config;
};

SplitView::SplitView() {
    this->impl = std::make_unique<Impl>();
}

SplitView::~SplitView() = default;
SplitView::SplitView(SplitView&&) noexcept = default;
SplitView& SplitView::operator=(SplitView&&) noexcept = default;

bool SplitView::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void SplitView::render(const Context& ctx, Children children) const {
    const auto& config = this->impl->config;

    ImGui::PushID(config.id.c_str());
    Extent2D<F32> size = Scale(ctx, {0.0f, config.height});
    if (config.fillHeight) {
        size.y = std::max(0.0f, ImGui::GetContentRegionAvail().y - Scale(ctx, config.reservedHeight));
    }

    if (ImGui::BeginTable(config.id.c_str(),
                          2,
                          ImGuiTableFlags_SizingStretchProp |
                          ImGuiTableFlags_BordersInnerV |
                          ImGuiTableFlags_NoHostExtendY,
                          Private::ToImVec2(size))) {
        ImGui::TableSetupColumn("##split-left", ImGuiTableColumnFlags_WidthFixed, Scale(ctx, config.leftWidth));
        ImGui::TableSetupColumn("##split-right", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableNextRow();

        ImGui::TableSetColumnIndex(0);
        if (!children.empty() && children[0]) {
            children[0](ctx);
        }

        ImGui::TableSetColumnIndex(1);
        if (children.size() > 1 && children[1]) {
            children[1](ctx);
        }

        ImGui::EndTable();
    }
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
