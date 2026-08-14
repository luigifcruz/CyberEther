#include <jetstream/render/sakura/components/node/field_grid.hh>

#include "../../helpers.hh"

namespace Jetstream::Sakura {

struct NodeFieldGrid::Impl {
    Config config;
};

NodeFieldGrid::NodeFieldGrid() {
    this->impl = std::make_unique<Impl>();
}

NodeFieldGrid::~NodeFieldGrid() = default;
NodeFieldGrid::NodeFieldGrid(NodeFieldGrid&&) noexcept = default;
NodeFieldGrid& NodeFieldGrid::operator=(NodeFieldGrid&&) noexcept = default;

bool NodeFieldGrid::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void NodeFieldGrid::render(const Context& ctx, const std::vector<Item>& items) const {
    const F32 availWidth = ImGui::GetContentRegionAvail().x;
    const F32 columnGap = std::max(0.0f,
        ImGui::GetStyle().ItemSpacing.y - Scale(ctx, 6.0f));
    const F32 minColumnWidth = Scale(ctx, impl->config.minColumnWidth);
    const U64 columns = std::min<U64>(
        std::max<U64>(1, items.size()),
        (availWidth > 0.0f && minColumnWidth > 0.0f)
            ? std::max<U64>(1, static_cast<U64>(
                (availWidth + columnGap) / (minColumnWidth + columnGap)))
            : 1);

    ImGui::PushID(impl->config.id.c_str());
    for (U64 i = 0; i < items.size();) {
        if (columns == 1 || items[i].fullWidth) {
            if (items[i].child) {
                items[i].child(ctx);
            }
            ++i;
            continue;
        }

        const U64 groupStart = i;
        while (i < items.size() && !items[i].fullWidth) {
            ++i;
        }

        const U64 groupEnd = i;
        const U64 groupColumns = std::min(columns, groupEnd - groupStart);
        const F32 colWidth = (availWidth - (groupColumns - 1) * columnGap) /
                             groupColumns;
        const ImVec2 groupPos = ImGui::GetCursorScreenPos();
        F32 rowY = groupPos.y;
        F32 rowHeight = 0.0f;
        for (U64 j = groupStart; j < groupEnd; ++j) {
            const U64 column = (j - groupStart) % groupColumns;
            if (column == 0 && j != groupStart) {
                rowY += rowHeight + ImGui::GetStyle().ItemSpacing.y;
                rowHeight = 0.0f;
            }
            ImGui::SetCursorScreenPos(ImVec2(
                groupPos.x + column * (colWidth + columnGap), rowY));

            ImGui::PushID(static_cast<int>(j));
            ImGui::BeginGroup();
            ImGuiWindow* window = ImGui::GetCurrentWindow();
            const F32 contentMaxX = window->ContentRegionRect.Max.x;
            window->ContentRegionRect.Max.x = ImGui::GetCursorScreenPos().x + colWidth;
            const F32 slack = Scale(ctx, 8.0f);
            const ImRect clipRect = window->ClipRect;
            ImGui::PushClipRect(ImVec2(clipRect.Min.x, clipRect.Min.y - slack),
                                ImVec2(window->ContentRegionRect.Max.x,
                                       clipRect.Max.y + slack),
                                false);
            if (items[j].child) {
                items[j].child(ctx);
            }
            ImGui::PopClipRect();
            window->ContentRegionRect.Max.x = contentMaxX;
            ImGui::EndGroup();
            rowHeight = std::max(rowHeight, ImGui::GetItemRectSize().y);
            ImGui::PopID();
        }
        ImGui::SetCursorScreenPos(ImVec2(groupPos.x, rowY + rowHeight));
        ImGui::Dummy(ImVec2(0.0f, 0.0f));
    }
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
