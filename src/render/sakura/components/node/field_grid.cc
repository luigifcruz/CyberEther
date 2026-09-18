#include <jetstream/render/sakura/components/node/field_grid.hh>

#include <jetstream/render/sakura/components/node/field.hh>

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
    if (items.empty()) {
        return;
    }
    const F32 availWidth = ImGui::GetContentRegionAvail().x;
    const F32 gap = Scale(ctx, NodeField::Gap);
    const F32 minColumnWidth = Scale(ctx, impl->config.minColumnWidth);
    const U64 columns = std::min<U64>(
        std::max<U64>(1, items.size()),
        (availWidth > 0.0f && minColumnWidth > 0.0f)
            ? std::max<U64>(1, static_cast<U64>(
                (availWidth + gap) / (minColumnWidth + gap)))
            : 1);

    ImGui::PushID(impl->config.id.c_str());
    ImGui::BeginGroup();
    const auto endGroupWithGap = [gap]() {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(ImGui::GetStyle().ItemSpacing.x, gap));
        ImGui::EndGroup();
        ImGui::PopStyleVar();
    };
    for (U64 i = 0; i < items.size();) {
        if (columns == 1 || items[i].fullWidth) {
            ImGui::BeginGroup();
            if (items[i].child) {
                items[i].child(ctx);
            }
            endGroupWithGap();
            ++i;
            continue;
        }

        const U64 groupStart = i;
        while (i < items.size() && !items[i].fullWidth) {
            ++i;
        }

        const U64 groupEnd = i;
        const U64 groupColumns = std::min(columns, groupEnd - groupStart);
        const F32 colWidth = (availWidth - (groupColumns - 1) * gap) / groupColumns;
        ImGui::BeginGroup();
        const ImVec2 groupPos = ImGui::GetCursorScreenPos();
        F32 rowY = groupPos.y;
        F32 rowHeight = 0.0f;
        for (U64 j = groupStart; j < groupEnd; ++j) {
            const U64 column = (j - groupStart) % groupColumns;
            if (column == 0 && j != groupStart) {
                rowY += rowHeight + gap;
                rowHeight = 0.0f;
            }
            ImGui::SetCursorScreenPos(ImVec2(
                groupPos.x + column * (colWidth + gap), rowY));

            ImGui::PushID(static_cast<int>(j));
            ImGui::BeginGroup();
            ImGuiWindow* window = ImGui::GetCurrentWindow();
            const F32 contentMaxX = window->ContentRegionRect.Max.x;
            window->ContentRegionRect.Max.x = ImGui::GetCursorScreenPos().x + colWidth;
            const ImRect clipRect = window->ClipRect;
            // Columns must not expand the parent clip into neighboring dock panes.
            ImGui::PushClipRect(clipRect.Min,
                                ImVec2(window->ContentRegionRect.Max.x,
                                       clipRect.Max.y),
                                true);
            if (items[j].child) {
                items[j].child(ctx);
            }
            ImGui::PopClipRect();
            window->ContentRegionRect.Max.x = contentMaxX;
            ImGui::EndGroup();
            rowHeight = std::max(rowHeight, ImGui::GetItemRectSize().y);
            ImGui::PopID();
        }
        endGroupWithGap();
    }
    ImGui::EndGroup();
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
