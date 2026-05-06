#include <jetstream/render/sakura/grid.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct Grid::Impl {
    Config config;
};

Grid::Grid() {
    this->impl = std::make_unique<Impl>();
}

Grid::~Grid() = default;
Grid::Grid(Grid&&) noexcept = default;
Grid& Grid::operator=(Grid&&) noexcept = default;

bool Grid::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void Grid::render(const Context& ctx, Children children) const {
    const auto& config = this->impl->config;

    ImGui::PushID(config.id.c_str());
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, Private::ToImVec2(Scale(ctx, config.cellPadding)));
    if (ImGui::BeginTable(config.id.c_str(),
                          std::max<U64>(1, config.columns),
                          ImGuiTableFlags_SizingStretchSame |
                          ImGuiTableFlags_PadOuterX |
                          ImGuiTableFlags_NoBordersInBody |
                          ImGuiTableFlags_NoBordersInBodyUntilResize |
                          ImGuiTableFlags_ScrollY,
                          Private::ToImVec2(Scale(ctx, config.size)))) {
        for (U64 i = 0; i < config.columns; ++i) {
            const std::string columnId = "##grid-column" + std::to_string(i);
            ImGui::TableSetupColumn(columnId.c_str(), ImGuiTableColumnFlags_WidthStretch);
        }

        for (U64 i = 0; i < children.size(); ++i) {
            ImGui::TableNextColumn();
            if (children[i]) {
                children[i](ctx);
            }
        }
        ImGui::EndTable();
    }
    ImGui::PopStyleVar();
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
