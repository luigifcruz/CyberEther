#include <jetstream/render/sakura/components/node/input_row.hh>

#include <jetstream/render/sakura/components/node/combo.hh>

#include "base.hh"

#include <algorithm>

namespace Jetstream::Sakura {

namespace {

constexpr F32 StackedRowSpacing = 3.0f;

}  // namespace

struct NodeInputRow::Impl {
    Config config;
    std::vector<NodeCombo> combos;
    bool stacked = false;

    void syncCombos() {
        combos.resize(config.combos.size());
        for (U64 i = 0; i < combos.size(); ++i) {
            const auto& combo = config.combos[i];
            combos[i].update({
                .id = config.id + ":combo" + std::to_string(i),
                .options = combo.options,
                .value = combo.value,
                .width = stacked ? 0.0f : combo.width,
                .onChange = combo.onChange,
            });
        }
    }
};

NodeInputRow::NodeInputRow() {
    this->impl = std::make_unique<Impl>();
}

NodeInputRow::~NodeInputRow() = default;
NodeInputRow::NodeInputRow(NodeInputRow&&) noexcept = default;
NodeInputRow& NodeInputRow::operator=(NodeInputRow&&) noexcept = default;

bool NodeInputRow::update(Config config) {
    impl->config = std::move(config);
    impl->syncCombos();
    return true;
}

void NodeInputRow::render(const Context& ctx) const {
    auto& impl = *this->impl;
    const auto& config = impl.config;

    const ImGuiStyle& style = ImGui::GetStyle();
    const F32 spacing = style.ItemSpacing.x;
    const F32 minShapeWidth = Scale(ctx, config.shapeMinWidth);
    const F32 minAxesWidth = config.onAxesChange ? Scale(ctx, config.axesMinWidth) : 0.0f;
    const F32 startX = ImGui::GetCursorPosX();
    const F32 available = ImGui::GetContentRegionAvail().x;

    F32 combosWidth = 0.0f;
    for (const auto& combo : config.combos) {
        combosWidth += Scale(ctx, combo.width) + spacing;
    }

    const F32 fieldsSpacing = config.onAxesChange ? spacing : 0.0f;
    const bool nextStacked =
        available < minShapeWidth + minAxesWidth + combosWidth + fieldsSpacing;
    if (nextStacked != impl.stacked) {
        impl.stacked = nextStacked;
        impl.syncCombos();
    }

    const F32 axesWidth = impl.stacked ? 0.0f : minAxesWidth;
    const F32 shapeWidth = impl.stacked
        ? available
        : std::max(minShapeWidth, available - combosWidth - axesWidth - (config.onAxesChange ? spacing : 0.0f));

    ImGui::PushID(config.id.c_str());
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(Scale(ctx, 6.0f), Scale(ctx, 3.0f)));

    const F32 frameHeight = ImGui::GetFrameHeight();
    const F32 stackedSpacingY = Scale(ctx, StackedRowSpacing);
    const F32 rowCount = static_cast<F32>(config.combos.size()) + (config.onAxesChange ? 2.0f : 1.0f);
    const F32 backgroundHeight = impl.stacked
        ? frameHeight * rowCount + stackedSpacingY * (rowCount - 1.0f)
        : frameHeight;
    const ImVec2 rowMin = ImGui::GetCursorScreenPos();
    ImGui::GetWindowDrawList()->AddRectFilled(
        rowMin,
        ImVec2(rowMin.x + available, rowMin.y + backgroundHeight),
        ImGui::GetColorU32(ImGuiCol_FrameBg),
        style.FrameRounding);

    std::string shape = config.shape;
    ImGui::SetNextItemWidth(shapeWidth);
    bool changed = ImGui::InputTextWithHint("##input",
                                            config.shapeHint.c_str(),
                                            &shape,
                                            ImGuiInputTextFlags_EnterReturnsTrue);
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        changed = true;
    }
    if (changed && config.onShapeChange) {
        config.onShapeChange(shape);
    }

    if (config.onAxesChange) {
        std::string axes = config.axes;
        if (impl.stacked) {
            const F32 stackedPull = style.ItemSpacing.y - stackedSpacingY;
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() - stackedPull);
            ImGui::SetNextItemWidth(available);
        } else {
            ImGui::SameLine();
            ImGui::SetCursorPosX(startX + shapeWidth + spacing);
            ImGui::SetNextItemWidth(axesWidth);
        }
        bool axesChanged = ImGui::InputTextWithHint("##axes",
                                                    config.axesHint.c_str(),
                                                    &axes,
                                                    ImGuiInputTextFlags_EnterReturnsTrue);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            axesChanged = true;
        }
        if (axesChanged) {
            config.onAxesChange(axes);
        }
    }

    if (impl.stacked) {
        const F32 stackedPull = style.ItemSpacing.y - stackedSpacingY;
        for (const auto& combo : impl.combos) {
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() - stackedPull);
            combo.render(ctx);
        }
    } else {
        F32 comboX = startX + shapeWidth + (config.onAxesChange ? axesWidth + spacing : 0.0f) + spacing;
        for (U64 i = 0; i < impl.combos.size(); ++i) {
            ImGui::SameLine();
            ImGui::SetCursorPosX(comboX);
            impl.combos[i].render(ctx);
            comboX += Scale(ctx, config.combos[i].width) + spacing;
        }
    }

    ImGui::PopStyleVar();
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
