#include <jetstream/render/sakura/components/node/uint_input.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct NodeUIntInput::Impl {
    Config config;
};

NodeUIntInput::NodeUIntInput() {
    this->impl = std::make_unique<Impl>();
}

NodeUIntInput::~NodeUIntInput() = default;
NodeUIntInput::NodeUIntInput(NodeUIntInput&&) noexcept = default;
NodeUIntInput& NodeUIntInput::operator=(NodeUIntInput&&) noexcept = default;

bool NodeUIntInput::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void NodeUIntInput::render(const Context& ctx) const {
    const auto& config = this->impl->config;

    U64 value = config.value;
    ImGui::PushID(config.id.c_str());
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(Scale(ctx, 6.0f), Scale(ctx, 3.0f)));
    ImGui::SetNextItemWidth(-FLT_MIN);
    const bool changed = ImGui::InputScalar("##uint",
                                            ImGuiDataType_U64,
                                            &value,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            ImGuiInputTextFlags_EnterReturnsTrue);
    if (!config.unit.empty()) {
        const ImVec2 inputPos = ImGui::GetItemRectMin();
        const ImVec2 inputSize = ImGui::GetItemRectSize();
        const ImVec2 textSize = ImGui::CalcTextSize(config.unit.c_str());
        const ImVec2 textPos(inputPos.x + inputSize.x - textSize.x - Scale(ctx, 3.0f),
                             inputPos.y + (inputSize.y - textSize.y) * 0.5f);
        ImGui::GetWindowDrawList()->AddText(textPos,
                                            ImGui::ColorConvertFloat4ToU32(Private::ImColor(ctx, "text_secondary")),
                                            config.unit.c_str());
    }
    if (changed && config.onChange) {
        config.onChange(value);
    }
    ImGui::PopStyleVar();
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
