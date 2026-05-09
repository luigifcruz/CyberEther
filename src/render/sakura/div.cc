#include <jetstream/render/sakura/div.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct Div::Impl {
    Config config;
};

Div::Div() {
    this->impl = std::make_unique<Impl>();
}

Div::~Div() = default;
Div::Div(Div&&) noexcept = default;
Div& Div::operator=(Div&&) noexcept = default;

bool Div::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void Div::render(const Context& ctx, Child child) const {
    const auto& config = this->impl->config;

    ImGui::PushID(config.id.c_str());

    ImGuiChildFlags childFlags = ImGuiChildFlags_None;
    if (config.border) {
        childFlags |= ImGuiChildFlags_Borders;
    }

    ImGuiWindowFlags windowFlags = ImGuiWindowFlags_None;
    if (!config.scrollbar) {
        windowFlags |= ImGuiWindowFlags_NoScrollbar;
    }
    if (!config.mouseScroll) {
        windowFlags |= ImGuiWindowFlags_NoScrollWithMouse;
    }
    if (!config.inputs) {
        windowFlags |= ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNav;
    }

    if (config.padding > 0.0f || config.rounding > 0.0f || config.onClick) {
        const F32 padding = Scale(ctx, config.padding);
        Extent2D<F32> size = Scale(ctx, config.size);
        if (size.x == 0.0f) {
            size.x = ImGui::GetContentRegionAvail().x;
        } else if (size.x < 0.0f) {
            size.x = std::max(0.0f, ImGui::GetContentRegionAvail().x + size.x);
        }
        if (size.y < 0.0f) {
            size.y = std::max(0.0f, ImGui::GetContentRegionAvail().y + size.y);
        }

        const Extent2D<F32> divStart = Private::ToExtent2D(ImGui::GetCursorScreenPos());
        const F32 contentWidth = std::max(0.0f, size.x - 2.0f * padding);

        ImDrawList* drawList = ImGui::GetWindowDrawList();
        drawList->ChannelsSplit(2);
        drawList->ChannelsSetCurrent(1);

        const std::string childId = config.id + "Content";
        ImGuiChildFlags contentChildFlags = ImGuiChildFlags_None;
        if (size.y == 0.0f) {
            contentChildFlags |= ImGuiChildFlags_AutoResizeY;
        }
        ImGui::SetCursorPos(Private::ToImVec2({ImGui::GetCursorPosX() + padding, ImGui::GetCursorPosY() + padding}));
        if (ImGui::BeginChild(childId.c_str(),
                              Private::ToImVec2({contentWidth, size.y == 0.0f ? 0.0f : std::max(0.0f, size.y - 2.0f * padding)}),
                              contentChildFlags,
                              windowFlags)) {
            if (child) {
                child(ctx);
            }
        }
        ImGui::EndChild();

        const Extent2D<F32> groupMax = Private::ToExtent2D(ImGui::GetItemRectMax());
        const Extent2D<F32> divMin = divStart;
        const Extent2D<F32> divMax = {divStart.x + size.x, size.y == 0.0f ? groupMax.y + padding : divStart.y + size.y};
        const bool hovered = config.inputs && (config.onClick || config.onDoubleClick) &&
                             ImGui::IsMouseHoveringRect(Private::ToImVec2(divMin), Private::ToImVec2(divMax));

        if (hovered) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                if (config.onClick) {
                    config.onClick();
                }
            }
            if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                if (config.onDoubleClick) {
                    config.onDoubleClick();
                }
            }
        }

        const std::string& backgroundKey = config.selected ? config.selectedColorKey
                                                           : (hovered ? config.hoveredColorKey : config.colorKey);
        drawList->ChannelsSetCurrent(0);
        drawList->AddRectFilled(Private::ToImVec2(divMin),
                                Private::ToImVec2(divMax),
                                ImGui::ColorConvertFloat4ToU32(Private::ImColor(ctx, backgroundKey)),
                                Scale(ctx, config.rounding));
        if (config.border) {
            drawList->AddRect(Private::ToImVec2(divMin),
                              Private::ToImVec2(divMax),
                              ImGui::ColorConvertFloat4ToU32(Private::ImColor(ctx, config.borderColorKey)),
                              Scale(ctx, config.rounding));
        }
        drawList->ChannelsMerge();

        ImGui::Dummy(Private::ToImVec2({0.0f, std::max(0.0f, divMax.y - ImGui::GetCursorScreenPos().y)}));
        ImGui::PopID();
        return;
    }

    if (ImGui::BeginChild(config.id.c_str(), Private::ToImVec2(Scale(ctx, config.size)), childFlags, windowFlags)) {
        if (child) {
            child(ctx);
        }
    }
    ImGui::EndChild();
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
