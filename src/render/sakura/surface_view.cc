#include <jetstream/render/sakura/surface_view.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct SurfaceView::Impl {
    Config config;
};

SurfaceView::SurfaceView() {
    this->impl = std::make_unique<Impl>();
}

SurfaceView::~SurfaceView() = default;
SurfaceView::SurfaceView(SurfaceView&&) noexcept = default;
SurfaceView& SurfaceView::operator=(SurfaceView&&) noexcept = default;

bool SurfaceView::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

SurfaceView::Result SurfaceView::render(const Context& ctx) const {
    const auto& config = this->impl->config;

    Result result;
    Extent2D<F32> displaySize = Scale(ctx, config.size);
    const Extent2D<F32> available = Private::ToExtent2D(ImGui::GetContentRegionAvail());
    if (displaySize.x <= 0.0f) {
        displaySize.x = available.x;
    }
    if (displaySize.y <= 0.0f) {
        displaySize.y = available.y;
    }

    const Extent2D<F32> availableLogicalSize = Unscale(ctx, displaySize);
    Extent2D<F32> size = availableLogicalSize;
    if (config.aspectRatioSize.has_value() && config.aspectRatioSize->x > 0.0f && config.aspectRatioSize->y > 0.0f) {
        const F32 aspectHeight = size.x * config.aspectRatioSize->y / config.aspectRatioSize->x;
        if (aspectHeight > 0.0f && aspectHeight < size.y) {
            size.y = aspectHeight;
        }
    }
    if (size.x <= 0.0f || size.y <= 0.0f) {
        return result;
    }
    const auto resolvedResize = ResolveSurfaceResize(ctx, size);
    if (config.onSize && resolvedResize.has_value()) {
        config.onSize({
            .availableLogicalSize = availableLogicalSize,
            .resolvedLogicalSize = size,
            .logicalSize = resolvedResize->logicalSize,
            .framebufferSize = resolvedResize->framebufferSize,
            .scale = resolvedResize->scale,
        });
    }
    displaySize = Scale(ctx, size);

    const U64 texture = config.onResolveTexture ? config.onResolveTexture() : config.texture;
    if (texture == 0) {
        return result;
    }

    const ImTextureRef textureRef(static_cast<ImTextureID>(texture));
    if (textureRef.GetTexID() == ImTextureID_Invalid) {
        return result;
    }

    const ImVec2 surfaceSize = Private::ToImVec2(displaySize);
    const ImVec2 cursorPos = ImGui::GetCursorScreenPos();
    const ImVec2 cursorEnd(cursorPos.x + surfaceSize.x, cursorPos.y + surfaceSize.y);
    const F32 rounding = config.rounding <= 0.0f ? ImGui::GetStyle().FrameRounding : config.rounding;
    ImGui::GetWindowDrawList()->AddImageRounded(textureRef,
                                                cursorPos,
                                                cursorEnd,
                                                ImVec2(0.0f, 0.0f),
                                                ImVec2(1.0f, 1.0f),
                                                IM_COL32_WHITE,
                                                rounding);

    ImGui::InvisibleButton(config.id.c_str(), surfaceSize);
    result.hovered = ImGui::IsItemHovered();
    if (result.hovered) {
        const ImVec2 mousePos = ImGui::GetMousePos();
        result.normalizedMouse = {(mousePos.x - cursorPos.x) / surfaceSize.x,
                                  (mousePos.y - cursorPos.y) / surfaceSize.y};
        result.leftClicked = ImGui::IsMouseClicked(ImGuiMouseButton_Left);
        result.rightClicked = ImGui::IsMouseClicked(ImGuiMouseButton_Right);
        result.leftReleased = ImGui::IsMouseReleased(ImGuiMouseButton_Left);
        result.rightReleased = ImGui::IsMouseReleased(ImGuiMouseButton_Right);

        const ImGuiIO& io = ImGui::GetIO();
        result.scrolled = io.MouseWheel != 0.0f || io.MouseWheelH != 0.0f;
        result.scroll = {io.MouseWheelH, io.MouseWheel};
    }

    if (config.detachOverlay && result.hovered) {
        const F32 buttonSize = Scale(ctx, 24.0f);
        const F32 buttonPadding = Scale(ctx, 8.0f);
        const ImVec2 buttonPos(cursorEnd.x - buttonSize - buttonPadding, cursorPos.y + buttonPadding);
        const ImVec2 buttonEnd(buttonPos.x + buttonSize, buttonPos.y + buttonSize);
        const ImVec2 mousePos = ImGui::GetMousePos();
        const bool buttonHovered = mousePos.x >= buttonPos.x && mousePos.x <= buttonEnd.x &&
                                   mousePos.y >= buttonPos.y && mousePos.y <= buttonEnd.y;

        ImU32 buttonColor = IM_COL32(30, 30, 30, 200);
        if (buttonHovered) {
            buttonColor = IM_COL32(60, 60, 60, 230);
            result.detachClicked = ImGui::IsMouseClicked(ImGuiMouseButton_Left);
        }

        ImDrawList* drawList = ImGui::GetWindowDrawList();
        drawList->AddRectFilled(buttonPos, buttonEnd, buttonColor, Scale(ctx, 4.0f));

        const char* icon = ICON_FA_UP_RIGHT_AND_DOWN_LEFT_FROM_CENTER;
        const ImVec2 textSize = ImGui::CalcTextSize(icon);
        const ImVec2 textPos(buttonPos.x + (buttonSize - textSize.x) * 0.5f,
                             buttonPos.y + (buttonSize - textSize.y) * 0.5f);
        drawList->AddText(textPos, IM_COL32(255, 255, 255, 255), icon);
    }

    if (result.hovered && config.onMouse) {
        MouseEvent event{};
        event.position = result.normalizedMouse;
        event.scroll = {0.0f, 0.0f};

        if (result.leftClicked) {
            event.type = MouseEventType::Click;
            event.button = MouseButton::Left;
            config.onMouse(event);
        } else if (result.rightClicked) {
            event.type = MouseEventType::Click;
            event.button = MouseButton::Right;
            config.onMouse(event);
        } else if (result.leftReleased) {
            event.type = MouseEventType::Release;
            event.button = MouseButton::Left;
            config.onMouse(event);
        } else if (result.rightReleased) {
            event.type = MouseEventType::Release;
            event.button = MouseButton::Right;
            config.onMouse(event);
        }

        if (result.scrolled) {
            event.type = MouseEventType::Scroll;
            event.scroll = result.scroll;
            config.onMouse(event);
        }

        event.type = MouseEventType::Move;
        config.onMouse(event);
    }

    if (result.detachClicked && config.onDetach) {
        config.onDetach();
    }

    return result;
}

}  // namespace Jetstream::Sakura
