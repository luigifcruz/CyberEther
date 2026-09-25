#include <jetstream/render/sakura/components/surface_view.hh>

#include "../helpers.hh"
#include "../../surface_input.hh"

#include <cmath>
#include <optional>

namespace Jetstream::Sakura {

namespace {

bool SameSurfaceResize(const SurfaceResize& lhs, const SurfaceResize& rhs) {
    return lhs.logicalSize.x == rhs.logicalSize.x &&
           lhs.logicalSize.y == rhs.logicalSize.y &&
           lhs.framebufferSize.x == rhs.framebufferSize.x &&
           lhs.framebufferSize.y == rhs.framebufferSize.y &&
           std::abs(lhs.scale - rhs.scale) <= 1e-6f;
}

Extent2D<F32> ResolveSurfaceLogicalDrawSize(const SurfaceView::Config& config,
                                            const Extent2D<F32>& available) {
    Extent2D<F32> size = config.size;
    if (size.x <= 0.0f) {
        size.x = available.x;
    }
    if (config.height.has_value()) {
        size.y = std::isfinite(*config.height) ? std::max(0.0f, *config.height) : 0.0f;
    } else if (size.y <= 0.0f) {
        size.y = available.y;
    }
    if (size.x <= 0.0f || size.y <= 0.0f) {
        return {0.0f, 0.0f};
    }

    return size;
}

}  // namespace

struct SurfaceView::Impl {
    Config config;
    std::optional<SurfaceResize> lastEmittedResize;
    int lastRenderedFrame = -1;
    detail::SurfaceInputState input;

    void resetInput() {
        if (config.onInput) input.reset(config.onInput);
    }

    ~Impl() { resetInput(); }
};

SurfaceView::SurfaceView() {
    this->impl = std::make_unique<Impl>();
}

SurfaceView::~SurfaceView() = default;
SurfaceView::SurfaceView(SurfaceView&&) noexcept = default;
SurfaceView& SurfaceView::operator=(SurfaceView&&) noexcept = default;

bool SurfaceView::update(Config config) {
    if (this->impl->config.id != config.id ||
        this->impl->config.textureSource != config.textureSource) {
        this->impl->lastEmittedResize.reset();
        this->impl->resetInput();
    }
    if (!config.onInput) this->impl->resetInput();
    this->impl->config = std::move(config);
    return true;
}

void SurfaceView::render(const Context& ctx) const {
    const auto& config = impl->config;
    const int frame = ImGui::GetFrameCount();
    if (impl->lastRenderedFrame >= 0 && frame > impl->lastRenderedFrame + 1) {
        impl->lastEmittedResize.reset();
    }
    impl->lastRenderedFrame = frame;

    bool detachClicked = false;
    const Extent2D<F32> available = Unscale(ctx, Private::ToExtent2D(ImGui::GetContentRegionAvail()));
    const Extent2D<F32> logicalDrawSize = ResolveSurfaceLogicalDrawSize(config, available);
    if (logicalDrawSize.x <= 0.0f || logicalDrawSize.y <= 0.0f) {
        impl->resetInput();
        return;
    }
    const auto resolvedResize = ResolveSurfaceResize(ctx, logicalDrawSize);
    if (resolvedResize.has_value() && config.onSize &&
        (!impl->lastEmittedResize.has_value() || !SameSurfaceResize(*impl->lastEmittedResize, *resolvedResize))) {
        impl->lastEmittedResize = *resolvedResize;
        config.onSize(*resolvedResize);
    }
    Extent2D<F32> displaySize = Scale(ctx, logicalDrawSize);
    if (resolvedResize.has_value()) {
        displaySize = FramebufferToDisplay(ctx, resolvedResize->framebufferSize);
    }

    const U64 texture = config.onResolveTexture
        ? config.onResolveTexture()
        : config.textureSource ? config.textureSource->raw() : config.texture;
    if (texture == 0) {
        impl->resetInput();
        return;
    }

    const ImTextureRef textureRef(static_cast<ImTextureID>(texture));
    if (textureRef.GetTexID() == ImTextureID_Invalid) {
        impl->resetInput();
        return;
    }

    const ImVec2 surfaceSize = Private::ToImVec2(displaySize);
    const ImVec2 cursorPos = Private::ToImVec2(
        SnapToFramebuffer(ctx, Private::ToExtent2D(ImGui::GetCursorScreenPos())));
    ImGui::SetCursorScreenPos(cursorPos);
    const ImVec2 cursorEnd(cursorPos.x + surfaceSize.x, cursorPos.y + surfaceSize.y);
    const F32 rounding = config.rounding <= 0.0f ? ImGui::GetStyle().FrameRounding : config.rounding;
    ImGui::GetWindowDrawList()->AddImageRounded(textureRef,
                                                cursorPos,
                                                cursorEnd,
                                                ImVec2(0.0f, 0.0f),
                                                ImVec2(1.0f, 1.0f),
                                                IM_COL32_WHITE,
                                                rounding);

    if (config.onInput) {
        ImGui::InvisibleButton(config.id.c_str(), surfaceSize,
                               ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
        detail::ForwardSurfaceInputEvents(cursorPos, surfaceSize, impl->input, config.onInput);
    } else {
        ImGui::Dummy(surfaceSize);
    }

    if (config.detachOverlay && ImGui::IsItemHovered()) {
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
            detachClicked = ImGui::IsMouseClicked(ImGuiMouseButton_Left);
        }

        ImDrawList* drawList = ImGui::GetWindowDrawList();
        drawList->AddRectFilled(buttonPos, buttonEnd, buttonColor, Scale(ctx, 4.0f));

        const char* icon = ICON_FA_UP_RIGHT_AND_DOWN_LEFT_FROM_CENTER;
        const ImVec2 textSize = ImGui::CalcTextSize(icon);
        const ImVec2 textPos(buttonPos.x + (buttonSize - textSize.x) * 0.5f,
                             buttonPos.y + (buttonSize - textSize.y) * 0.5f);
        drawList->AddText(textPos, IM_COL32(255, 255, 255, 255), icon);
    }

    if (detachClicked && config.onDetach) {
        config.onDetach();
    }
}

}  // namespace Jetstream::Sakura
