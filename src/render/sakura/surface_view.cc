#include "surface_view.hh"

#include "helpers.hh"
#include "surface_presenter.hh"

#include "jetstream/render/tools/imgui_blend_ext.h"

#include <cmath>
#include <optional>

namespace Jetstream::Sakura {

namespace {

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

void PremultiplyVertexColors(ImDrawList* drawList, const int vtxStart) {
    for (int i = vtxStart; i < drawList->VtxBuffer.Size; ++i) {
        ImU32& col = drawList->VtxBuffer[i].col;
        const ImU32 alpha = (col >> IM_COL32_A_SHIFT) & 0xFF;
        const auto scale = [&](const int shift) {
            return ((((col >> shift) & 0xFF) * alpha + 127) / 255) << shift;
        };
        col = scale(IM_COL32_R_SHIFT) | scale(IM_COL32_G_SHIFT) | scale(IM_COL32_B_SHIFT) |
              (alpha << IM_COL32_A_SHIFT);
    }
}

}  // namespace

struct SurfaceView::Impl {
    Config config;
    Private::SurfacePresenter presenter;
};

SurfaceView::SurfaceView() {
    this->impl = std::make_unique<Impl>();
}

SurfaceView::~SurfaceView() = default;
SurfaceView::SurfaceView(SurfaceView&&) noexcept = default;
SurfaceView& SurfaceView::operator=(SurfaceView&&) noexcept = default;

void SurfaceView::update(Config config) {
    if (this->impl->config.id != config.id ||
        this->impl->config.textureSource != config.textureSource) {
        this->impl->presenter.resetResize();
    }
    if (this->impl->config.id != config.id) {
        this->impl->presenter.resetSettle();
    }
    this->impl->config = std::move(config);
}

void SurfaceView::render(const Context& ctx) const {
    const auto& config = impl->config;
    const int frame = ImGui::GetFrameCount();
    impl->presenter.beginFrame(frame);

    const Extent2D<F32> available = Unscale(ctx, Private::ToExtent2D(ImGui::GetContentRegionAvail()));
    const Extent2D<F32> logicalDrawSize = ResolveSurfaceLogicalDrawSize(config, available);
    if (logicalDrawSize.x <= 0.0f || logicalDrawSize.y <= 0.0f) {
        return;
    }
    const auto resolvedResize = ResolveSurfaceResize(ctx, logicalDrawSize);
    impl->presenter.emitResize(resolvedResize, config.onSize);
    Extent2D<F32> displaySize = Scale(ctx, logicalDrawSize);
    if (resolvedResize.has_value()) {
        displaySize = FramebufferToDisplay(ctx, resolvedResize->framebufferSize);
    }

    const U64 texture = config.resolveTexture
        ? config.resolveTexture()
        : config.textureSource ? config.textureSource->raw() : config.texture;
    if (texture == 0) {
        return;
    }

    const ImTextureRef textureRef(static_cast<ImTextureID>(texture));
    if (textureRef.GetTexID() == ImTextureID_Invalid) {
        return;
    }

    const ImVec2 surfaceSize = Private::ToImVec2(displaySize);
    const ImVec2 cursorPos = Private::ToImVec2(
        SnapToFramebuffer(ctx, Private::ToExtent2D(ImGui::GetCursorScreenPos())));
    ImGui::SetCursorScreenPos(cursorPos);
    const ImVec2 cursorEnd(cursorPos.x + surfaceSize.x, cursorPos.y + surfaceSize.y);
    auto* drawList = ImGui::GetWindowDrawList();
    ImRect imageRect(cursorPos, cursorEnd);
    imageRect.Expand(drawList->_FringeScale);
    const bool visible = imageRect.Overlaps(ImRect(drawList->GetClipRectMin(), drawList->GetClipRectMax()));
    if (config.textureSource && config.textureSource->raw() == texture) {
        config.textureSource->presentationHint(frame, visible);
    }
    const bool settled = impl->presenter.settle(config.textureSource.get(), resolvedResize, frame);
    if (visible && settled) {
        const bool premultiplied = config.premultiplied && ImGui::PremultipliedAlphaCallback;
        if (premultiplied) {
            drawList->AddCallback(ImGui::PremultipliedAlphaCallback, nullptr);
        }
        const int vtxStart = drawList->VtxBuffer.Size;
        drawList->AddImageRounded(textureRef,
                                  cursorPos,
                                  cursorEnd,
                                  ImVec2(0.0f, 0.0f),
                                  ImVec2(1.0f, 1.0f),
                                  IM_COL32_WHITE,
                                  std::max(0.0f, config.rounding));
        if (premultiplied) {
            PremultiplyVertexColors(drawList, vtxStart);
            drawList->AddCallback(ImDrawCallback_ResetRenderState, nullptr);
        }
    }

    ImGui::Dummy(surfaceSize);
}

}  // namespace Jetstream::Sakura
