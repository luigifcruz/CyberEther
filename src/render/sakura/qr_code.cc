#include <jetstream/render/sakura/qr_code.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct QrCode::Impl {
    Config config;
};

QrCode::QrCode() {
    this->impl = std::make_unique<Impl>();
}

QrCode::~QrCode() = default;
QrCode::QrCode(QrCode&&) noexcept = default;
QrCode& QrCode::operator=(QrCode&&) noexcept = default;

bool QrCode::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void QrCode::render(const Context& ctx) const {
    const auto& config = this->impl->config;

    if (config.width <= 0 || config.data.empty()) {
        return;
    }

    ImGui::PushID(config.id.c_str());

    const int border = 2;
    const F32 moduleSize = Scale(ctx, config.moduleSize);
    const int totalWidth = config.width + border * 2;
    const F32 qrSize = totalWidth * moduleSize;
    Extent2D<F32> pos = Private::ToExtent2D(ImGui::GetCursorScreenPos());
    const F32 columnWidth = ImGui::GetContentRegionAvail().x;
    pos.x += std::max(0.0f, (columnWidth - qrSize) * 0.5f);

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(Private::ToImVec2(pos), Private::ToImVec2({pos.x + qrSize, pos.y + qrSize}), IM_COL32(255, 255, 255, 255));

    for (int y = 0; y < config.width; y++) {
        for (int x = 0; x < config.width; x++) {
            const U64 idx = static_cast<U64>(y * config.width + x);
            if (idx < config.data.size() && config.data[idx]) {
                const Extent2D<F32> p1 = {pos.x + (x + border) * moduleSize, pos.y + (y + border) * moduleSize};
                const Extent2D<F32> p2 = {p1.x + moduleSize, p1.y + moduleSize};
                drawList->AddRectFilled(Private::ToImVec2(p1), Private::ToImVec2(p2), IM_COL32(0, 0, 0, 255));
            }
        }
    }

    ImGui::Dummy(Private::ToImVec2({qrSize, qrSize}));
    if (ImGui::IsItemHovered()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    }
    if (ImGui::IsItemClicked() && config.onClick) {
        config.onClick();
    }

    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
