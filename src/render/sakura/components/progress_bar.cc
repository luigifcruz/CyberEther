#include <jetstream/render/sakura/components/progress_bar.hh>

#include "../helpers.hh"

#include <algorithm>
#include <limits>

namespace Jetstream::Sakura {

namespace {

constexpr F64 kColorFadeSeconds = 0.25;

}  // namespace

struct ProgressBar::Impl {
    Config config;
    std::string fadeKey;
    ImVec4 fadeFrom;
    F64 fadeStart = -std::numeric_limits<F64>::infinity();

    ImVec4 blend(const ImVec4& target, const F64 now) const {
        const F32 t = static_cast<F32>(std::clamp((now - fadeStart) / kColorFadeSeconds, 0.0, 1.0));
        if (t >= 1.0f) {
            return target;
        }
        const F32 eased = t * t * (3.0f - 2.0f * t);
        return {
            fadeFrom.x + (target.x - fadeFrom.x) * eased,
            fadeFrom.y + (target.y - fadeFrom.y) * eased,
            fadeFrom.z + (target.z - fadeFrom.z) * eased,
            fadeFrom.w + (target.w - fadeFrom.w) * eased,
        };
    }

    ImVec4 fillColor(const Context& ctx) {
        const F64 now = ImGui::GetTime();
        if (fadeKey != config.colorKey) {
            if (!fadeKey.empty()) {
                fadeFrom = blend(Private::ImColor(ctx, fadeKey), now);
                fadeStart = now;
            }
            fadeKey = config.colorKey;
        }
        return blend(Private::ImColor(ctx, config.colorKey), now);
    }
};

ProgressBar::ProgressBar() {
    this->impl = std::make_unique<Impl>();
}

ProgressBar::~ProgressBar() = default;
ProgressBar::ProgressBar(ProgressBar&&) noexcept = default;
ProgressBar& ProgressBar::operator=(ProgressBar&&) noexcept = default;

bool ProgressBar::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void ProgressBar::render(const Context& ctx) const {
    const auto& config = this->impl->config;

    ImGui::PushID(config.id.c_str());
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, this->impl->fillColor(ctx));
    if (!config.backgroundColorKey.empty()) {
        ImGui::PushStyleColor(ImGuiCol_FrameBg, Private::ImColor(ctx, config.backgroundColorKey));
    }
    ImGui::ProgressBar(config.value, Private::ToImVec2(Scale(ctx, config.size)), config.overlay.c_str());
    if (!config.backgroundColorKey.empty()) {
        ImGui::PopStyleColor();
    }
    ImGui::PopStyleColor();
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
