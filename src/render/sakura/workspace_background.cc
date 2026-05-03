#include <jetstream/render/sakura/workspace_background.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct WorkspaceBackground::Impl {
    struct Particle {
        Extent2D<F32> position;
        F32 velocity = 0.0f;
        F32 radius = 0.0f;
        F32 alpha = 0.0f;
        F32 phase = 0.0f;
    };

    Config config;
    std::vector<Particle> particles;
    Extent2D<F32> previousAreaPos = {0.0f, 0.0f};
    Extent2D<F32> previousAreaSize = {0.0f, 0.0f};
    std::mt19937 randomGenerator{std::random_device{}()};
};

WorkspaceBackground::WorkspaceBackground() {
    this->impl = std::make_unique<Impl>();
}

WorkspaceBackground::~WorkspaceBackground() = default;
WorkspaceBackground::WorkspaceBackground(WorkspaceBackground&&) noexcept = default;
WorkspaceBackground& WorkspaceBackground::operator=(WorkspaceBackground&&) noexcept = default;

bool WorkspaceBackground::update(Config config) {
    if (this->impl->config.particleCount != config.particleCount) {
        this->impl->particles.clear();
    }
    this->impl->config = std::move(config);
    return true;
}

void WorkspaceBackground::render(const Context& ctx) {
    auto& impl = *this->impl;
    const auto& config = impl.config;

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImDrawList* drawList = ImGui::GetBackgroundDrawList();
    const ImVec2 rectMin(viewport->Pos.x, viewport->Pos.y);
    const ImVec2 rectMax(viewport->Pos.x + viewport->Size.x, viewport->Pos.y + viewport->Size.y);
    drawList->AddRectFilled(rectMin,
                            rectMax,
                            ImGui::ColorConvertFloat4ToU32(Private::ImColor(ctx, config.backgroundColorKey)));

    if (!config.particles || config.particleCount == 0) {
        return;
    }

    const F32 topOffset = Scale(ctx, config.topOffset);
    const Extent2D<F32> areaPos{viewport->Pos.x, viewport->Pos.y + topOffset};
    const Extent2D<F32> areaSize{viewport->Size.x, std::max(0.0f, viewport->Size.y - topOffset)};
    if (areaSize.x <= 0.0f || areaSize.y <= 0.0f) {
        return;
    }

    const bool areaChanged = impl.previousAreaPos != areaPos || impl.previousAreaSize != areaSize;
    const std::size_t particleCount = static_cast<std::size_t>(config.particleCount);
    if (impl.particles.size() != particleCount || areaChanged) {
        impl.particles.clear();
        impl.particles.reserve(particleCount);
        impl.previousAreaPos = areaPos;
        impl.previousAreaSize = areaSize;

        std::uniform_real_distribution<F32> unit(0.0f, 1.0f);
        for (std::size_t i = 0; i < particleCount; ++i) {
            impl.particles.push_back({
                .position = {
                    areaPos.x + unit(impl.randomGenerator) * areaSize.x,
                    areaPos.y + unit(impl.randomGenerator) * areaSize.y,
                },
                .velocity = 10.0f + unit(impl.randomGenerator) * 30.0f,
                .radius = 1.0f + unit(impl.randomGenerator) * 2.5f,
                .alpha = 0.15f + unit(impl.randomGenerator) * 0.25f,
                .phase = unit(impl.randomGenerator) * 6.28318f,
            });
        }
    }

    const ImVec4 baseColor = Private::ImColor(ctx, config.particleColorKey, ImVec4(0.2f, 0.7f, 1.0f, 1.0f));
    const F32 time = static_cast<F32>(ImGui::GetTime());
    const F32 deltaTime = ImGui::GetIO().DeltaTime;
    std::uniform_real_distribution<F32> unit(0.0f, 1.0f);

    ImGui::PushID(config.id.c_str());
    for (auto& particle : impl.particles) {
        particle.position.x += particle.velocity * 0.15f * deltaTime * ScalingFactor(ctx);
        particle.position.y -= particle.velocity * deltaTime * ScalingFactor(ctx);

        const F32 waveOffset = std::sin(time * 0.5f + particle.phase) * Scale(ctx, 2.0f);
        const Extent2D<F32> renderPos{particle.position.x + waveOffset, particle.position.y};

        if (renderPos.y < areaPos.y - Scale(ctx, 10.0f)) {
            particle.position.y = areaPos.y + areaSize.y + Scale(ctx, 10.0f);
            particle.position.x = areaPos.x + unit(impl.randomGenerator) * areaSize.x;
        }
        if (renderPos.x > areaPos.x + areaSize.x + Scale(ctx, 10.0f)) {
            particle.position.x = areaPos.x - Scale(ctx, 10.0f);
        }

        const F32 twinkle = std::sin(time * 2.0f + particle.phase) * 0.1f + 0.9f;
        const F32 finalAlpha = particle.alpha * twinkle;
        const ImU32 color = ImGui::ColorConvertFloat4ToU32(ImVec4(baseColor.x,
                                                                  baseColor.y,
                                                                  baseColor.z,
                                                                  finalAlpha));
        const ImU32 glowColor = ImGui::ColorConvertFloat4ToU32(ImVec4(baseColor.x,
                                                                      baseColor.y,
                                                                      baseColor.z,
                                                                      finalAlpha * 0.3f));

        drawList->AddCircleFilled(Private::ToImVec2(renderPos), particle.radius * Scale(ctx, 2.5f), glowColor, 12);
        drawList->AddCircleFilled(Private::ToImVec2(renderPos), particle.radius * ScalingFactor(ctx), color, 12);
    }
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
