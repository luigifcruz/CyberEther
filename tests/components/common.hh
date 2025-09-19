#pragma once

#include <thread>
#include <memory>
#include <iostream>

#include "jetstream/base.hh"

using namespace Jetstream;

constexpr static Device ComputeDevice = Device::CPU;
constexpr static Device RenderDevice  = Device::Metal;
using ViewportPlatform = Viewport::GLFW<RenderDevice>;

class MagnifierGlass {
 public:
    void render(ImTextureRef textureRef, const ImVec2& imagePos, const ImVec2& imageSize, 
                const ImVec2& windowPos, const ImVec2& totalSize, float contentWidth) {
        if (!enabled) return;

        // Handle mouse interaction
        ImVec2 mousePos = ImGui::GetMousePos();
        ImVec2 imageTopLeft = ImVec2(windowPos.x + imagePos.x, windowPos.y + imagePos.y);
        ImVec2 relativeMousePos = ImVec2(mousePos.x - imageTopLeft.x, mousePos.y - imageTopLeft.y);

        bool mouseOverImage = (relativeMousePos.x >= 0 && relativeMousePos.x <= imageSize.x &&
                             relativeMousePos.y >= 0 && relativeMousePos.y <= imageSize.y);

        if (mouseOverImage && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            center = relativeMousePos;
            dragging = true;
        }

        if (dragging) {
            if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                center = relativeMousePos;
            } else {
                dragging = false;
            }
        }

        // Initialize center if not set
        if (center.x == 0.0f && center.y == 0.0f) {
            center = ImVec2(imageSize.x * 0.5f, imageSize.y * 0.5f);
        }

        // Draw magnifier glass
        ImDrawList* drawList = ImGui::GetWindowDrawList();

        // Calculate magnified region
        float sourceRadius = radius / zoom;
        ImVec2 sourceTopLeft = ImVec2(center.x - sourceRadius, center.y - sourceRadius);
        ImVec2 sourceBottomRight = ImVec2(center.x + sourceRadius, center.y + sourceRadius);

        // Clamp source region to image bounds
        sourceTopLeft.x = std::max(0.0f, sourceTopLeft.x);
        sourceTopLeft.y = std::max(0.0f, sourceTopLeft.y);
        sourceBottomRight.x = std::min(imageSize.x, sourceBottomRight.x);
        sourceBottomRight.y = std::min(imageSize.y, sourceBottomRight.y);

        // Calculate UV coordinates for the texture
        ImVec2 uv0 = ImVec2(sourceTopLeft.x / imageSize.x, sourceTopLeft.y / imageSize.y);
        ImVec2 uv1 = ImVec2(sourceBottomRight.x / imageSize.x, sourceBottomRight.y / imageSize.y);

        // Position magnifier glass
        ImVec2 magnifierPos = ImVec2(imageTopLeft.x + center.x + radius + 20,
                                   imageTopLeft.y + center.y - radius);

        // Keep magnifier on screen
        if (magnifierPos.x + radius * 2 > windowPos.x + contentWidth) {
            magnifierPos.x = imageTopLeft.x + center.x - radius * 2 - 20;
        }
        if (magnifierPos.y < windowPos.y) {
            magnifierPos.y = windowPos.y + 10;
        }
        if (magnifierPos.y + radius * 2 > windowPos.y + totalSize.y) {
            magnifierPos.y = windowPos.y + totalSize.y - radius * 2 - 10;
        }

        // Draw magnifier background circle
        drawList->AddCircleFilled(ImVec2(magnifierPos.x + radius, magnifierPos.y + radius),
                                radius + 3, IM_COL32(60, 60, 60, 255));

        // Draw magnified content
        drawList->AddImageQuad(
            textureRef,
            magnifierPos,
            ImVec2(magnifierPos.x + radius * 2, magnifierPos.y),
            ImVec2(magnifierPos.x + radius * 2, magnifierPos.y + radius * 2),
            ImVec2(magnifierPos.x, magnifierPos.y + radius * 2),
            uv0, ImVec2(uv1.x, uv0.y), uv1, ImVec2(uv0.x, uv1.y)
        );

        // Draw magnifier border
        drawList->AddCircle(ImVec2(magnifierPos.x + radius, magnifierPos.y + radius),
                          radius, IM_COL32(200, 200, 200, 255), 0, 3.0f);

        // Draw crosshair on original image
        ImVec2 crosshairCenter = ImVec2(imageTopLeft.x + center.x, imageTopLeft.y + center.y);
        drawList->AddLine(ImVec2(crosshairCenter.x - 10, crosshairCenter.y),
                        ImVec2(crosshairCenter.x + 10, crosshairCenter.y),
                        IM_COL32(255, 255, 255, 200), 2.0f);
        drawList->AddLine(ImVec2(crosshairCenter.x, crosshairCenter.y - 10),
                        ImVec2(crosshairCenter.x, crosshairCenter.y + 10),
                        IM_COL32(255, 255, 255, 200), 2.0f);

        // Draw source region indicator
        drawList->AddCircle(crosshairCenter, sourceRadius, IM_COL32(255, 255, 0, 150), 0, 2.0f);
    }

    void renderControls() {
        ImGui::Separator();
        ImGui::Text("Magnifier Glass:");
        ImGui::Checkbox("Enable Magnifier", &enabled);

        if (enabled) {
            ImGui::SliderFloat("Zoom Level", &zoom, 1.5f, 8.0f, "%.1fx");
            ImGui::SliderFloat("Magnifier Size", &radius, 50.0f, 200.0f, "%.0f px");

            if (ImGui::Button("Reset Position")) {
                center = ImVec2(0.0f, 0.0f);
            }

            ImGui::TextWrapped("Click and drag on the image to position the magnifier.");
        }
    }

 public:
    bool enabled = false;
    ImVec2 center = {0.0f, 0.0f};
    float radius = 100.0f;
    float zoom = 3.0f;
    bool dragging = false;
};

template<typename ComponentType>
class TestUIBase {
 public:
    TestUIBase(Instance& instance, const std::string& testName) 
        : instance(instance), testName(testName) {}

    virtual ~TestUIBase() = default;

    Result run() {
        JST_CHECK(setupComponent());
        JST_CHECK(setupFramebuffer());
        JST_CHECK(setupSurface());

        ImGui::GetStyle().Colors[ImGuiCol_WindowBg] = ImVec4(0.1, 0.1f, 0.1f, 1.0f);

        instance.start();

        graphicalWorker = std::thread([&]{
            closing = false;
            while (instance.presenting()) {
                this->graphicalThreadLoop();
            }
        });

        while (instance.running()) {
            instance.viewport().waitEvents();
        }

        closing = true;
        instance.reset();
        instance.stop();

        if (graphicalWorker.joinable()) {
            graphicalWorker.join();
        }

        instance.destroy();

        return Result::SUCCESS;
    }

 protected:
    virtual Result setupComponent() = 0;
    virtual Result setupFramebuffer() = 0;
    virtual Result setupSurface() = 0;
    virtual Result updateComponent() { return Result::SUCCESS; }
    virtual Result renderInfoPanel([[maybe_unused]] const ImVec2& totalSize, [[maybe_unused]] const ImVec2& contentSize) = 0;
    virtual std::string getComponentName() const = 0;

    Result drawSplitLayout() {
        static ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration |
                                        ImGuiWindowFlags_NoMove |
                                        ImGuiWindowFlags_NoSavedSettings;
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);

        ImGui::Begin(testName.c_str(), nullptr, flags);

        const auto& scale = ImGui::GetIO().DisplayFramebufferScale;
        const auto& totalSize = ImGui::GetContentRegionAvail();

        // Split layout: 75% content, 25% info
        const float contentWidth = totalSize.x * 0.75f;
        const float infoWidth = totalSize.x * 0.25f;

        // Left side - Content framebuffer
        ImGui::BeginChild("ContentFramebuffer", ImVec2(contentWidth, totalSize.y), true);

        const auto& contentSize = ImGui::GetContentRegionAvail();
        auto [width, height] = surface->size({
            static_cast<U64>(contentSize.x * scale.x),
            static_cast<U64>(contentSize.y * scale.y)
        });

        updatePixelSize(width, height);
        JST_CHECK_THROW(component->present());

        ImVec2 imagePos = ImGui::GetCursorPos();
        ImVec2 imageSize = ImVec2(static_cast<F32>(width / scale.x), static_cast<F32>(height / scale.y));

        ImGui::Image(ImTextureRef(framebufferTexture->raw()), imageSize);

        // Render magnifier glass if enabled
        ImVec2 windowPos = ImGui::GetWindowPos();
        magnifier.render(ImTextureRef(framebufferTexture->raw()), imagePos, imageSize, 
                        windowPos, totalSize, contentWidth);

        ImGui::EndChild();

        // Right side - Info panel
        ImGui::SameLine();
        ImGui::BeginChild("[CONTENT FRAMEBUFFER][INFO]", ImVec2(infoWidth, totalSize.y), true);

        ImGui::Text("Component: %s", getComponentName().c_str());
        ImGui::Separator();
        ImGui::Text("Performance:");
        ImGui::Text("  FPS: %.1f", ImGui::GetIO().Framerate);
        ImGui::Separator();
        ImGui::Text("Dimensions:");
        ImGui::Text("  Window: %.0fx%.0f", totalSize.x, totalSize.y);
        ImGui::Text("  Content: %.0fx%.0f", contentSize.x, contentSize.y);
        ImGui::Text("  Buffer: %llux%llu", surface->size().x, surface->size().y);

        JST_CHECK_THROW(renderInfoPanel(totalSize, contentSize));

        magnifier.renderControls();

        ImGui::EndChild();
        ImGui::End();

        return Result::SUCCESS;
    }

    virtual void updatePixelSize(U64 width, U64 height) = 0;

    void graphicalThreadLoop() {
        if (instance.begin() == Result::SKIP) {
            return;
        }

        if (!closing) {
            JST_CHECK_THROW(updateComponent());
            JST_CHECK_THROW(drawSplitLayout());
        }

        JST_CHECK_THROW(instance.present());
        if (instance.end() == Result::SKIP) {
            return;
        }
    }

 protected:
    Instance& instance;
    std::string testName;
    std::thread graphicalWorker;
    bool closing = false;

    std::shared_ptr<Render::Texture> framebufferTexture;
    std::shared_ptr<Render::Surface> surface;
    std::shared_ptr<ComponentType> component;

    MagnifierGlass magnifier;
};

inline int runComponentTest(const std::string& appName, const std::string& windowTitle,
                           const Extent2D<U64>& windowSize, 
                           std::function<void(Instance&)> testFunction) {
    std::cout << appName << " - CyberEther" << std::endl;

    if (Backend::Initialize<ComputeDevice>({}) != Result::SUCCESS) {
        JST_FATAL("Cannot initialize compute backend.");
        return 1;
    }

    if (Backend::Initialize<RenderDevice>({}) != Result::SUCCESS) {
        JST_FATAL("Cannot initialize render backend.");
        return 1;
    }

    Instance instance;

    Viewport::Config viewportCfg;
    viewportCfg.vsync = true;
    viewportCfg.size = windowSize;
    viewportCfg.title = windowTitle;
    JST_CHECK_THROW(instance.buildViewport<ViewportPlatform>(viewportCfg));

    Render::Window::Config renderCfg;
    renderCfg.scale = 1.0f;
    JST_CHECK_THROW(instance.buildRender<RenderDevice>(renderCfg));

    testFunction(instance);

    Backend::DestroyAll();

    std::cout << "Goodbye from " << appName << "!" << std::endl;

    return 0;
}