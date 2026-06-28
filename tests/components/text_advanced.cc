#include "common.hh"
#include "jetstream/render/components/text.hh"
#include <chrono>

#include <jetstream/fmt/format.h>

class TextTestUI : public TestUIBase<Render::Components::Text> {
 public:
    TextTestUI(const std::shared_ptr<Instance>& instance)
        : TestUIBase(instance, "Surface Test") {
        startTime = std::chrono::steady_clock::now();
        frameCounter = 0;
    }

 protected:
    Result setupComponent() override {
        if (!render->hasFont("default_mono")) {
            JST_ERROR("Font 'default_mono' not found.");
            return Result::ERROR;
        }

        Render::Components::Text::Config cfg;
        cfg.maxCharacters = 512;
        cfg.color = {0.9f, 0.9f, 0.9f, 1.0f}; // Light gray for general text
        cfg.font = render->font("default_mono");
        cfg.elements = {
            {"title", {.scale = 3.0f, .position = {0.0f, 0.8f}, .alignment = {1, 1}, .fill = "Text Component Demo"}},
            {"large", {.scale = 2.0f, .position = {-0.8f, 0.5f}, .fill = "Large Text (2.0x)"}},
            {"medium", {.scale = 1.5f, .position = {-0.8f, 0.3f}, .fill = "Medium Text (1.5x)"}},
            {"small", {.position = {-0.8f, 0.1f}, .fill = "Small Text (1.0x)"}},
            {"tiny", {.scale = 0.8f, .position = {-0.8f, -0.1f}, .fill = "Tiny Text (0.8x)"}},
            {"color_info", {.scale = 1.2f, .position = {0.0f, 0.4f}, .alignment = {1, 1}, .fill = "Global Color Demo"}},
            {"color_note", {.scale = 0.9f, .position = {0.0f, 0.25f}, .alignment = {1, 1}, .fill = "All text uses the same color"}},
            {"fps", {.scale = 1.2f, .position = {0.7f, 0.7f}, .alignment = {2, 1}, .fill = "FPS: 0.0"}},
            {"time", {.position = {0.7f, 0.55f}, .alignment = {2, 1}, .fill = "Time: 0.00s"}},
            {"counter", {.position = {0.7f, 0.4f}, .alignment = {2, 1}, .fill = "Count: 0"}},
            {"top_left", {.position = {-0.9f, 0.9f}, .alignment = {0, 2}, .fill = "Top Left (0,2)"}},
            {"top_center", {.position = {0.0f, 0.9f}, .alignment = {1, 2}, .fill = "Top Center (1,2)"}},
            {"top_right", {.position = {0.9f, 0.9f}, .alignment = {2, 2}, .fill = "Top Right (2,2)"}},
            {"bottom_left", {.position = {-0.9f, -0.9f}, .fill = "Bottom Left (0,0)"}},
            {"bottom_center", {.position = {0.0f, -0.9f}, .alignment = {1, 0}, .fill = "Bottom Center (1,0)"}},
            {"bottom_right", {.position = {0.9f, -0.9f}, .alignment = {2, 0}, .fill = "Bottom Right (2,0)"}},
            {"bouncing", {.scale = 1.5f, .position = {0.0f, -0.5f}, .alignment = {1, 1}, .fill = "Bouncing Text!"}},
            {"info", {.scale = 0.9f, .position = {-0.8f, -0.3f}, .fill = "Interactive text demo"}},
            {"controls", {.scale = 0.8f, .position = {-0.8f, -0.4f}, .fill = "Dynamic animations"}},
        };
        JST_CHECK(render->build(component, cfg));
        return render->bind(component);
    }

    Result setupFramebuffer() override {
        return render->build<Render::Texture>(framebufferTexture, Render::Texture::Config{
            .size = {800, 600}
        });
    }

    Result setupSurface() override {
        Render::Surface::Config cfg;
        cfg.framebuffer = framebufferTexture;
        cfg.clearColor = {0.1f, 0.1f, 0.2f, 1.0f};
        cfg.multisampled = false;

        JST_CHECK(component->surface(cfg));
        JST_CHECK(render->build(surface, cfg));
        return render->bind(surface);
    }

    void updatePixelSize(U64 width, U64 height) override {
        component->updatePixelSize({2.0f / static_cast<F32>(width), 2.0f / static_cast<F32>(height)});
    }

    std::string getComponentName() const override {
        return "Text Demo";
    }

    Result updateComponent() override {
        frameCounter++;
        return updateDynamicContent();
    }

    Result renderInfoPanel([[maybe_unused]] const ImVec2& totalSize, [[maybe_unused]] const ImVec2& contentSize) override {
        ImGui::Text("  Frames: %llu", frameCounter);
        ImGui::Separator();
        ImGui::Text("Elements: 17 text objects");
        ImGui::Text("Features:");
        ImGui::BulletText("Multiple sizes/scales");
        ImGui::BulletText("9-point alignment");
        ImGui::BulletText("Dynamic updates");
        ImGui::BulletText("Real-time animations");
        ImGui::Separator();
        ImGui::Checkbox("Enable Animation", &enableAnimation);
        if (enableAnimation) {
            ImGui::SliderFloat("Speed", &animationSpeed, 0.1f, 5.0f, "%.1f");
        }
        ImGui::SliderFloat("Text Scale", &textScale, 0.5f, 3.0f, "%.1f");
        if (ImGui::Button("Reset")) {
            startTime = std::chrono::steady_clock::now();
            frameCounter = 0;
        }
        return Result::SUCCESS;
    }

 private:
    std::chrono::steady_clock::time_point startTime;
    U64 frameCounter = 0;
    bool enableAnimation = true;
    F32 textScale = 1.0f;
    F32 animationSpeed = 1.0f;
    F32 colorR = 0.9f;
    F32 colorG = 0.9f;
    F32 colorB = 0.9f;

    Result updateDynamicContent() {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<F32>(now - startTime).count();

        if (enableAnimation) {
            if (frameCounter % 30 == 0) {
                F32 fps = ImGui::GetIO().Framerate;
                auto fpsConfig = component->get("fps");
                fpsConfig.fill = jst::fmt::format("FPS: {:.1f}", fps);
                JST_CHECK(component->update("fps", fpsConfig));
            }

            auto timeConfig = component->get("time");
            timeConfig.fill = jst::fmt::format("Time: {:.2f}s", elapsed);
            JST_CHECK(component->update("time", timeConfig));

            auto counterConfig = component->get("counter");
            counterConfig.fill = jst::fmt::format("Count: {}", frameCounter);
            JST_CHECK(component->update("counter", counterConfig));

            F32 bounce = std::sin(elapsed * animationSpeed * 3.0f) * 0.2f;
            auto bouncingConfig = component->get("bouncing");
            bouncingConfig.position = {0.0f, -0.5f + bounce};
            JST_CHECK(component->update("bouncing", bouncingConfig));

            F32 titleScale = 3.0f + std::sin(elapsed * animationSpeed * 0.5f) * 0.3f;
            auto titleConfig = component->get("title");
            titleConfig.scale = titleScale * textScale;
            JST_CHECK(component->update("title", titleConfig));
        }

        auto currentColor = component->getConfig().color;
        if (currentColor.r != colorR || currentColor.g != colorG || currentColor.b != colorB) {
        }

        return Result::SUCCESS;
    }
};

int main() {
    return runComponentTest("Text Component Demo", "Text Component Demo", {1400, 900}, [](const std::shared_ptr<Instance>& instance) {
        TextTestUI(instance).run();
    });
}
