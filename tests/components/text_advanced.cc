#include "common.hh"
#include "jetstream/render/components/text.hh"
#include <chrono>
#include <format>

class TextTestUI : public TestUIBase<Render::Components::Text> {
 public:
    TextTestUI(Instance& instance) : TestUIBase(instance, "Surface Test") {
        startTime = std::chrono::steady_clock::now();
        frameCounter = 0;
    }

 protected:
    Result setupComponent() override {
        if (!instance.window().hasFont("default_mono")) {
            JST_ERROR("Font 'default_mono' not found.");
            return Result::ERROR;
        }

        Render::Components::Text::Config cfg;
        cfg.maxCharacters = 512;
        cfg.color = {0.9f, 0.9f, 0.9f, 1.0f}; // Light gray for general text
        cfg.font = instance.window().font("default_mono");
        cfg.elements = {
            {"title", {3.0f, {0.0f, 0.8f}, {1, 1}, 0.0f, "Text Component Demo"}},
            {"large", {2.0f, {-0.8f, 0.5f}, {0, 0}, 0.0f, "Large Text (2.0x)"}},
            {"medium", {1.5f, {-0.8f, 0.3f}, {0, 0}, 0.0f, "Medium Text (1.5x)"}},
            {"small", {1.0f, {-0.8f, 0.1f}, {0, 0}, 0.0f, "Small Text (1.0x)"}},
            {"tiny", {0.8f, {-0.8f, -0.1f}, {0, 0}, 0.0f, "Tiny Text (0.8x)"}},
            {"color_info", {1.2f, {0.0f, 0.4f}, {1, 1}, 0.0f, "Global Color Demo"}},
            {"color_note", {0.9f, {0.0f, 0.25f}, {1, 1}, 0.0f, "All text uses the same color"}},
            {"fps", {1.2f, {0.7f, 0.7f}, {2, 1}, 0.0f, "FPS: 0.0"}},
            {"time", {1.0f, {0.7f, 0.55f}, {2, 1}, 0.0f, "Time: 0.00s"}},
            {"counter", {1.0f, {0.7f, 0.4f}, {2, 1}, 0.0f, "Count: 0"}},
            {"top_left", {1.0f, {-0.9f, 0.9f}, {0, 2}, 0.0f, "Top Left (0,2)"}},
            {"top_center", {1.0f, {0.0f, 0.9f}, {1, 2}, 0.0f, "Top Center (1,2)"}},
            {"top_right", {1.0f, {0.9f, 0.9f}, {2, 2}, 0.0f, "Top Right (2,2)"}},
            {"bottom_left", {1.0f, {-0.9f, -0.9f}, {0, 0}, 0.0f, "Bottom Left (0,0)"}},
            {"bottom_center", {1.0f, {0.0f, -0.9f}, {1, 0}, 0.0f, "Bottom Center (1,0)"}},
            {"bottom_right", {1.0f, {0.9f, -0.9f}, {2, 0}, 0.0f, "Bottom Right (2,0)"}},
            {"bouncing", {1.5f, {0.0f, -0.5f}, {1, 1}, 0.0f, "Bouncing Text!"}},
            {"info", {0.9f, {-0.8f, -0.3f}, {0, 0}, 0.0f, "Interactive text demo"}},
            {"controls", {0.8f, {-0.8f, -0.4f}, {0, 0}, 0.0f, "Dynamic animations"}},
        };
        JST_CHECK(instance.window().build(component, cfg));
        return instance.window().bind(component);
    }

    Result setupFramebuffer() override {
        return instance.window().build<Render::Texture>(framebufferTexture, Render::Texture::Config{
            .size = {800, 600}
        });
    }

    Result setupSurface() override {
        Render::Surface::Config cfg;
        cfg.framebuffer = framebufferTexture;
        cfg.clearColor = {0.1f, 0.1f, 0.2f, 1.0f};
        cfg.multisampled = false;

        JST_CHECK(component->surface(cfg));
        JST_CHECK(instance.window().build(surface, cfg));
        return instance.window().bind(surface);
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
                fpsConfig.fill = std::format("FPS: {:.1f}", fps);
                JST_CHECK(component->update("fps", fpsConfig));
            }

            auto timeConfig = component->get("time");
            timeConfig.fill = std::format("Time: {:.2f}s", elapsed);
            JST_CHECK(component->update("time", timeConfig));

            auto counterConfig = component->get("counter");
            counterConfig.fill = std::format("Count: {}", frameCounter);
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
    return runComponentTest("Text Component Demo", "Text Component Demo", {1400, 900}, [](Instance& instance) {
        TextTestUI(instance).run();
    });
}