#include "common.hh"
#include "jetstream/render/components/text.hh"

class SurfaceTestUI : public TestUIBase<Render::Components::Text> {
 public:
    SurfaceTestUI(Instance& instance) : TestUIBase(instance, "Surface Test") {}

 protected:
    Result setupComponent() override {
        if (!instance.window().hasFont("default_mono")) {
            JST_ERROR("Font 'default_mono' not found.");
            return Result::ERROR;
        }

        Render::Components::Text::Config cfg;
        cfg.maxCharacters = 64;
        cfg.color = {1.0f, 1.0f, 1.0f, 1.0f};
        cfg.font = instance.window().font("default_mono");
        cfg.elements = {
            {"hello", {2.5f, {0.0f, 0.0f}, {1, 1}, 0.0f, "Hello Surface!"}},
            {"info", {1.0f, {0.0f, -0.5f}, {1, 1}, 0.0f, "Text Component Test"}},
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
        cfg.clearColor = {0.2f, 0.4f, 0.8f, 1.0f}; // Blue color
        cfg.multisampled = false;

        JST_CHECK(component->surface(cfg));
        JST_CHECK(instance.window().build(surface, cfg));
        return instance.window().bind(surface);
    }

    void updatePixelSize(U64 width, U64 height) override {
        component->updatePixelSize({2.0f / static_cast<F32>(width), 2.0f / static_cast<F32>(height)});
    }

    std::string getComponentName() const override {
        return "Text";
    }

    Result renderInfoPanel([[maybe_unused]] const ImVec2& totalSize, [[maybe_unused]] const ImVec2& contentSize) override {
        ImGui::Separator();
        ImGui::Text("Elements:");
        ImGui::BulletText("Hello Surface!");
        ImGui::BulletText("Text Component Test");
        return Result::SUCCESS;
    }
};

int main() {
    return runComponentTest("Surface Test", "Surface Test", {1200, 800}, [](Instance& instance) {
        SurfaceTestUI(instance).run();
    });
}