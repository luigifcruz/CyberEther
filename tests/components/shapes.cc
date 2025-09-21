#include "common.hh"
#include "jetstream/render/components/shapes.hh"

class ShapesTestUI : public TestUIBase<Render::Components::Shapes> {
 public:
    ShapesTestUI(Instance& instance) : TestUIBase(instance, "Shapes Test") {}

 protected:
    Result setupComponent() override {
        Render::Components::Shapes::Config cfg;
        cfg.pixelSize = {2.0f / 800.0f, 2.0f / 600.0f};
        cfg.elements = {
            {"rectangles", {
                .type = Render::Components::Shapes::Type::RECT,
                .numberOfInstances = 1,
                .cornerRadius = 0.0f,
                .borderWidth = 0.0f,
                .borderColor = {0.7f, 0.7f, 0.7f, 1.0f}
            }},
            {"circles", {
                .type = Render::Components::Shapes::Type::CIRCLE,
                .numberOfInstances = 1,
                .cornerRadius = 0.0f,
                .borderWidth = 0.0f,
                .borderColor = {0.3f, 0.7f, 0.9f, 1.0f}
            }},
            {"triangles", {
                .type = Render::Components::Shapes::Type::TRIANGLE,
                .numberOfInstances = 1,
                .cornerRadius = 0.0f,
                .borderWidth = 0.0f,
                .borderColor = {0.9f, 0.3f, 0.7f, 1.0f}
            }},
        };

        JST_CHECK(instance.window().build(component, cfg));
        JST_CHECK(instance.window().bind(component));

        setupShapes();
        return Result::SUCCESS;
    }

    Result setupFramebuffer() override {
        return instance.window().build<Render::Texture>(framebufferTexture, Render::Texture::Config{
            .size = {800, 600}
        });
    }

    Result setupSurface() override {
        Render::Surface::Config cfg;
        cfg.framebuffer = framebufferTexture;
        cfg.clearColor = {0.1f, 0.1f, 0.1f, 1.0f};
        cfg.multisampled = false;

        JST_CHECK(component->surface(cfg));
        JST_CHECK(instance.window().build(surface, cfg));
        return instance.window().bind(surface);
    }

    void updatePixelSize(U64 width, U64 height) override {
        component->updatePixelSize({2.0f / static_cast<F32>(width), 2.0f / static_cast<F32>(height)});
    }

    std::string getComponentName() const override {
        return "Shapes";
    }

    Result renderInfoPanel([[maybe_unused]] const ImVec2& totalSize, [[maybe_unused]] const ImVec2& contentSize) override {
        ImGui::Separator();
        ImGui::Text("Elements:");
        ImGui::BulletText("Rectangle (Red)");
        ImGui::BulletText("Circle (Green)");
        ImGui::BulletText("Triangle (Blue)");
        return Result::SUCCESS;
    }

 private:
    void setupShapes() {
        {
            std::span<ColorRGBA<F32>> colors;
            JST_CHECK_THROW(component->getColors("rectangles", colors));
            colors[0] = {1.0f, 0.0f, 0.0f, 1.0f};
            JST_CHECK_THROW(component->updateColors("rectangles"));

            std::span<Extent2D<F32>> positions;
            JST_CHECK_THROW(component->getPositions("rectangles", positions));
            positions[0] = {-0.6f, 0.0f};
            JST_CHECK_THROW(component->updatePositions("rectangles"));

            std::span<Extent2D<F32>> sizes;
            JST_CHECK_THROW(component->getSizes("rectangles", sizes));
            sizes[0] = {50.0f, 50.0f};
            JST_CHECK_THROW(component->updateSizes("rectangles"));
        }

        {
            std::span<ColorRGBA<F32>> colors;
            JST_CHECK_THROW(component->getColors("circles", colors));
            colors[0] = {0.0f, 1.0f, 0.0f, 1.0f};
            JST_CHECK_THROW(component->updateColors("circles"));

            std::span<Extent2D<F32>> positions;
            JST_CHECK_THROW(component->getPositions("circles", positions));
            positions[0] = {0.0f, 0.0f};
            JST_CHECK_THROW(component->updatePositions("circles"));

            std::span<Extent2D<F32>> sizes;
            JST_CHECK_THROW(component->getSizes("circles", sizes));
            sizes[0] = {50.0f, 50.0f};
            JST_CHECK_THROW(component->updateSizes("circles"));
        }

        {
            std::span<ColorRGBA<F32>> colors;
            JST_CHECK_THROW(component->getColors("triangles", colors));
            colors[0] = {0.0f, 0.0f, 1.0f, 1.0f};
            JST_CHECK_THROW(component->updateColors("triangles"));

            std::span<Extent2D<F32>> positions;
            JST_CHECK_THROW(component->getPositions("triangles", positions));
            positions[0] = {0.6f, 0.0f};
            JST_CHECK_THROW(component->updatePositions("triangles"));

            std::span<Extent2D<F32>> sizes;
            JST_CHECK_THROW(component->getSizes("triangles", sizes));
            sizes[0] = {50.0f, 50.0f};
            JST_CHECK_THROW(component->updateSizes("triangles"));
        }
    }
};

int main() {
    return runComponentTest("Shapes Test", "Shapes Test", {1200, 800}, [](Instance& instance) {
        ShapesTestUI(instance).run();
    });
}