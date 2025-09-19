#include "common.hh"
#include "jetstream/render/components/shapes.hh"
#include <cmath>
#include <vector>
#include <string>

class ShapesTestUI : public TestUIBase<Render::Components::Shapes> {
 public:
    ShapesTestUI(Instance& instance) : TestUIBase(instance, "Shapes Test Grid") {}

 protected:
    Result setupComponent() override {
        Render::Components::Shapes::Config cfg;
        cfg.pixelSize = {2.0f / 1400.0f, 2.0f / 900.0f};

        cfg.elements = {
            {"basic_triangle", {.type = Render::Components::Shapes::Type::TRIANGLE, .numberOfInstances = 1}},
            {"basic_rect", {.type = Render::Components::Shapes::Type::RECT, .numberOfInstances = 1}},
            {"basic_circle", {.type = Render::Components::Shapes::Type::CIRCLE, .numberOfInstances = 1}},
            {"basic_rect_line", {.type = Render::Components::Shapes::Type::LINE, .numberOfInstances = 1}},

            {"rotate_triangle", {.type = Render::Components::Shapes::Type::TRIANGLE, .numberOfInstances = 1}},
            {"rotate_rect", {.type = Render::Components::Shapes::Type::RECT, .numberOfInstances = 1}},
            {"rotate_circle", {.type = Render::Components::Shapes::Type::CIRCLE, .numberOfInstances = 1}},
            {"rotate_rect_line", {.type = Render::Components::Shapes::Type::LINE, .numberOfInstances = 1}},

            {"corner_triangle", {.type = Render::Components::Shapes::Type::TRIANGLE, .numberOfInstances = 1, .cornerRadius = 0.5f}},
            {"corner_rect", {.type = Render::Components::Shapes::Type::RECT, .numberOfInstances = 1, .cornerRadius = 0.5f}},
            {"corner_circle", {.type = Render::Components::Shapes::Type::CIRCLE, .numberOfInstances = 1, .cornerRadius = 0.5f}},
            {"corner_rect_line", {.type = Render::Components::Shapes::Type::LINE, .numberOfInstances = 1, .cornerRadius = 0.5f}},

            {"border_triangle", {.type = Render::Components::Shapes::Type::TRIANGLE, .numberOfInstances = 1, .borderWidth = 2.0f, .borderColor = {1.0f, 0.3f, 0.3f, 1.0f}}},
            {"border_rect", {.type = Render::Components::Shapes::Type::RECT, .numberOfInstances = 1, .borderWidth = 1.0f, .borderColor = {0.3f, 1.0f, 0.3f, 1.0f}}},
            {"border_circle", {.type = Render::Components::Shapes::Type::CIRCLE, .numberOfInstances = 1, .borderWidth = 1.0f, .borderColor = {0.3f, 0.3f, 1.0f, 1.0f}}},
            {"border_rect_line", {.type = Render::Components::Shapes::Type::LINE, .numberOfInstances = 1, .borderWidth = 1.0f, .borderColor = {1.0f, 1.0f, 0.3f, 1.0f}}},

            {"triangle_s", {.type = Render::Components::Shapes::Type::TRIANGLE, .numberOfInstances = 1}},
            {"triangle_m", {.type = Render::Components::Shapes::Type::TRIANGLE, .numberOfInstances = 1}},
            {"triangle_l", {.type = Render::Components::Shapes::Type::TRIANGLE, .numberOfInstances = 1}},
            {"triangle_xl", {.type = Render::Components::Shapes::Type::TRIANGLE, .numberOfInstances = 1}},

            {"rect_s", {.type = Render::Components::Shapes::Type::RECT, .numberOfInstances = 1}},
            {"rect_m", {.type = Render::Components::Shapes::Type::RECT, .numberOfInstances = 1}},
            {"rect_l", {.type = Render::Components::Shapes::Type::RECT, .numberOfInstances = 1}},
            {"rect_xl", {.type = Render::Components::Shapes::Type::RECT, .numberOfInstances = 1}},

            {"circle_s", {.type = Render::Components::Shapes::Type::CIRCLE, .numberOfInstances = 1}},
            {"circle_m", {.type = Render::Components::Shapes::Type::CIRCLE, .numberOfInstances = 1}},
            {"circle_l", {.type = Render::Components::Shapes::Type::CIRCLE, .numberOfInstances = 1}},
            {"circle_xl", {.type = Render::Components::Shapes::Type::CIRCLE, .numberOfInstances = 1}},

            {"line_s", {.type = Render::Components::Shapes::Type::LINE, .numberOfInstances = 1}},
            {"line_m", {.type = Render::Components::Shapes::Type::LINE, .numberOfInstances = 1}},
            {"line_l", {.type = Render::Components::Shapes::Type::LINE, .numberOfInstances = 1}},
            {"line_xl", {.type = Render::Components::Shapes::Type::LINE, .numberOfInstances = 1}},
        };

        JST_CHECK(instance.window().build(component, cfg));
        JST_CHECK(instance.window().bind(component));
        setupShapeGrid();
        return Result::SUCCESS;
    }

    Result setupFramebuffer() override {
        return instance.window().build<Render::Texture>(framebufferTexture, Render::Texture::Config{
            .size = {1400, 900}
        });
    }

    Result setupSurface() override {
        Render::Surface::Config cfg;
        cfg.framebuffer = framebufferTexture;
        cfg.clearColor = {0.05f, 0.05f, 0.05f, 1.0f};
        cfg.multisampled = false;

        JST_CHECK(component->surface(cfg));
        JST_CHECK(instance.window().build(surface, cfg));
        return instance.window().bind(surface);
    }

    void updatePixelSize(U64 width, U64 height) override {
        component->updatePixelSize({2.0f / static_cast<F32>(width), 2.0f / static_cast<F32>(height)});
    }

    std::string getComponentName() const override {
        return "Shapes Grid";
    }

    Result updateComponent() override {
        updateRotatingShapes();
        return Result::SUCCESS;
    }

    Result renderInfoPanel([[maybe_unused]] const ImVec2& totalSize, [[maybe_unused]] const ImVec2& contentSize) override {
        ImGui::Separator();
        ImGui::Text("Grid Layout (8x4):");
        ImGui::TextWrapped("Basic shapes, rotating variations, corner radius, border width, and size variations");
        ImGui::Separator();
        ImGui::Text("Shape Types:");
        ImGui::BulletText("Triangles");
        ImGui::BulletText("Rectangles");
        ImGui::BulletText("Circles");
        ImGui::BulletText("Lines (thin rects)");
        return Result::SUCCESS;
    }

 private:
    void setupShapeGrid() {
        const float gridSpacingX = 0.4f;
        const float gridSpacingY = 0.22f;
        const float startX = -0.6f;
        const float startY = 0.75f;
        const float defaultSize = 35.0f;
        const float sizeS = 20.0f;
        const float sizeM = 30.0f;
        const float sizeL = 40.0f;
        const float sizeXL = 50.0f;
        const float lineThickness = 4.0f;
        ColorRGBA<F32> basicColor = {0.8f, 0.8f, 0.8f, 1.0f};
        ColorRGBA<F32> rotateColor = {0.9f, 0.6f, 0.2f, 1.0f};
        ColorRGBA<F32> cornerColor = {0.2f, 0.8f, 0.5f, 1.0f};
        ColorRGBA<F32> borderColor = {0.3f, 0.6f, 0.9f, 1.0f};
        ColorRGBA<F32> triangleColor = {1.0f, 0.3f, 0.3f, 1.0f};
        ColorRGBA<F32> rectColor = {0.3f, 1.0f, 0.3f, 1.0f};
        ColorRGBA<F32> circleColor = {0.3f, 0.3f, 1.0f, 1.0f};
        ColorRGBA<F32> lineColor = {1.0f, 1.0f, 0.3f, 1.0f};

        auto setupShape = [&](const std::string& name, float x, float y, float width, float height,
                             const ColorRGBA<F32>& color, float rotation = 0.0f) {
            std::span<ColorRGBA<F32>> colors;
            JST_CHECK_THROW(component->getColors(name, colors));
            colors[0] = color;
            JST_CHECK_THROW(component->updateColors(name));

            std::span<Extent2D<F32>> positions;
            JST_CHECK_THROW(component->getPositions(name, positions));
            positions[0] = {x, y};
            JST_CHECK_THROW(component->updatePositions(name));

            std::span<Extent2D<F32>> sizes;
            JST_CHECK_THROW(component->getSizes(name, sizes));
            sizes[0] = {width, height};
            JST_CHECK_THROW(component->updateSizes(name));

            if (rotation != 0.0f) {
                std::span<F32> rotations;
                JST_CHECK_THROW(component->getRotations(name, rotations));
                rotations[0] = rotation;
                JST_CHECK_THROW(component->updateRotations(name));
            }
        };

        setupShape("basic_triangle", startX, startY, defaultSize, defaultSize, basicColor);
        setupShape("basic_rect", startX + gridSpacingX, startY, defaultSize, defaultSize, basicColor);
        setupShape("basic_circle", startX + 2*gridSpacingX, startY, defaultSize, defaultSize, basicColor);
        setupShape("basic_rect_line", startX + 3*gridSpacingX, startY, defaultSize, lineThickness, basicColor);

        setupShape("rotate_triangle", startX, startY - gridSpacingY, defaultSize, defaultSize, rotateColor);
        setupShape("rotate_rect", startX + gridSpacingX, startY - gridSpacingY, defaultSize, defaultSize, rotateColor);
        setupShape("rotate_circle", startX + 2*gridSpacingX, startY - gridSpacingY, defaultSize, defaultSize, rotateColor);
        setupShape("rotate_rect_line", startX + 3*gridSpacingX, startY - gridSpacingY, defaultSize, lineThickness, rotateColor);

        setupShape("corner_triangle", startX, startY - 2*gridSpacingY, defaultSize, defaultSize, cornerColor);
        setupShape("corner_rect", startX + gridSpacingX, startY - 2*gridSpacingY, defaultSize, defaultSize, cornerColor);
        setupShape("corner_circle", startX + 2*gridSpacingX, startY - 2*gridSpacingY, defaultSize, defaultSize, cornerColor);
        setupShape("corner_rect_line", startX + 3*gridSpacingX, startY - 2*gridSpacingY, defaultSize, lineThickness, cornerColor);

        setupShape("border_triangle", startX, startY - 3*gridSpacingY, defaultSize, defaultSize, borderColor);
        setupShape("border_rect", startX + gridSpacingX, startY - 3*gridSpacingY, defaultSize, defaultSize, borderColor);
        setupShape("border_circle", startX + 2*gridSpacingX, startY - 3*gridSpacingY, defaultSize, defaultSize, borderColor);
        setupShape("border_rect_line", startX + 3*gridSpacingX, startY - 3*gridSpacingY, defaultSize, lineThickness, borderColor);

        setupShape("triangle_s", startX, startY - 4*gridSpacingY, sizeS, sizeS, triangleColor);
        setupShape("triangle_m", startX + gridSpacingX, startY - 4*gridSpacingY, sizeM, sizeM, triangleColor);
        setupShape("triangle_l", startX + 2*gridSpacingX, startY - 4*gridSpacingY, sizeL, sizeL, triangleColor);
        setupShape("triangle_xl", startX + 3*gridSpacingX, startY - 4*gridSpacingY, sizeXL, sizeXL, triangleColor);

        setupShape("rect_s", startX, startY - 5*gridSpacingY, sizeS, sizeS, rectColor);
        setupShape("rect_m", startX + gridSpacingX, startY - 5*gridSpacingY, sizeM, sizeM, rectColor);
        setupShape("rect_l", startX + 2*gridSpacingX, startY - 5*gridSpacingY, sizeL, sizeL, rectColor);
        setupShape("rect_xl", startX + 3*gridSpacingX, startY - 5*gridSpacingY, sizeXL, sizeXL, rectColor);

        setupShape("circle_s", startX, startY - 6*gridSpacingY, sizeS, sizeS, circleColor);
        setupShape("circle_m", startX + gridSpacingX, startY - 6*gridSpacingY, sizeM, sizeM, circleColor);
        setupShape("circle_l", startX + 2*gridSpacingX, startY - 6*gridSpacingY, sizeL, sizeL, circleColor);
        setupShape("circle_xl", startX + 3*gridSpacingX, startY - 6*gridSpacingY, sizeXL, sizeXL, circleColor);

        setupShape("line_s", startX, startY - 7*gridSpacingY, sizeS, lineThickness, lineColor);
        setupShape("line_m", startX + gridSpacingX, startY - 7*gridSpacingY, sizeM, lineThickness, lineColor);
        setupShape("line_l", startX + 2*gridSpacingX, startY - 7*gridSpacingY, sizeL, lineThickness, lineColor);
        setupShape("line_xl", startX + 3*gridSpacingX, startY - 7*gridSpacingY, sizeXL, lineThickness, lineColor);
    }

    void updateRotatingShapes() {
        static float rotation = 0.0f;
        rotation += 0.02f;

        const std::vector<std::string> rotatingShapes = {
            "rotate_triangle", "rotate_rect", "rotate_circle", "rotate_rect_line"
        };

        for (const auto& shapeName : rotatingShapes) {
            std::span<F32> rotations;
            if (component->getRotations(shapeName, rotations) == Result::SUCCESS) {
                rotations[0] = rotation;
                component->updateRotations(shapeName);
            }
        }
    }
};

int main() {
    return runComponentTest("Shapes Test Grid", "Shapes Test Grid", {1400, 900}, [](Instance& instance) {
        ShapesTestUI(instance).run();
    });
}
