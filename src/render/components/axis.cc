#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "jetstream/render/base.hh"
#include "jetstream/render/components/axis.hh"
#include "jetstream/render/components/text.hh"

#include "resources/shaders/global_shaders.hh"

namespace Jetstream::Render::Components {

Axis::Axis(const Config& config) {
    this->config = config;
    this->pimpl = std::make_unique<Impl>(this->config);
}

Axis::~Axis() {
    pimpl.reset();
}

struct Axis::Impl {
    struct GridUniformBuffer {
        glm::mat4 transform;
        glm::vec2 thickness;
        F32 zoom;
        U32 numberOfLines;
    };

    const Config& config;

    GridUniformBuffer gridUniforms;
    Extent2D<F32> padScale = {1.0f, 1.0f};

    U64 totalLines = 0;

    std::vector<F32> gridPoints;
    std::vector<F32> gridVerticesData;

    bool updateGridPointsFlag = false;
    bool updateGridUniformsFlag = false;

    std::shared_ptr<Render::Buffer> gridUniformBuffer;
    std::shared_ptr<Render::Buffer> gridPointsBuffer;
    std::shared_ptr<Render::Buffer> gridVerticesBuffer;

    std::shared_ptr<Render::Kernel> gridKernel;

    std::shared_ptr<Render::Vertex> gridVertex;
    std::shared_ptr<Render::Draw> drawGridVertex;
    std::shared_ptr<Render::Program> gridProgram;

    std::shared_ptr<Text> text;

    void generateGridPoints();
    void computePaddingScale();
    Result repositionLabels();

    Impl(const Config& config) : config(config) {}
};

Result Axis::create(Window* window) {
    JST_DEBUG("[AXIS] Creating axis component.");

    if (!config.font) {
        JST_ERROR("[AXIS] Font not provided.");
        return Result::ERROR;
    }

    if (config.numberOfVerticalLines < 2) {
        JST_ERROR("[AXIS] Need at least 2 vertical lines.");
        return Result::ERROR;
    }

    if (config.numberOfHorizontalLines < 2) {
        JST_ERROR("[AXIS] Need at least 2 horizontal lines.");
        return Result::ERROR;
    }

    pimpl->totalLines = config.numberOfVerticalLines +
                        config.numberOfHorizontalLines;

    // 2 points per line, 2 coords per point.
    pimpl->gridPoints.resize(pimpl->totalLines * 4, 0.0f);
    // 6 vertices per line, 4 floats per vertex.
    pimpl->gridVerticesData.resize(pimpl->totalLines * 6 * 4, 0.0f);

    // Grid uniform buffer.
    {
        Render::Buffer::Config cfg;
        cfg.buffer = &pimpl->gridUniforms;
        cfg.elementByteSize = sizeof(pimpl->gridUniforms);
        cfg.size = 1;
        cfg.target = Render::Buffer::Target::UNIFORM;
        JST_CHECK(window->build(pimpl->gridUniformBuffer, cfg));
        JST_CHECK(window->bind(pimpl->gridUniformBuffer));
    }

    // Grid points storage buffer.
    {
        Render::Buffer::Config cfg;
        cfg.buffer = pimpl->gridPoints.data();
        cfg.elementByteSize = sizeof(F32);
        cfg.size = pimpl->gridPoints.size();
        cfg.target = Render::Buffer::Target::STORAGE;
        cfg.enableZeroCopy = false;
        JST_CHECK(window->build(pimpl->gridPointsBuffer, cfg));
        JST_CHECK(window->bind(pimpl->gridPointsBuffer));
    }

    // Grid vertices buffer (compute output + vertex input).
    {
        Render::Buffer::Config cfg;
        cfg.buffer = pimpl->gridVerticesData.data();
        cfg.elementByteSize = sizeof(F32);
        cfg.size = pimpl->gridVerticesData.size();
        cfg.target = Render::Buffer::Target::VERTEX |
                     Render::Buffer::Target::STORAGE;
        cfg.enableZeroCopy = false;
        JST_CHECK(window->build(pimpl->gridVerticesBuffer, cfg));
        JST_CHECK(window->bind(pimpl->gridVerticesBuffer));
    }

    // Thick-lines compute kernel.
    {
        Render::Kernel::Config cfg;
        cfg.gridSize = {pimpl->totalLines, 1, 1};
        cfg.kernels = GlobalKernelsPackage["thicklines"];
        cfg.buffers = {
            {pimpl->gridUniformBuffer,
             Render::Kernel::AccessMode::READ},
            {pimpl->gridPointsBuffer,
             Render::Kernel::AccessMode::READ},
            {pimpl->gridVerticesBuffer,
             Render::Kernel::AccessMode::WRITE},
        };
        JST_CHECK(window->build(pimpl->gridKernel, cfg));
    }

    // Vertex layout.
    {
        Render::Vertex::Config cfg;
        cfg.vertices = {
            {pimpl->gridVerticesBuffer, 4},
        };
        JST_CHECK(window->build(pimpl->gridVertex, cfg));
    }

    // Draw call.
    {
        Render::Draw::Config cfg;
        cfg.buffer = pimpl->gridVertex;
        cfg.mode = Render::Draw::Mode::TRIANGLES;
        JST_CHECK(window->build(pimpl->drawGridVertex, cfg));
    }

    // Grid render program.
    {
        Render::Program::Config cfg;
        cfg.shaders = GlobalShadersPackage["grid"];
        cfg.draws = {pimpl->drawGridVertex};
        cfg.buffers = {
            {pimpl->gridUniformBuffer,
             Render::Program::Target::VERTEX |
             Render::Program::Target::FRAGMENT},
        };
        cfg.enableAlphaBlending = true;
        JST_CHECK(window->build(pimpl->gridProgram, cfg));
    }

    // Text component for labels.
    {
        const U64 maxChars = 128;
        Text::Config cfg;
        cfg.maxCharacters = maxChars;
        cfg.color = config.labelColor;
        cfg.font = config.font;

        // Axis titles.
        cfg.elements["x-title"] = {
            0.85f, {0.0f, -0.99f}, {1, 2}, 0.0f, config.xTitle
        };
        cfg.elements["y-title"] = {
            0.85f, {-0.99f, 0.0f}, {1, 0}, 90.0f, config.yTitle
        };

        // X tick labels (interior lines only).
        for (U64 i = 1; i < config.numberOfVerticalLines - 1; i++) {
            cfg.elements[jst::fmt::format("x{:02d}", i)] = {
                0.85f, {0.0f, 0.99f}, {1, 0}, 0.0f, ""
            };
        }

        // Y tick labels (interior lines only).
        for (U64 i = 1; i < config.numberOfHorizontalLines - 1; i++) {
            cfg.elements[jst::fmt::format("y{:02d}", i)] = {
                0.85f, {-0.99f, 0.0f}, {2, 1}, 0.0f, ""
            };
        }

        JST_CHECK(window->build(pimpl->text, cfg));
        JST_CHECK(window->bind(pimpl->text));
    }

    // Generate grid geometry.
    pimpl->generateGridPoints();

    // Set initial uniform state.
    pimpl->computePaddingScale();

    auto transform = glm::mat4(1.0f);
    transform = glm::scale(transform,
        glm::vec3(pimpl->padScale.x, pimpl->padScale.y, 1.0f));

    pimpl->gridUniforms.transform = transform;
    pimpl->gridUniforms.thickness = {
        config.pixelSize.x * config.thickness * 3.0f,
        config.pixelSize.y * config.thickness * 3.0f
    };
    pimpl->gridUniforms.zoom = 1.0f;
    pimpl->gridUniforms.numberOfLines = pimpl->totalLines;

    pimpl->updateGridUniformsFlag = true;

    return Result::SUCCESS;
}

Result Axis::destroy(Window* window) {
    JST_CHECK(window->unbind(pimpl->text));
    JST_CHECK(window->unbind(pimpl->gridPointsBuffer));
    JST_CHECK(window->unbind(pimpl->gridVerticesBuffer));
    JST_CHECK(window->unbind(pimpl->gridUniformBuffer));

    return Result::SUCCESS;
}

Result Axis::surfaceUnderlay(Render::Surface::Config& surfaceConfig) {
    surfaceConfig.kernels.push_back(pimpl->gridKernel);
    surfaceConfig.programs.push_back(pimpl->gridProgram);

    return Result::SUCCESS;
}

Result Axis::surfaceOverlay(Render::Surface::Config& surfaceConfig) {
    JST_CHECK(pimpl->text->surface(surfaceConfig));

    return Result::SUCCESS;
}

Result Axis::present() {
    if (pimpl->updateGridPointsFlag) {
        pimpl->gridPointsBuffer->update();
        pimpl->gridKernel->update();
        pimpl->updateGridPointsFlag = false;
    }

    if (pimpl->updateGridUniformsFlag) {
        pimpl->gridUniformBuffer->update();
        pimpl->gridKernel->update();
        pimpl->updateGridUniformsFlag = false;
    }

    JST_CHECK(pimpl->text->present());

    return Result::SUCCESS;
}

Result Axis::updatePixelSize(const Extent2D<F32>& pixelSize) {
    if (config.pixelSize == pixelSize) {
        return Result::SUCCESS;
    }

    config.pixelSize = pixelSize;

    pimpl->computePaddingScale();

    auto transform = glm::mat4(1.0f);
    transform = glm::scale(transform,
        glm::vec3(pimpl->padScale.x, pimpl->padScale.y, 1.0f));

    pimpl->gridUniforms.transform = transform;
    pimpl->gridUniforms.thickness = {
        pixelSize.x * config.thickness * 3.0f,
        pixelSize.y * config.thickness * 3.0f
    };

    pimpl->updateGridUniformsFlag = true;

    JST_CHECK(pimpl->text->updatePixelSize(pixelSize));
    JST_CHECK(pimpl->repositionLabels());

    return Result::SUCCESS;
}

Result Axis::updateTickLabels(const std::vector<std::string>& xLabels,
                              const std::vector<std::string>& yLabels) {
    const U64 xCount = config.numberOfVerticalLines - 2;
    const U64 yCount = config.numberOfHorizontalLines - 2;

    for (U64 i = 0; i < xCount; i++) {
        const auto id = jst::fmt::format("x{:02d}", i + 1);
        auto element = pimpl->text->get(id);

        if (i < xLabels.size() && !xLabels[i].empty()) {
            element.fill = xLabels[i];
        } else {
            element.fill = " ";
        }

        JST_CHECK(pimpl->text->update(id, element));
    }

    for (U64 i = 0; i < yCount; i++) {
        const auto id = jst::fmt::format("y{:02d}", i + 1);
        auto element = pimpl->text->get(id);

        if (i < yLabels.size() && !yLabels[i].empty()) {
            element.fill = yLabels[i];
        } else {
            element.fill = " ";
        }

        JST_CHECK(pimpl->text->update(id, element));
    }

    return Result::SUCCESS;
}

Result Axis::updateTitles(const std::string& xTitle,
                          const std::string& yTitle) {
    {
        auto element = pimpl->text->get("x-title");
        element.fill = xTitle;
        JST_CHECK(pimpl->text->update("x-title", element));
    }

    {
        auto element = pimpl->text->get("y-title");
        element.fill = yTitle;
        JST_CHECK(pimpl->text->update("y-title", element));
    }

    return Result::SUCCESS;
}

const Extent2D<F32>& Axis::paddingScale() const {
    return pimpl->padScale;
}

void Axis::Impl::generateGridPoints() {
    const U64 numCols = config.numberOfVerticalLines;
    const U64 numRows = config.numberOfHorizontalLines;

    const F32 xStep  = 2.0f / (numCols - 1);
    const F32 yStep  = 2.0f / (numRows - 1);
    const F32 xStart = -1.0f;
    const F32 yStart = -1.0f;
    const F32 xEnd   =  1.0f;
    const F32 yEnd   =  1.0f;

    const U64 stride = 4;

    for (U64 row = 0; row < numRows; row++) {
        const F32 y = yStart + row * yStep;

        gridPoints[(row * stride) + 0] = xStart;
        gridPoints[(row * stride) + 1] = y;
        gridPoints[(row * stride) + 2] = xEnd;
        gridPoints[(row * stride) + 3] = y;
    }

    for (U64 col = 0; col < numCols; col++) {
        const F32 x = xStart + col * xStep;

        gridPoints[((col + numRows) * stride) + 0] = x;
        gridPoints[((col + numRows) * stride) + 1] = yStart;
        gridPoints[((col + numRows) * stride) + 2] = x;
        gridPoints[((col + numRows) * stride) + 3] = yEnd;
    }

    updateGridPointsFlag = true;
}

void Axis::Impl::computePaddingScale() {
    const auto PadSize = (8.0f + 4.0f + 8.0f) * 2.0f;
    padScale = {
        1.0f - config.pixelSize.x * PadSize,
        1.0f - config.pixelSize.y * PadSize,
    };
}

Result Axis::Impl::repositionLabels() {
    const auto& ps = config.pixelSize;

    // X-axis title at bottom center.
    {
        auto element = text->get("x-title");
        element.position = {0.0f, -1.0f + ps.y * 3.0f};
        JST_CHECK(text->update("x-title", element));
    }

    // Y-axis title at left center, rotated 90 deg.
    {
        auto element = text->get("y-title");
        element.position = {-1.0f + ps.x * 3.0f, 0.0f};
        JST_CHECK(text->update("y-title", element));
    }

    // X tick labels along top edge.
    const U64 numCols = config.numberOfVerticalLines;
    for (U64 i = 1; i < numCols - 1; i++) {
        const auto id = jst::fmt::format("x{:02d}", i);
        auto element = text->get(id);
        const F32 x = (2.0f * padScale.x / (numCols - 1)) * i -
                      padScale.x;
        element.position = {x, 1.0f - ps.y * 5.0f};
        JST_CHECK(text->update(id, element));
    }

    // Y tick labels along left edge.
    const U64 numRows = config.numberOfHorizontalLines;
    for (U64 i = 1; i < numRows - 1; i++) {
        const auto id = jst::fmt::format("y{:02d}", i);
        auto element = text->get(id);
        const F32 y = (2.0f * padScale.y / (numRows - 1)) * i -
                      padScale.y;
        element.position = {-1.0f + ps.x * 5.0f, y};
        JST_CHECK(text->update(id, element));
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render::Components
