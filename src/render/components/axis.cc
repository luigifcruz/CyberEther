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
        glm::vec4 lineInfo;
        glm::vec4 gridColor;
        glm::vec4 majorColor;
    };

    const Config& config;

    GridUniformBuffer gridUniforms;
    Extent2D<F32> padScale = {1.0f, 1.0f};
    F32 zoom = 1.0f;
    F32 translation = 0.0f;

    U64 totalLines = 0;
    U64 maxTotalLines = 0;
    U64 currentVerticalLines = 0;
    U64 currentHorizontalLines = 0;
    U64 maxVerticalLines = 0;
    U64 maxHorizontalLines = 0;
    U64 dividerLines = 0;
    U64 currentMajorTicks = 0;
    U64 currentMinorTicks = 0;
    U64 maxMajorTicks = 0;
    U64 maxMinorTicks = 0;

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
    void computeTickCount(U64 numCols, U64 numRows,
                          U64& outMajX, U64& outMinX,
                          U64& outMajY, U64& outMinY,
                          bool showFrameTicks) const;
    void recomputeTickCount();
    Result syncTickGeometry();
    glm::mat4 gridTransform(F32 zoom, F32 translation) const;
    F32 bandLocalY(U64 i, U64 numRows) const;
    F32 mapGridY(F32 localY) const;
    Result repositionLabels();
    Result repositionXLabels();

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

    pimpl->currentVerticalLines = config.numberOfVerticalLines;
    pimpl->currentHorizontalLines = config.numberOfHorizontalLines;
    pimpl->maxVerticalLines = config.maxNumberOfVerticalLines > 0
        ? std::max(config.maxNumberOfVerticalLines, config.numberOfVerticalLines)
        : config.numberOfVerticalLines;
    pimpl->maxHorizontalLines = config.maxNumberOfHorizontalLines > 0
        ? std::max(config.maxNumberOfHorizontalLines, config.numberOfHorizontalLines)
        : config.numberOfHorizontalLines;

    pimpl->dividerLines = (config.verticalScale < 1.0f) ? 1 : 0;

    {
        U64 majX = 0, minx = 0, majY = 0, miny = 0;
        pimpl->computeTickCount(pimpl->maxVerticalLines,
                                pimpl->maxHorizontalLines,
                                majX, minx, majY, miny,
                                true);
        pimpl->maxMajorTicks = majX + majY;
        pimpl->maxMinorTicks = minx + miny;
    }

    pimpl->recomputeTickCount();

    const U64 currentInteriorLines = config.showInteriorGrid
        ? pimpl->currentVerticalLines + pimpl->currentHorizontalLines - 4
        : 0;
    const U64 maxInteriorLines = config.showInteriorGrid
        ? pimpl->maxVerticalLines + pimpl->maxHorizontalLines - 4
        : 0;
    pimpl->totalLines = currentInteriorLines + pimpl->currentMajorTicks +
                        pimpl->currentMinorTicks + pimpl->dividerLines + 4;
    pimpl->maxTotalLines = maxInteriorLines + pimpl->maxMajorTicks +
                           pimpl->maxMinorTicks + pimpl->dividerLines + 4;

    // 2 points per line, 2 coords per point.
    pimpl->gridPoints.resize(pimpl->maxTotalLines * 4, 0.0f);
    // 6 vertices per line, 4 floats per vertex.
    pimpl->gridVerticesData.resize(pimpl->maxTotalLines * 6 * 4, 0.0f);

    // Grid uniform buffer.
    {
        Render::Buffer::Config cfg;
        cfg.buffer = &pimpl->gridUniforms;
        cfg.elementByteSize = sizeof(pimpl->gridUniforms);
        cfg.size = 1;
        cfg.target = Render::Buffer::Target::UNIFORM;
        JST_CHECK(window->build(pimpl->gridUniformBuffer, cfg));
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
    }

    // Thick-lines compute kernel.
    {
        Render::Kernel::Config cfg;
        cfg.gridSize = {pimpl->maxTotalLines, 1, 1};
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
        const U64 maxV = pimpl->maxVerticalLines;
        const U64 maxH = pimpl->maxHorizontalLines;
        const U64 maxChars = std::max<U64>(256, (maxV + maxH) * 16);

        Text::Config cfg;
        cfg.maxCharacters = maxChars;
        cfg.color = config.labelColor;
        cfg.font = config.font;

        // Axis titles.
        cfg.elements["x-title"] = {
            .scale = 0.85f,
            .position = {0.0f, -0.99f},
            .alignment = {1, 2},
            .fill = config.xTitle,
        };
        cfg.elements["y-title"] = {
            .scale = 0.85f,
            .position = {-0.99f, 0.0f},
            .alignment = {1, 0},
            .rotationDeg = 90.0f,
            .fill = config.yTitle,
        };

        // X tick labels (interior lines only, pre-allocate for max).
        for (U64 i = 1; i < maxV - 1; i++) {
            cfg.elements[jst::fmt::format("x{:02d}", i)] = {
                .scale = 0.85f,
                .position = {0.0f, 0.99f},
                .alignment = {1, 0},
            };
        }

        // Y tick labels (interior lines only, pre-allocate for max).
        for (U64 i = 1; i < maxH - 1; i++) {
            if (config.yLabelOnRight) {
                cfg.elements[jst::fmt::format("y{:02d}", i)] = {
                    .scale = 0.85f,
                    .position = {0.85f, 0.0f},
                    .alignment = {2, 1},
                };
            } else {
                cfg.elements[jst::fmt::format("y{:02d}", i)] = {
                    .scale = 0.85f,
                    .position = {-0.99f, 0.0f},
                    .alignment = {2, 1},
                };
            }
        }

        JST_CHECK(window->build(pimpl->text, cfg));
        JST_CHECK(window->bind(pimpl->text));
    }

    // Generate grid geometry.
    pimpl->generateGridPoints();
    pimpl->drawGridVertex->updateVertexCount(pimpl->totalLines * 6);

    // Set initial uniform state.
    pimpl->computePaddingScale();

    pimpl->gridUniforms.transform = pimpl->gridTransform(1.0f, 0.0f);
    pimpl->gridUniforms.thickness = {
        config.pixelSize.x * config.thickness * 3.0f,
        config.pixelSize.y * config.thickness * 3.0f
    };
    pimpl->gridUniforms.zoom = 1.0f;
    pimpl->gridUniforms.numberOfLines = pimpl->maxTotalLines;
    pimpl->gridUniforms.lineInfo = {
        static_cast<float>(config.showInteriorGrid && pimpl->currentHorizontalLines >= 2
            ? pimpl->currentHorizontalLines - 2 : 0),
        static_cast<float>(config.showInteriorGrid && pimpl->currentVerticalLines >= 2
            ? pimpl->currentVerticalLines - 2 : 0),
        static_cast<float>(pimpl->currentMajorTicks),
        static_cast<float>(pimpl->currentMinorTicks),
    };
    pimpl->gridUniforms.gridColor = {
        config.gridColor.r, config.gridColor.g, config.gridColor.b, config.gridColor.a
    };
    pimpl->gridUniforms.majorColor = {
        config.majorGridColor.r, config.majorGridColor.g,
        config.majorGridColor.b, config.majorGridColor.a
    };

    pimpl->updateGridUniformsFlag = true;

    return Result::SUCCESS;
}

Result Axis::destroy(Window* window) {
    JST_CHECK(window->unbind(pimpl->text));
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

    pimpl->gridUniforms.transform = pimpl->gridTransform(pimpl->zoom, pimpl->translation);
    pimpl->gridUniforms.thickness = {
        pixelSize.x * config.thickness * 3.0f,
        pixelSize.y * config.thickness * 3.0f
    };
    pimpl->gridUniforms.lineInfo = {
        static_cast<float>(config.showInteriorGrid && pimpl->currentHorizontalLines >= 2
            ? pimpl->currentHorizontalLines - 2 : 0),
        static_cast<float>(config.showInteriorGrid && pimpl->currentVerticalLines >= 2
            ? pimpl->currentVerticalLines - 2 : 0),
        static_cast<float>(pimpl->currentMajorTicks),
        static_cast<float>(pimpl->currentMinorTicks),
    };
    pimpl->gridUniforms.gridColor = {
        config.gridColor.r, config.gridColor.g, config.gridColor.b, config.gridColor.a
    };
    pimpl->gridUniforms.majorColor = {
        config.majorGridColor.r, config.majorGridColor.g,
        config.majorGridColor.b, config.majorGridColor.a
    };

    pimpl->updateGridUniformsFlag = true;

    if (pimpl->currentMajorTicks + pimpl->currentMinorTicks > 0) {
        pimpl->generateGridPoints();
    }

    JST_CHECK(pimpl->text->updatePixelSize(pixelSize));
    JST_CHECK(pimpl->repositionLabels());

    return Result::SUCCESS;
}

Result Axis::updateZoom(F32 zoom, F32 translation) {
    pimpl->zoom = zoom;
    pimpl->translation = translation;
    pimpl->gridUniforms.transform = pimpl->gridTransform(zoom, translation);
    pimpl->gridUniforms.zoom = zoom;
    pimpl->updateGridUniformsFlag = true;

    return Result::SUCCESS;
}

Result Axis::updateScissorRect(const Render::ScissorRect& rect) {
    pimpl->gridProgram->scissorRect(rect);
    return Result::SUCCESS;
}

Result Axis::updateLineCount(U64 verticalLines, U64 horizontalLines) {
    verticalLines = std::clamp(verticalLines, U64{2}, pimpl->maxVerticalLines);
    horizontalLines = std::clamp(horizontalLines, U64{2}, pimpl->maxHorizontalLines);

    if (verticalLines == pimpl->currentVerticalLines &&
        horizontalLines == pimpl->currentHorizontalLines) {
        return Result::SUCCESS;
    }

    pimpl->currentVerticalLines = verticalLines;
    pimpl->currentHorizontalLines = horizontalLines;
    const U64 interiorLines = config.showInteriorGrid
        ? verticalLines + horizontalLines - 4
        : 0;

    pimpl->recomputeTickCount();

    pimpl->totalLines = interiorLines + pimpl->currentMajorTicks +
                        pimpl->currentMinorTicks + pimpl->dividerLines + 4;

    pimpl->generateGridPoints();
    pimpl->drawGridVertex->updateVertexCount(pimpl->totalLines * 6);

    pimpl->gridUniforms.numberOfLines = pimpl->maxTotalLines;
    pimpl->gridUniforms.lineInfo = {
        static_cast<float>(config.showInteriorGrid && horizontalLines >= 2
            ? horizontalLines - 2 : 0),
        static_cast<float>(config.showInteriorGrid && verticalLines >= 2
            ? verticalLines - 2 : 0),
        static_cast<float>(pimpl->currentMajorTicks),
        static_cast<float>(pimpl->currentMinorTicks),
    };
    pimpl->updateGridUniformsFlag = true;

    JST_CHECK(pimpl->repositionLabels());
    JST_CHECK(pimpl->repositionXLabels());

    return Result::SUCCESS;
}

Result Axis::setShowFrameTicks(bool visible) {
    if (visible == config.showFrameTicks) {
        return Result::SUCCESS;
    }

    config.showFrameTicks = visible;

    JST_CHECK(pimpl->syncTickGeometry());

    return Result::SUCCESS;
}

Result Axis::updateTickLabels(const std::vector<std::string>& xLabels,
                              const std::vector<std::string>& yLabels) {
    const U64 xCount = pimpl->maxVerticalLines - 2;
    const U64 yCount = pimpl->maxHorizontalLines - 2;
    const U64 xVisible = pimpl->currentVerticalLines - 2;
    const U64 yVisible = pimpl->currentHorizontalLines - 2;

    for (U64 i = 0; i < xCount; i++) {
        const auto id = jst::fmt::format("x{:02d}", i + 1);
        auto element = pimpl->text->get(id);

        if (i < xVisible && i < xLabels.size() && !xLabels[i].empty()) {
            element.fill = xLabels[i];
        } else {
            element.fill = " ";
        }

        JST_CHECK(pimpl->text->update(id, element));
    }

    for (U64 i = 0; i < yCount; i++) {
        const auto id = jst::fmt::format("y{:02d}", i + 1);
        auto element = pimpl->text->get(id);

        if (i < yVisible && i < yLabels.size() && !yLabels[i].empty()) {
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

U64 Axis::currentVerticalLineCount() const {
    return pimpl->currentVerticalLines;
}

U64 Axis::currentHorizontalLineCount() const {
    return pimpl->currentHorizontalLines;
}

void Axis::Impl::computeTickCount(U64 numCols, U64 numRows,
                                  U64& outMajX, U64& outMinX,
                                  U64& outMajY, U64& outMinY,
                                  const bool showFrameTicks) const {
    outMajX = outMinX = outMajY = outMinY = 0;
    if (!showFrameTicks) {
        return;
    }
    const bool combined = (dividerLines > 0);
    const U64 xEdgeCount = combined ? 1 : 2;
    if (numCols >= 3) {
        outMajX = xEdgeCount * (numCols - 2);
    }
    if (numCols >= 2) {
        outMinX = xEdgeCount * 4 * (numCols - 1);
    }
    if (numRows >= 3) {
        outMajY = 2 * (numRows - 2);
    }
    if (numRows >= 2) {
        const U64 yMinorSegments = combined
            ? (numRows - 1)
            : (numRows - 1 + dividerLines);
        outMinY = 2 * 4 * yMinorSegments;
    }
}

void Axis::Impl::recomputeTickCount() {
    U64 majX = 0, minx = 0, majY = 0, miny = 0;
    computeTickCount(currentVerticalLines, currentHorizontalLines,
                     majX, minx, majY, miny,
                     config.showFrameTicks);
    currentMajorTicks = majX + majY;
    currentMinorTicks = minx + miny;
}

Result Axis::Impl::syncTickGeometry() {
    recomputeTickCount();

    const U64 interiorLines = config.showInteriorGrid
        ? currentVerticalLines + currentHorizontalLines - 4
        : 0;
    totalLines = interiorLines + currentMajorTicks +
                 currentMinorTicks + dividerLines + 4;

    generateGridPoints();
    drawGridVertex->updateVertexCount(totalLines * 6);

    gridUniforms.lineInfo = {
        static_cast<float>(config.showInteriorGrid && currentHorizontalLines >= 2
            ? currentHorizontalLines - 2 : 0),
        static_cast<float>(config.showInteriorGrid && currentVerticalLines >= 2
            ? currentVerticalLines - 2 : 0),
        static_cast<float>(currentMajorTicks),
        static_cast<float>(currentMinorTicks),
    };
    updateGridUniformsFlag = true;

    return Result::SUCCESS;
}

void Axis::Impl::generateGridPoints() {
    const U64 numCols = currentVerticalLines;
    const U64 numRows = currentHorizontalLines;
    const U64 interiorH = (config.showInteriorGrid && numRows >= 2)
        ? (numRows - 2) : 0;
    const U64 interiorV = (config.showInteriorGrid && numCols >= 2)
        ? (numCols - 2) : 0;

    const F32 xStep  = 2.0f / (numCols - 1);
    const F32 xStart = -1.0f;
    const F32 xEnd   =  1.0f;
    const F32 yStart    = -1.0f;
    const F32 yEnd      =  1.0f;
    const F32 bandBottom = bandLocalY(0, numRows);
    const F32 bandTop    = bandLocalY(numRows - 1, numRows);

    const F32 tickMajX = config.pixelSize.y * config.majorTickLengthPx;
    const F32 tickMinX = config.pixelSize.y * config.minorTickLengthPx;
    const F32 tickMajY = config.pixelSize.x * config.majorTickLengthPx;
    const F32 tickMinY = config.pixelSize.x * config.minorTickLengthPx;

    const U64 stride = 4;

    std::fill(gridPoints.begin(), gridPoints.end(), 0.0f);

    U64 idx = 0;

    for (U64 row = 1; row <= interiorH; row++) {
        const F32 y = bandLocalY(row, numRows);
        gridPoints[(idx * stride) + 0] = xStart;
        gridPoints[(idx * stride) + 1] = y;
        gridPoints[(idx * stride) + 2] = xEnd;
        gridPoints[(idx * stride) + 3] = y;
        idx++;
    }

    for (U64 col = 1; col <= interiorV; col++) {
        const F32 x = xStart + col * xStep;
        gridPoints[(idx * stride) + 0] = x;
        gridPoints[(idx * stride) + 1] = bandBottom;
        gridPoints[(idx * stride) + 2] = x;
        gridPoints[(idx * stride) + 3] = bandTop;
        idx++;
    }

    if (config.showFrameTicks) {
        std::vector<F32> majX;
        std::vector<F32> majY;
        if (numCols >= 3) {
            majX.reserve(numCols);
            for (U64 col = 1; col <= numCols - 2; col++) {
                majX.push_back(xStart + col * xStep);
            }
        }
        if (numRows >= 3) {
            majY.reserve(numRows);
            for (U64 row = 1; row <= numRows - 2; row++) {
                majY.push_back(bandLocalY(row, numRows));
            }
        }

        for (const F32 x : majX) {
            gridPoints[(idx * stride) + 0] = x;
            gridPoints[(idx * stride) + 1] = yEnd;
            gridPoints[(idx * stride) + 2] = x;
            gridPoints[(idx * stride) + 3] = yEnd - tickMajX;
            idx++;
            if (dividerLines == 0) {
                gridPoints[(idx * stride) + 0] = x;
                gridPoints[(idx * stride) + 1] = yStart;
                gridPoints[(idx * stride) + 2] = x;
                gridPoints[(idx * stride) + 3] = yStart + tickMajX;
                idx++;
            }
        }
        for (const F32 y : majY) {
            gridPoints[(idx * stride) + 0] = xStart;
            gridPoints[(idx * stride) + 1] = y;
            gridPoints[(idx * stride) + 2] = xStart + tickMajY;
            gridPoints[(idx * stride) + 3] = y;
            idx++;
            gridPoints[(idx * stride) + 0] = xEnd;
            gridPoints[(idx * stride) + 1] = y;
            gridPoints[(idx * stride) + 2] = xEnd - tickMajY;
            gridPoints[(idx * stride) + 3] = y;
            idx++;
        }

        auto emitMinorX = [&](F32 x) {
            gridPoints[(idx * stride) + 0] = x;
            gridPoints[(idx * stride) + 1] = yEnd;
            gridPoints[(idx * stride) + 2] = x;
            gridPoints[(idx * stride) + 3] = yEnd - tickMinX;
            idx++;
            if (dividerLines == 0) {
                gridPoints[(idx * stride) + 0] = x;
                gridPoints[(idx * stride) + 1] = yStart;
                gridPoints[(idx * stride) + 2] = x;
                gridPoints[(idx * stride) + 3] = yStart + tickMinX;
                idx++;
            }
        };

        for (U64 col = 0; col + 1 < numCols; col++) {
            const F32 a = xStart + col * xStep;
            const F32 b = a + xStep;
            for (U64 s = 1; s <= 4; s++) {
                emitMinorX(a + (b - a) *
                    (static_cast<F32>(s) / 5.0f));
            }
        }

        std::vector<F32> minorYBounds;
        minorYBounds.reserve(numRows + dividerLines);
        minorYBounds.push_back(yStart);
        if (dividerLines > 0) {
            minorYBounds.push_back(bandBottom);
        }
        for (U64 row = 1; row <= numRows - 2; row++) {
            minorYBounds.push_back(bandLocalY(row, numRows));
        }
        minorYBounds.push_back(yEnd);

        for (size_t i = 0; i + 1 < minorYBounds.size(); i++) {
            if (dividerLines > 0 && i == 0) {
                continue;
            }
            const F32 a = minorYBounds[i];
            const F32 b = minorYBounds[i + 1];
            for (U64 s = 1; s <= 4; s++) {
                const F32 y = a + (b - a) * (static_cast<F32>(s) / 5.0f);
                gridPoints[(idx * stride) + 0] = xStart;
                gridPoints[(idx * stride) + 1] = y;
                gridPoints[(idx * stride) + 2] = xStart + tickMinY;
                gridPoints[(idx * stride) + 3] = y;
                idx++;
                gridPoints[(idx * stride) + 0] = xEnd;
                gridPoints[(idx * stride) + 1] = y;
                gridPoints[(idx * stride) + 2] = xEnd - tickMinY;
                gridPoints[(idx * stride) + 3] = y;
                idx++;
            }
        }
    }

    // Divider between the line panel and the panel below it (waterfall).
    if (dividerLines > 0) {
        gridPoints[(idx * stride) + 0] = xStart;
        gridPoints[(idx * stride) + 1] = bandBottom;
        gridPoints[(idx * stride) + 2] = xEnd;
        gridPoints[(idx * stride) + 3] = bandBottom;
        idx++;
    }

    // Frame: bottom, top, left, right.
    gridPoints[(idx * stride) + 0] = xStart; gridPoints[(idx * stride) + 1] = yStart;
    gridPoints[(idx * stride) + 2] = xEnd;   gridPoints[(idx * stride) + 3] = yStart; idx++;
    gridPoints[(idx * stride) + 0] = xStart; gridPoints[(idx * stride) + 1] = yEnd;
    gridPoints[(idx * stride) + 2] = xEnd;   gridPoints[(idx * stride) + 3] = yEnd;   idx++;
    gridPoints[(idx * stride) + 0] = xStart; gridPoints[(idx * stride) + 1] = yStart;
    gridPoints[(idx * stride) + 2] = xStart; gridPoints[(idx * stride) + 3] = yEnd;   idx++;
    gridPoints[(idx * stride) + 0] = xEnd;   gridPoints[(idx * stride) + 1] = yStart;
    gridPoints[(idx * stride) + 2] = xEnd;   gridPoints[(idx * stride) + 3] = yEnd;   idx++;

    updateGridPointsFlag = true;
}

void Axis::Impl::computePaddingScale() {
    const auto PadSize = (8.0f + 4.0f + 8.0f) * 2.0f;
    padScale = {
        1.0f - config.pixelSize.x * PadSize,
        1.0f - config.pixelSize.y * PadSize,
    };
}

glm::mat4 Axis::Impl::gridTransform(F32 zoom, F32 translation) const {
    auto transform = glm::mat4(1.0f);
    transform = glm::translate(transform, glm::vec3(translation * zoom, 0.0f, 0.0f));
    transform = glm::scale(transform, glm::vec3(padScale.x, padScale.y, 1.0f));
    return transform;
}

F32 Axis::Impl::bandLocalY(U64 i, U64 numRows) const {
    const F32 bottom = 1.0f - 2.0f * config.verticalScale;
    const F32 top = 1.0f;
    return bottom + (top - bottom) *
                    (static_cast<F32>(i) / static_cast<F32>(numRows - 1));
}

F32 Axis::Impl::mapGridY(F32 localY) const {
    return localY * padScale.y;
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
    const U64 numCols = currentVerticalLines;
    for (U64 i = 1; i < numCols - 1; i++) {
        const auto id = jst::fmt::format("x{:02d}", i);
        auto element = text->get(id);
        const F32 x = (2.0f * padScale.x / (numCols - 1)) * i -
                      padScale.x;
        element.position = {x, 1.0f - ps.y * 5.0f};
        JST_CHECK(text->update(id, element));
    }

    // Y tick labels along the configured edge.
    const U64 numRows = currentHorizontalLines;
    const F32 sideTickOffset = config.majorTickLengthPx + 4.0f;
    for (U64 i = 1; i < numRows - 1; i++) {
        const auto id = jst::fmt::format("y{:02d}", i);
        auto element = text->get(id);
        const F32 localY = bandLocalY(i, numRows);
        const F32 y = mapGridY(localY);
        if (config.yLabelOnRight) {
            element.position = {padScale.x - ps.x * sideTickOffset, y};
            element.alignment = {2, 1};
        } else {
            element.position = {-padScale.x - ps.x * 4.0f, y};
            element.alignment = {2, 1};
        }
        JST_CHECK(text->update(id, element));
    }

    return Result::SUCCESS;
}

Result Axis::Impl::repositionXLabels() {
    const auto& ps = config.pixelSize;
    const U64 numCols = currentVerticalLines;

    for (U64 i = 1; i < numCols - 1; i++) {
        const auto id = jst::fmt::format("x{:02d}", i);
        auto element = text->get(id);

        // Base NDC position of the interior grid line (before zoom/pan).
        const F32 baseX = (2.0f / (numCols - 1)) * i - 1.0f;

        // Apply the same pan/zoom that interior grid lines get.
        const F32 x = (baseX * zoom + translation * zoom) * padScale.x;

        element.position = {x, 1.0f - ps.y * 5.0f};
        JST_CHECK(text->update(id, element));
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render::Components
