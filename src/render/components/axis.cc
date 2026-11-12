#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>

#include "jetstream/render/base.hh"
#include "jetstream/render/components/axis.hh"
#include "jetstream/render/components/text.hh"
#include "jetstream/tools/numeric.hh"

#include "resources/shaders/global_shaders.hh"

namespace Jetstream::Render::Components {

namespace {

constexpr F32 kTickLabelScale = 0.85f;
constexpr U64 kMaxVerticalLines = 65;
constexpr U64 kMaxHorizontalLines = 17;
constexpr U64 kMinorTicksPerInterval = 4;

std::string FormatTickValue(F64 value, const Axis::TickScale& scale) {
    // Avoid "-0" for a tick that lands on zero.
    if (std::abs(value) < scale.step * 1e-9) {
        value = 0.0;
    }
    return jst::fmt::format("{:.{}f}", value, scale.decimals);
}

}  // namespace

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
    bool updateVertexCountFlag = false;

    std::shared_ptr<Render::Buffer> gridUniformBuffer;
    std::shared_ptr<Render::Buffer> gridPointsBuffer;
    std::shared_ptr<Render::Buffer> gridVerticesBuffer;

    std::shared_ptr<Render::Kernel> gridKernel;

    std::shared_ptr<Render::Vertex> gridVertex;
    std::shared_ptr<Render::Draw> drawGridVertex;
    std::shared_ptr<Render::Program> gridProgram;

    std::shared_ptr<Text> text;

    TickFormatter xTickFormatter;
    TickFormatter yTickFormatter;

    bool xRangeEnabled = false;
    F64 xRangeLeft = -1.0;
    F64 xRangeRight = 1.0;
    TickScale xTicks;
    std::vector<F32> xTickPositions;
    std::vector<F32> xMinorPositions;

    bool yRangeEnabled = false;
    F64 yRangeBottom = -1.0;
    F64 yRangeTop = 1.0;
    PositionMap yRangeMap;
    F64 yRangeStep = 0.0;
    TickScale yTicks;
    std::vector<F32> yTickPositions;
    std::vector<bool> yLabelVisible;
    std::vector<F32> yMinorPositions;

    void generateGridPoints();
    void computePaddingScale();
    static void computeTickCount(U64 numCols, U64 numRows,
                                 U64& outMajX, U64& outMinX,
                                 U64& outMajY, U64& outMinY,
                                 bool showFrameTicks, bool combined);
    void recomputeTickCount();
    Result syncTickGeometry();
    glm::mat4 gridTransform(F32 zoom, F32 translation) const;
    F32 bandLocalY(U64 i, U64 numRows) const;
    F32 mapGridY(F32 localY) const;
    Result repositionLabels();
    Result repositionXLabels();
    Result syncResponsiveGrid();
    Result syncTickLabels();
    U64 responsiveLineCount(F32 availablePx,
                            F32 requiredSpacingPx,
                            U64 maximum) const;
    F32 xLinePosition(U64 i, U64 numCols) const;
    F32 xRangePosition(F64 value) const;
    F32 yLinePosition(U64 i, U64 numRows) const;
    F32 yRangePosition(F64 value) const;
    void computeYRangeTicks(F32 availableHeightPx,
                            F32 minimumSpacingPx,
                            F32 collisionPadding);
    void computeXRangeTicks(F32 availableWidthPx,
                            F32 minimumSpacingPx,
                            F32 collisionPadding);

    Impl(const Config& config) : config(config) {}
};

Result Axis::create(Window* window) {
    JST_DEBUG("[AXIS] Creating axis component.");

    if (!config.font) {
        JST_ERROR("[AXIS] Font not provided.");
        return Result::ERROR;
    }

    pimpl->currentVerticalLines = 3;
    pimpl->currentHorizontalLines = 3;
    pimpl->maxVerticalLines = kMaxVerticalLines;
    pimpl->maxHorizontalLines = kMaxHorizontalLines;

    pimpl->dividerLines = (config.verticalScale < 1.0f) ? 1 : 0;

    {
        U64 majX = 0, minx = 0, majY = 0, miny = 0;
        // Full-height mode draws ticks on both horizontal edges. Reserve that
        // worst case regardless of the initial divider and tick visibility.
        pimpl->computeTickCount(pimpl->maxVerticalLines,
                                pimpl->maxHorizontalLines,
                                majX, minx, majY, miny,
                                true, false);
        pimpl->maxMajorTicks = majX + majY;
        pimpl->maxMinorTicks = minx + miny;
    }

    pimpl->recomputeTickCount();

    const U64 currentInteriorLines = config.showInteriorGrid ? 2 : 0;
    const U64 maxInteriorLines = config.showInteriorGrid
        ? kMaxVerticalLines + kMaxHorizontalLines - 4
        : 0;
    U64 currentTicks = 0;
    U64 currentGridLines = 0;
    U64 maxTicks = 0;
    U64 maxGridLines = 0;
    U64 gridPointScalarCount = 0;
    U64 gridVertexScalarCount = 0;
    U64 gridVertexCount = 0;
    if (!detail::CheckedAdd(pimpl->currentMajorTicks,
                            pimpl->currentMinorTicks, currentTicks) ||
        !detail::CheckedAdd(currentInteriorLines, currentTicks,
                            currentGridLines) ||
        !detail::CheckedAdd(currentGridLines, pimpl->dividerLines + 4,
                            pimpl->totalLines) ||
        !detail::CheckedAdd(pimpl->maxMajorTicks, pimpl->maxMinorTicks,
                            maxTicks) ||
        !detail::CheckedAdd(maxInteriorLines, maxTicks, maxGridLines) ||
        !detail::CheckedAdd(maxGridLines, 5,
                            pimpl->maxTotalLines) ||
        !detail::CheckedMultiply(pimpl->maxTotalLines, 4,
                                 gridPointScalarCount) ||
        !detail::CheckedMultiply(pimpl->maxTotalLines, 24,
                                 gridVertexScalarCount) ||
        !detail::CheckedMultiply(pimpl->maxTotalLines, 6,
                                 gridVertexCount) ||
        pimpl->maxTotalLines > std::numeric_limits<U32>::max() ||
        gridVertexCount > std::numeric_limits<U32>::max() ||
        gridPointScalarCount > pimpl->gridPoints.max_size() ||
        gridVertexScalarCount > pimpl->gridVerticesData.max_size()) {
        JST_ERROR("[AXIS] Geometry exceeds the supported rendering range.");
        return Result::ERROR;
    }

    // 2 points per line, 2 coords per point.
    try {
        pimpl->gridPoints.resize(gridPointScalarCount, 0.0f);
        // 6 vertices per line, 4 floats per vertex.
        pimpl->gridVerticesData.resize(gridVertexScalarCount, 0.0f);
    } catch (const std::bad_alloc&) {
        JST_ERROR("[AXIS] Failed to allocate grid geometry.");
        return Result::ERROR;
    } catch (const std::length_error&) {
        JST_ERROR("[AXIS] Grid geometry exceeds container capacity.");
        return Result::ERROR;
    }

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
                .scale = kTickLabelScale,
                .position = {0.0f, 0.99f},
                .alignment = {1, 0},
            };
        }

        // Y tick labels (interior lines only, pre-allocate for max).
        for (U64 i = 1; i < maxH - 1; i++) {
            if (config.yLabelOnRight && config.yLabelOutside) {
                cfg.elements[jst::fmt::format("y{:02d}", i)] = {
                    .scale = kTickLabelScale,
                    .position = {0.99f, 0.0f},
                    .alignment = {0, 1},
                };
            } else if (config.yLabelOnRight) {
                cfg.elements[jst::fmt::format("y{:02d}", i)] = {
                    .scale = kTickLabelScale,
                    .position = {0.85f, 0.0f},
                    .alignment = {2, 1},
                };
            } else {
                cfg.elements[jst::fmt::format("y{:02d}", i)] = {
                    .scale = kTickLabelScale,
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
    pimpl->updateVertexCountFlag = true;

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
    // Draw commands are created when the owning surface is bound.
    if (pimpl->updateVertexCountFlag) {
        JST_CHECK(pimpl->drawGridVertex->updateVertexCount(pimpl->totalLines * 6));
        pimpl->updateVertexCountFlag = false;
    }

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
    JST_CHECK(pimpl->syncResponsiveGrid());

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

Result Axis::updateVerticalScale(F32 scale) {
    if (!std::isfinite(scale) || scale <= 0.0f || scale > 1.0f) {
        return Result::ERROR;
    }
    if (config.verticalScale == scale) {
        return Result::SUCCESS;
    }
    config.verticalScale = scale;
    pimpl->dividerLines = scale < 1.0f ? 1 : 0;
    JST_CHECK(pimpl->syncTickGeometry());
    JST_CHECK(pimpl->repositionLabels());
    return pimpl->syncResponsiveGrid();
}

Result Axis::setShowFrameTicks(bool visible) {
    if (visible == config.showFrameTicks) {
        return Result::SUCCESS;
    }

    config.showFrameTicks = visible;

    JST_CHECK(pimpl->syncTickGeometry());

    return Result::SUCCESS;
}

Result Axis::updateTickFormatters(TickFormatter xFormatter,
                                  TickFormatter yFormatter) {
    pimpl->xTickFormatter = std::move(xFormatter);
    pimpl->yTickFormatter = std::move(yFormatter);
    return pimpl->syncResponsiveGrid();
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

Result Axis::updateXRange(F64 left, F64 right) {
    if (!std::isfinite(left) || !std::isfinite(right) || !(right > left)) {
        return Result::ERROR;
    }
    pimpl->xRangeEnabled = true;
    pimpl->xRangeLeft = left;
    pimpl->xRangeRight = right;
    if (!pimpl->text) {
        return Result::SUCCESS;
    }
    return pimpl->syncResponsiveGrid();
}

Result Axis::clearXRange() {
    if (!pimpl->xRangeEnabled) {
        return Result::SUCCESS;
    }
    pimpl->xRangeEnabled = false;
    pimpl->xTicks = {};
    pimpl->xTickPositions.clear();
    pimpl->xMinorPositions.clear();
    if (!pimpl->text) {
        return Result::SUCCESS;
    }
    // Force the evenly divided grid to be regenerated.
    pimpl->currentVerticalLines = 0;
    return pimpl->syncResponsiveGrid();
}

Result Axis::updateYRange(F64 bottom, F64 top, PositionMap toPosition, F64 step) {
    if (!std::isfinite(bottom) || !std::isfinite(top) || !(top > bottom)) {
        return Result::ERROR;
    }
    pimpl->yRangeEnabled = true;
    pimpl->yRangeBottom = bottom;
    pimpl->yRangeTop = top;
    pimpl->yRangeMap = std::move(toPosition);
    pimpl->yRangeStep = (std::isfinite(step) && step > 0.0) ? step : 0.0;
    if (!pimpl->text) {
        return Result::SUCCESS;
    }
    return pimpl->syncResponsiveGrid();
}

Result Axis::clearYRange() {
    if (!pimpl->yRangeEnabled) {
        return Result::SUCCESS;
    }
    pimpl->yRangeEnabled = false;
    pimpl->yRangeMap = {};
    pimpl->yTicks = {};
    pimpl->yTickPositions.clear();
    pimpl->yLabelVisible.clear();
    pimpl->yMinorPositions.clear();
    if (!pimpl->text) {
        return Result::SUCCESS;
    }
    // Force the evenly divided grid to be regenerated.
    pimpl->currentHorizontalLines = 0;
    return pimpl->syncResponsiveGrid();
}

const Extent2D<F32>& Axis::paddingScale() const {
    return pimpl->padScale;
}

const Axis::TickScale& Axis::currentYTicks() const {
    return pimpl->yTicks;
}

const Axis::TickScale& Axis::currentXTicks() const {
    return pimpl->xTicks;
}

Axis::TickScale Axis::ComputeTicks(const F64 left,
                                   const F64 right,
                                   const F64 minimumStep) {
    TickScale scale;
    const F64 span = right - left;
    if (!std::isfinite(left) || !std::isfinite(right) || !(span > 0.0) ||
        !std::isfinite(span)) {
        return scale;
    }

    const F64 minimum = (std::isfinite(minimumStep) && minimumStep > 0.0)
        ? minimumStep
        : span / 10.0;
    const F64 base = std::pow(10.0, std::floor(std::log10(minimum)));
    U64 mantissa = 10;
    for (const U64 candidate : {1, 2, 5}) {
        // Loose enough to absorb single-precision pixel sizes, so a minimum
        // of 10.000001 still picks 10.
        if (static_cast<F64>(candidate) * base >= minimum * (1.0 - 1e-6)) {
            mantissa = candidate;
            break;
        }
    }
    scale.step = static_cast<F64>(mantissa) * base;
    scale.minorDivisions = (mantissa == 2) ? 4 : 5;
    scale.decimals = static_cast<U64>(std::max(
        0.0, -std::floor(std::log10(scale.step) + 1e-9)));

    // Skip ticks that would sit on the frame itself.
    const F64 margin = span * 1e-6;
    const F64 first = std::ceil((left + margin) / scale.step);
    const F64 last = std::floor((right - margin) / scale.step);
    constexpr U64 kMaxValues = 1024;
    for (F64 k = first; k <= last && scale.values.size() < kMaxValues; k += 1.0) {
        scale.values.push_back(k * scale.step);
    }

    return scale;
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
                                  const bool showFrameTicks,
                                  const bool combined) {
    outMajX = outMinX = outMajY = outMinY = 0;
    if (!showFrameTicks) {
        return;
    }
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
        outMinY = 2 * 4 * (numRows - 1);
    }
}

void Axis::Impl::recomputeTickCount() {
    U64 majX = 0, minx = 0, majY = 0, miny = 0;
    computeTickCount(currentVerticalLines, currentHorizontalLines,
                     majX, minx, majY, miny,
                     config.showFrameTicks, dividerLines > 0);
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
    updateVertexCountFlag = true;

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
        const F32 y = yLinePosition(row, numRows);
        gridPoints[(idx * stride) + 0] = xStart;
        gridPoints[(idx * stride) + 1] = y;
        gridPoints[(idx * stride) + 2] = xEnd;
        gridPoints[(idx * stride) + 3] = y;
        idx++;
    }

    for (U64 col = 1; col <= interiorV; col++) {
        const F32 x = xLinePosition(col, numCols);
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
                majX.push_back(xLinePosition(col, numCols));
            }
        }
        if (numRows >= 3) {
            majY.reserve(numRows);
            for (U64 row = 1; row <= numRows - 2; row++) {
                majY.push_back(yLinePosition(row, numRows));
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

        if (xRangeEnabled) {
            // Minor ticks on round values too. Unused reserved slots stay
            // zero-length lines so the tick counts seen by the shader hold.
            const U64 reserved = kMinorTicksPerInterval * (numCols - 1);
            const U64 used = std::min<U64>(xMinorPositions.size(), reserved);
            for (U64 i = 0; i < used; i++) {
                emitMinorX(xMinorPositions[i]);
            }
            idx += (reserved - used) * (dividerLines == 0 ? 2 : 1);
        } else {
            for (U64 col = 0; col + 1 < numCols; col++) {
                const F32 a = xStart + col * xStep;
                const F32 b = a + xStep;
                for (U64 s = 1; s <= kMinorTicksPerInterval; s++) {
                    emitMinorX(a + (b - a) *
                        (static_cast<F32>(s) / 5.0f));
                }
            }
        }

        auto emitMinorY = [&](F32 y) {
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
        };

        if (yRangeEnabled) {
            const U64 reserved = kMinorTicksPerInterval * (numRows - 1);
            const U64 used = std::min<U64>(yMinorPositions.size(), reserved);
            for (U64 i = 0; i < used; i++) {
                emitMinorY(yMinorPositions[i]);
            }
            idx += (reserved - used) * 2;
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

        for (size_t i = 0; !yRangeEnabled && i + 1 < minorYBounds.size(); i++) {
            if (dividerLines > 0 && i == 0) {
                continue;
            }
            const F32 a = minorYBounds[i];
            const F32 b = minorYBounds[i + 1];
            for (U64 s = 1; s <= kMinorTicksPerInterval; s++) {
                emitMinorY(a + (b - a) * (static_cast<F32>(s) / 5.0f));
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
        const F32 x = xLinePosition(i, numCols) * padScale.x;
        element.position = {x, 1.0f - ps.y * 5.0f};
        JST_CHECK(text->update(id, element));
    }

    // Y tick labels along the configured edge.
    const U64 numRows = currentHorizontalLines;
    const F32 sideTickOffset = config.majorTickLengthPx + 4.0f;
    for (U64 i = 1; i < numRows - 1; i++) {
        const auto id = jst::fmt::format("y{:02d}", i);
        auto element = text->get(id);
        const F32 localY = yLinePosition(i, numRows);
        const F32 y = mapGridY(localY);
        if (config.yLabelOnRight && config.yLabelOutside) {
            element.position = {padScale.x + ps.x * 4.0f, y};
            element.alignment = {0, 1};
        } else if (config.yLabelOnRight) {
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
        const F32 baseX = xLinePosition(i, numCols);

        // Apply the same pan/zoom that interior grid lines get.
        const F32 x = (baseX * zoom + translation * zoom) * padScale.x;

        element.position = {x, 1.0f - ps.y * 5.0f};
        JST_CHECK(text->update(id, element));
    }

    return Result::SUCCESS;
}

F32 Axis::Impl::xLinePosition(const U64 i, const U64 numCols) const {
    if (xRangeEnabled && i >= 1 && i <= xTickPositions.size()) {
        return xTickPositions[i - 1];
    }
    return (2.0f / static_cast<F32>(numCols - 1)) * static_cast<F32>(i) - 1.0f;
}

F32 Axis::Impl::xRangePosition(const F64 value) const {
    return static_cast<F32>(-1.0 + 2.0 * (value - xRangeLeft) /
                                   (xRangeRight - xRangeLeft));
}

F32 Axis::Impl::yLinePosition(const U64 i, const U64 numRows) const {
    if (yRangeEnabled && i >= 1 && i <= yTickPositions.size()) {
        return yTickPositions[i - 1];
    }
    return bandLocalY(i, numRows);
}

F32 Axis::Impl::yRangePosition(const F64 value) const {
    F32 position = yRangeMap
        ? yRangeMap(value)
        : static_cast<F32>(-1.0 + 2.0 * (value - yRangeBottom) /
                                  (yRangeTop - yRangeBottom));
    if (!std::isfinite(position)) {
        position = value > yRangeBottom ? 1.0f : -1.0f;
    }
    // From the plot band's -1..1 to the grid's local coordinates.
    const F32 bottom = bandLocalY(0, 2);
    const F32 top = bandLocalY(1, 2);
    return bottom + (top - bottom) * (std::clamp(position, -1.0f, 1.0f) + 1.0f) * 0.5f;
}

void Axis::Impl::computeYRangeTicks(const F32 availableHeightPx,
                                    const F32 minimumSpacingPx,
                                    const F32 collisionPadding) {
    const F64 heightPx = std::max<F64>(availableHeightPx, 1.0);
    const F64 span = yRangeTop - yRangeBottom;
    const F64 labelPx = static_cast<F64>(config.font->lineHeight()) * kTickLabelScale;
    // Lines follow the requested spacing; labels only need to clear each
    // other, so a dense grid can carry a label on every other line.
    constexpr F64 kMinimumLinePx = 12.0;
    const F64 neededPx = std::max<F64>(minimumSpacingPx, kMinimumLinePx);
    const F64 labelSpacingPx = labelPx + collisionPadding;
    const U64 maxTicks = maxHorizontalLines - 2;
    const F32 bandBottom = bandLocalY(0, 2);
    const F32 bandTop = bandLocalY(1, 2);
    const auto toPx = [&](const F64 value) {
        return static_cast<F64>(yRangePosition(value) - bandBottom) /
               std::max<F64>(bandTop - bandBottom, 1e-6) * heightPx;
    };

    // Pick the step where the mapping is most stretched, then drop labels
    // that would collide where it is compressed.
    F64 pxPerUnit = 0.0;
    constexpr U64 kSamples = 64;
    for (U64 i = 0; i < kSamples; i++) {
        const F64 a = yRangeBottom + span * static_cast<F64>(i) / kSamples;
        const F64 b = yRangeBottom + span * static_cast<F64>(i + 1) / kSamples;
        pxPerUnit = std::max(pxPerUnit, std::abs(toPx(b) - toPx(a)) / (b - a));
    }
    pxPerUnit = std::max(pxPerUnit, 1e-12);

    // A requested step wins; it only widens if its lines would merge.
    F64 minimumStep = yRangeStep > 0.0
        ? std::max(yRangeStep, kMinimumLinePx / pxPerUnit)
        : neededPx / pxPerUnit;
    std::vector<F64> kept;
    TickScale ticks;
    for (U64 attempt = 0; attempt < 8; attempt++) {
        ticks = ComputeTicks(yRangeBottom, yRangeTop, minimumStep);
        kept.clear();
        F64 lastPx = -std::numeric_limits<F64>::infinity();
        for (const F64 value : ticks.values) {
            const F64 px = toPx(value);
            const F64 lineSpacingPx = yRangeStep > 0.0 ? kMinimumLinePx : neededPx;
            if (px - lastPx >= lineSpacingPx * (1.0 - 1e-6)) {
                kept.push_back(value);
                lastPx = px;
            }
        }
        if (kept.size() <= maxTicks) {
            break;
        }
        minimumStep = ticks.step * 1.01;
    }
    if (kept.size() > maxTicks) {
        kept.resize(maxTicks);
    }

    yTicks = ticks;
    yTicks.values = kept;
    yTickPositions.clear();
    yLabelVisible.clear();
    F64 lastLabelPx = -std::numeric_limits<F64>::infinity();
    for (const F64 value : kept) {
        yTickPositions.push_back(yRangePosition(value));
        const F64 px = toPx(value);
        const bool visible = px >= labelPx * 0.5 && px <= heightPx - labelPx * 0.5 &&
                             px - lastLabelPx >= labelSpacingPx * (1.0 - 1e-6);
        if (visible) {
            lastLabelPx = px;
        }
        yLabelVisible.push_back(visible);
    }

    // Minor ticks on round sub-steps, wherever they stay a few pixels apart.
    yMinorPositions.clear();
    if (yTicks.step > 0.0) {
        const F64 minor = yTicks.step / static_cast<F64>(yTicks.minorDivisions);
        const F64 first = std::ceil((yRangeBottom + span * 1e-6) / minor);
        const F64 last = std::floor((yRangeTop - span * 1e-6) / minor);
        const U64 reserved = kMinorTicksPerInterval * (kept.size() + 1);
        for (F64 k = first; k <= last && yMinorPositions.size() < reserved; k += 1.0) {
            if (std::llround(k) % static_cast<I64>(yTicks.minorDivisions) == 0) {
                continue;
            }
            const F64 value = k * minor;
            if (std::abs(toPx(value + minor) - toPx(value)) < 4.0) {
                continue;
            }
            yMinorPositions.push_back(yRangePosition(value));
        }
    }
}

void Axis::Impl::computeXRangeTicks(const F32 availableWidthPx,
                                    const F32 minimumSpacingPx,
                                    const F32 collisionPadding) {
    const F64 span = xRangeRight - xRangeLeft;
    const F64 widthPx = std::max<F64>(availableWidthPx, 1.0);
    const F64 unitsPerPx = span / widthPx;
    const U64 maxTicks = maxVerticalLines - 2;

    // Widen the step until the labels fit; the label width depends on the
    // number of decimals, which depends on the step.
    F64 spacingPx = std::max<F64>(minimumSpacingPx, 1.0);
    TickScale ticks;
    for (U64 attempt = 0; attempt < 4; attempt++) {
        ticks = ComputeTicks(xRangeLeft, xRangeRight, spacingPx * unitsPerPx);
        F32 labelWidth = 0.0f;
        for (const F64 value : {xRangeLeft, xRangeRight}) {
            labelWidth = std::max(labelWidth,
                text->advance(FormatTickValue(value, ticks), kTickLabelScale));
        }
        const F64 stepPx = ticks.step / unitsPerPx;
        F64 neededPx = std::max<F64>(spacingPx, labelWidth + collisionPadding);
        if (ticks.values.size() > maxTicks) {
            neededPx = std::max(neededPx, widthPx / static_cast<F64>(maxTicks));
        }
        if (stepPx >= neededPx * (1.0 - 1e-9) && ticks.values.size() <= maxTicks) {
            break;
        }
        spacingPx = std::max(neededPx, stepPx * 1.01);
    }
    if (ticks.values.size() > maxTicks) {
        ticks.values.resize(maxTicks);
    }

    xTicks = ticks;
    xTickPositions.clear();
    for (const F64 value : xTicks.values) {
        xTickPositions.push_back(xRangePosition(value));
    }

    xMinorPositions.clear();
    if (xTicks.step > 0.0) {
        const F64 minor = xTicks.step / static_cast<F64>(xTicks.minorDivisions);
        const F64 margin = span * 1e-6;
        const F64 first = std::ceil((xRangeLeft + margin) / minor);
        const F64 last = std::floor((xRangeRight - margin) / minor);
        const U64 reserved = kMinorTicksPerInterval * (xTicks.values.size() + 1);
        for (F64 k = first; k <= last && xMinorPositions.size() < reserved; k += 1.0) {
            if (std::llround(k) % static_cast<I64>(xTicks.minorDivisions) == 0) {
                continue;
            }
            xMinorPositions.push_back(xRangePosition(k * minor));
        }
    }
}

U64 Axis::Impl::responsiveLineCount(const F32 availablePx,
                                    const F32 requiredSpacingPx,
                                    const U64 maximum) const {
    if (!std::isfinite(availablePx) || availablePx <= 0.0f) {
        return 3;
    }
    const F32 spacing = std::max(requiredSpacingPx, 1.0f);
    U64 count = std::clamp<U64>(
        static_cast<U64>(std::floor(availablePx / spacing)) + 1,
        3,
        maximum);
    if (count % 2 == 0) {
        --count;
    }
    return count;
}

Result Axis::Impl::syncResponsiveGrid() {
    const F32 availableWidthPx = config.pixelSize.x > 0.0f
        ? 2.0f * std::max(padScale.x, 0.0f) / config.pixelSize.x
        : 0.0f;
    const F32 availableHeightPx = config.pixelSize.y > 0.0f
        ? 2.0f * std::max(padScale.y, 0.0f) * config.verticalScale /
              config.pixelSize.y
        : 0.0f;
    F32 maxXLabelWidth = 0.0f;
    if (xTickFormatter) {
        for (const F32 position : {-1.0f, 0.0f, 1.0f}) {
            maxXLabelWidth = std::max(
                maxXLabelWidth,
                text->advance(xTickFormatter(position), kTickLabelScale));
        }
    }
    const F32 collisionPadding =
        std::max(config.labelCollisionPaddingPx, 0.0f);
    const F32 responsiveXSpacing = std::min(
        std::max(config.minXLabelSpacingPx, 0.0f),
        availableWidthPx / 4.0f);
    const F32 responsiveYSpacing = std::min(
        std::max(config.minYLabelSpacingPx, 0.0f),
        availableHeightPx / 4.0f);
    const F32 xRequiredSpacing = std::max(
        responsiveXSpacing,
        maxXLabelWidth + collisionPadding);
    const F32 yRequiredSpacing = std::max(
        responsiveYSpacing,
        static_cast<F32>(config.font->lineHeight()) * kTickLabelScale +
            collisionPadding);

    U64 verticalLines = 0;
    bool xTicksMoved = false;
    if (xRangeEnabled) {
        const auto previousPositions = xTickPositions;
        const auto previousMinors = xMinorPositions;
        computeXRangeTicks(availableWidthPx, responsiveXSpacing,
                           collisionPadding);
        verticalLines = xTickPositions.size() + 2;
        xTicksMoved = previousPositions != xTickPositions ||
                      previousMinors != xMinorPositions;
    } else {
        verticalLines = responsiveLineCount(
            availableWidthPx, xRequiredSpacing, maxVerticalLines);
    }
    U64 horizontalLines = 0;
    bool yTicksMoved = false;
    if (yRangeEnabled) {
        const auto previousPositions = yTickPositions;
        const auto previousMinors = yMinorPositions;
        // Aim for about ten divisions, like a bench analyzer's graticule.
        computeYRangeTicks(availableHeightPx,
                           std::min(responsiveYSpacing, availableHeightPx / 10.0f),
                           collisionPadding);
        horizontalLines = yTickPositions.size() + 2;
        yTicksMoved = previousPositions != yTickPositions ||
                      previousMinors != yMinorPositions;
    } else {
        horizontalLines = responsiveLineCount(
            availableHeightPx, yRequiredSpacing, maxHorizontalLines);
    }

    if (verticalLines != currentVerticalLines ||
        horizontalLines != currentHorizontalLines ||
        xTicksMoved || yTicksMoved) {
        currentVerticalLines = verticalLines;
        currentHorizontalLines = horizontalLines;
        const U64 interiorLines = config.showInteriorGrid
            ? verticalLines + horizontalLines - 4
            : 0;
        recomputeTickCount();
        totalLines = interiorLines + currentMajorTicks + currentMinorTicks +
                     dividerLines + 4;
        generateGridPoints();
        updateVertexCountFlag = true;
        gridUniforms.lineInfo = {
            static_cast<float>(config.showInteriorGrid
                ? horizontalLines - 2 : 0),
            static_cast<float>(config.showInteriorGrid
                ? verticalLines - 2 : 0),
            static_cast<float>(currentMajorTicks),
            static_cast<float>(currentMinorTicks),
        };
        updateGridUniformsFlag = true;
        JST_CHECK(repositionLabels());
        JST_CHECK(repositionXLabels());
    }

    return syncTickLabels();
}

Result Axis::Impl::syncTickLabels() {
    const U64 xCount = currentVerticalLines - 2;
    const U64 yCount = currentHorizontalLines - 2;

    for (U64 i = 0; i < maxVerticalLines - 2; ++i) {
        const auto id = jst::fmt::format("x{:02d}", i + 1);
        auto element = text->get(id);
        if (i < xCount && xRangeEnabled) {
            element.fill = FormatTickValue(xTicks.values[i], xTicks);
        } else if (i < xCount && xTickFormatter) {
            const F32 position =
                (2.0f * static_cast<F32>(i + 1) /
                 static_cast<F32>(currentVerticalLines - 1)) - 1.0f;
            element.fill = xTickFormatter(position);
        } else {
            element.fill = " ";
        }
        JST_CHECK(text->update(id, element));
    }

    for (U64 i = 0; i < maxHorizontalLines - 2; ++i) {
        const auto id = jst::fmt::format("y{:02d}", i + 1);
        auto element = text->get(id);
        if (i < yCount && yRangeEnabled) {
            element.fill = yLabelVisible[i]
                ? FormatTickValue(yTicks.values[i], yTicks)
                : " ";
        } else if (i < yCount && yTickFormatter) {
            const F32 position =
                (2.0f * static_cast<F32>(i + 1) /
                 static_cast<F32>(currentHorizontalLines - 1)) - 1.0f;
            element.fill = yTickFormatter(position);
        } else {
            element.fill = " ";
        }
        JST_CHECK(text->update(id, element));
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render::Components
