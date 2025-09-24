#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "jetstream/backend/devices/cpu/helpers.hh"
#include "jetstream/modules/lineplot.hh"
#include "jetstream/render/utils.hh"

#include "resources/shaders/lineplot_shaders.hh"
#include "resources/shaders/global_shaders.hh"
#include "jetstream/constants.hh"

#include "benchmark.cc"

namespace Jetstream {

template<Device D, typename T>
struct Lineplot<D, T>::GImpl {
    struct {
        glm::mat4 transform;
        F32 thickness[2];
        F32 zoom;
        U32 numberOfLines;
    } gridUniforms;

    struct {
        glm::mat4 transform;
        F32 thickness[2];
        F32 zoom;
        U32 numberOfPoints;
    } signalUniforms;

    struct {
        glm::mat4 transform;
    } cursorUniforms;

    Extent2D<F32> pixelSize;
    Extent2D<F32> paddingScale;

    Tensor<D, F32> signalPoints;
    Tensor<D, F32> signalVertices;
    Tensor<Device::CPU, F32> gridPoints;
    Tensor<Device::CPU, F32> cursorSignalPoint;
    Tensor<D, F32> gridVertices;

    std::shared_ptr<Render::Buffer> signalPointsBuffer;
    std::shared_ptr<Render::Buffer> signalVerticesBuffer;
    std::shared_ptr<Render::Buffer> signalUniformBuffer;
    std::shared_ptr<Render::Buffer> gridPointsBuffer;
    std::shared_ptr<Render::Buffer> gridVerticesBuffer;
    std::shared_ptr<Render::Buffer> gridUniformBuffer;
    std::shared_ptr<Render::Buffer> cursorVerticesBuffer;
    std::shared_ptr<Render::Buffer> cursorIndicesBuffer;
    std::shared_ptr<Render::Buffer> cursorUniformBuffer;

    std::shared_ptr<Render::Texture> framebufferTexture;
    std::shared_ptr<Render::Texture> lutTexture;

    std::shared_ptr<Render::Kernel> gridKernel;
    std::shared_ptr<Render::Kernel> signalKernel;

    std::shared_ptr<Render::Program> signalProgram;
    std::shared_ptr<Render::Program> gridProgram;
    std::shared_ptr<Render::Program> cursorProgram;

    std::shared_ptr<Render::Surface> surface;

    std::shared_ptr<Render::Vertex> gridVertex;
    std::shared_ptr<Render::Vertex> signalVertex;
    std::shared_ptr<Render::Vertex> cursorVertex;

    std::shared_ptr<Render::Draw> drawGridVertex;
    std::shared_ptr<Render::Draw> drawSignalVertex;
    std::shared_ptr<Render::Draw> drawCursorVertex;

    std::shared_ptr<Render::Components::Text> text;

    U64 numberOfElements = 0;
    U64 numberOfBatches = 0;
    F32 normalizationFactor = 0.0f;

    Extent2D<F32> cursorPos = {0.0f, 0.0f};

    bool updateGridPointsFlag = false;
    bool updateSignalPointsFlag = false;
    bool updateCursorUniformBufferFlag = false;
    bool updateSignalUniformBufferFlag = false;
    bool updateGridUniformBufferFlag = false;

    void updateState(Lineplot<D, T>& m);
    void updateCursorState(Lineplot<D, T>& m);
    void generateGridPoints(Lineplot<D, T>& m);
};

template<Device D, typename T>
Result Lineplot<D, T>::create() {
    JST_DEBUG("Initializing Lineplot module.");
    JST_INIT_IO();

    // Check parameters.

    if (input.buffer.rank() > 2) {
        JST_ERROR("Invalid input rank ({}). It should be `1` or `2`.", input.buffer.rank());
        return Result::ERROR;
    }

    // Calculate parameters.

    const U64 last_axis = input.buffer.rank() - 1;
    gimpl->numberOfElements = input.buffer.shape()[last_axis] / config.decimation;
    gimpl->numberOfBatches = (input.buffer.rank() == 2) ? input.buffer.shape()[0] : 1;
    gimpl->normalizationFactor = 1.0f / (0.5f * gimpl->numberOfBatches);

    // Check shape.

    if (gimpl->numberOfElements < 2) {
        JST_ERROR("Invalid number of elements ({}). It should be at least '2'.", gimpl->numberOfElements);
        return Result::ERROR;
    }

    // Allocate internal buffers.

    JST_CHECK(gimpl->signalPoints.create(D, mem2::TypeToDataType<F32>(), {gimpl->numberOfElements, 2}));
    JST_CHECK(gimpl->signalVertices.create(D, mem2::TypeToDataType<F32>(), {gimpl->numberOfElements - 1, 4, 4}));

    JST_CHECK(gimpl->gridPoints.create(Device::CPU, mem2::TypeToDataType<F32>(), {config.numberOfVerticalLines + config.numberOfHorizontalLines, 2, 2}));
    JST_CHECK(gimpl->gridVertices.create(D, mem2::TypeToDataType<F32>(), {config.numberOfVerticalLines + config.numberOfHorizontalLines, 6, 4}));

    JST_CHECK(gimpl->cursorSignalPoint.create(Device::CPU, mem2::TypeToDataType<F32>(), {2}));

    return Result::SUCCESS;
}

template<Device D, typename T>
void Lineplot<D, T>::info() const {
    JST_DEBUG("  Averaging: {}", config.averaging);
    JST_DEBUG("  Decimation: {}", config.decimation);
    JST_DEBUG("  Number of Vertical Lines: {}", config.numberOfVerticalLines);
    JST_DEBUG("  Number of Horizontal Lines: {}", config.numberOfHorizontalLines);
    JST_DEBUG("  Size: [{}, {}]", config.viewSize.x, config.viewSize.y);
    JST_DEBUG("  Zoom: {}", config.zoom);
    JST_DEBUG("  Translation: {}", config.translation);
    JST_DEBUG("  Scale: {}", config.scale);
    JST_DEBUG("  Thickness: {}", config.thickness);
}

template<Device D, typename T>
Result Lineplot<D, T>::createPresent() {
    // Grid element.

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &gimpl->gridUniforms;
        cfg.elementByteSize = sizeof(gimpl->gridUniforms);
        cfg.size = 1;
        cfg.target = Render::Buffer::Target::UNIFORM;
        JST_CHECK(window->build(gimpl->gridUniformBuffer, cfg));
        JST_CHECK(window->bind(gimpl->gridUniformBuffer));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = gimpl->gridPoints.data();
        cfg.elementByteSize = sizeof(F32);
        cfg.size = gimpl->gridPoints.size();
        cfg.target = Render::Buffer::Target::STORAGE;
        cfg.enableZeroCopy = false;
        JST_CHECK(window->build(gimpl->gridPointsBuffer, cfg));
        JST_CHECK(window->bind(gimpl->gridPointsBuffer));
    }

    {
        auto [buffer, enableZeroCopy] = ConvertToOptimalStorage(window, gimpl->gridVertices);

        Render::Buffer::Config cfg;
        cfg.buffer = buffer;
        cfg.elementByteSize = sizeof(F32);
        cfg.size = gimpl->gridVertices.size();
        cfg.target = Render::Buffer::Target::VERTEX | Render::Buffer::Target::STORAGE;
        cfg.enableZeroCopy = enableZeroCopy;
        JST_CHECK(window->build(gimpl->gridVerticesBuffer, cfg));
        JST_CHECK(window->bind(gimpl->gridVerticesBuffer));
    }

    {
        Render::Kernel::Config cfg;
        cfg.gridSize = {config.numberOfVerticalLines + config.numberOfHorizontalLines, 1, 1};
        cfg.kernels = GlobalKernelsPackage["thicklines"];
        cfg.buffers = {
            {gimpl->gridUniformBuffer, Render::Kernel::AccessMode::READ},
            {gimpl->gridPointsBuffer, Render::Kernel::AccessMode::READ},
            {gimpl->gridVerticesBuffer, Render::Kernel::AccessMode::WRITE},
        };
        JST_CHECK(window->build(gimpl->gridKernel, cfg));
    }

    {
        Render::Vertex::Config cfg;
        cfg.vertices = {
            {gimpl->gridVerticesBuffer, 4},
        };
        JST_CHECK(window->build(gimpl->gridVertex, cfg));
    }

    {
        Render::Draw::Config cfg;
        cfg.buffer = gimpl->gridVertex;
        cfg.mode = Render::Draw::Mode::TRIANGLES;
        JST_CHECK(window->build(gimpl->drawGridVertex, cfg));
    }

    {
        Render::Program::Config cfg;
        cfg.shaders = ShadersPackage["grid"];
        cfg.draws = {
            gimpl->drawGridVertex,
        };
        cfg.buffers = {
            {gimpl->gridUniformBuffer, Render::Program::Target::VERTEX |
                                Render::Program::Target::FRAGMENT},
        };
        cfg.enableAlphaBlending = true;
        JST_CHECK(window->build(gimpl->gridProgram, cfg));
    }

    // Cursor element.

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &gimpl->cursorUniforms;
        cfg.elementByteSize = sizeof(gimpl->cursorUniforms);
        cfg.size = 1;
        cfg.target = Render::Buffer::Target::UNIFORM;
        JST_CHECK(window->build(gimpl->cursorUniformBuffer, cfg));
        JST_CHECK(window->bind(gimpl->cursorUniformBuffer));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &FillScreenVertices;
        cfg.elementByteSize = sizeof(F32);
        cfg.size = 12;
        cfg.target = Render::Buffer::Target::VERTEX;
        JST_CHECK(window->build(gimpl->cursorVerticesBuffer, cfg));
        JST_CHECK(window->bind(gimpl->cursorVerticesBuffer));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &FillScreenIndices;
        cfg.elementByteSize = sizeof(U32);
        cfg.size = 6;
        cfg.target = Render::Buffer::Target::VERTEX_INDICES;
        JST_CHECK(window->build(gimpl->cursorIndicesBuffer, cfg));
        JST_CHECK(window->bind(gimpl->cursorIndicesBuffer));
    }

    {
        Render::Vertex::Config cfg;
        cfg.vertices = {
            {gimpl->cursorVerticesBuffer, 3},
        };
        cfg.indices = gimpl->cursorIndicesBuffer;
        JST_CHECK(window->build(gimpl->cursorVertex, cfg));
    }

    {
        Render::Draw::Config cfg;
        cfg.buffer = gimpl->cursorVertex;
        cfg.mode = Render::Draw::Mode::TRIANGLES;
        JST_CHECK(window->build(gimpl->drawCursorVertex, cfg));
    }

    {
        Render::Program::Config cfg;
        cfg.shaders = ShadersPackage["cursor"];
        cfg.draws = {
            gimpl->drawCursorVertex,
        };
        cfg.buffers = {
            {gimpl->cursorUniformBuffer, Render::Program::Target::VERTEX |
                                  Render::Program::Target::FRAGMENT},
        };
        cfg.enableAlphaBlending = true;
        JST_CHECK(window->build(gimpl->cursorProgram, cfg));
    }

    // Signal element.

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &gimpl->signalUniforms;
        cfg.elementByteSize = sizeof(gimpl->signalUniforms);
        cfg.size = 1;
        cfg.target = Render::Buffer::Target::UNIFORM;
        JST_CHECK(window->build(gimpl->signalUniformBuffer, cfg));
        JST_CHECK(window->bind(gimpl->signalUniformBuffer));
    }

    {
        auto [buffer, enableZeroCopy] = ConvertToOptimalStorage(window, gimpl->signalPoints);

        Render::Buffer::Config cfg;
        cfg.buffer = buffer;
        cfg.elementByteSize = sizeof(F32);
        cfg.size = gimpl->signalPoints.size();
        cfg.target = Render::Buffer::Target::STORAGE;
        cfg.enableZeroCopy = enableZeroCopy;
        JST_CHECK(window->build(gimpl->signalPointsBuffer, cfg));
        JST_CHECK(window->bind(gimpl->signalPointsBuffer));
    }

    {
        auto [buffer, enableZeroCopy] = ConvertToOptimalStorage(window, gimpl->signalVertices);

        Render::Buffer::Config cfg;
        cfg.buffer = buffer;
        cfg.elementByteSize = sizeof(F32);
        cfg.size = gimpl->signalVertices.size();
        cfg.target = Render::Buffer::Target::VERTEX | Render::Buffer::Target::STORAGE;
        cfg.enableZeroCopy = enableZeroCopy;
        JST_CHECK(window->build(gimpl->signalVerticesBuffer, cfg));
        JST_CHECK(window->bind(gimpl->signalVerticesBuffer));
    }

    {
        Render::Kernel::Config cfg;
        cfg.gridSize = {gimpl->numberOfElements - 1, 1, 1};
        cfg.kernels = GlobalKernelsPackage["thicklinestrip"];
        cfg.buffers = {
            {gimpl->signalUniformBuffer, Render::Kernel::AccessMode::READ},
            {gimpl->signalPointsBuffer, Render::Kernel::AccessMode::READ},
            {gimpl->signalVerticesBuffer, Render::Kernel::AccessMode::WRITE},
        };
        JST_CHECK(window->build(gimpl->signalKernel, cfg));
    }

    {
        Render::Vertex::Config cfg;
        cfg.vertices = {
            {gimpl->signalVerticesBuffer, 4},
        };
        JST_CHECK(window->build(gimpl->signalVertex, cfg));
    }

    {
        Render::Draw::Config cfg;
        cfg.buffer = gimpl->signalVertex;
        cfg.mode = Render::Draw::Mode::TRIANGLE_STRIP;
        JST_CHECK(window->build(gimpl->drawSignalVertex, cfg));
    }

    {
        Render::Texture::Config cfg;
        cfg.size = {256, 1};
        cfg.buffer = (uint8_t*)TurboLutBytes;
        JST_CHECK(window->build(gimpl->lutTexture, cfg));
        JST_CHECK(window->bind(gimpl->lutTexture));
    }

    {
        Render::Program::Config cfg;
        cfg.shaders = ShadersPackage["signal"];
        cfg.draws = {
            gimpl->drawSignalVertex,
        };
        cfg.textures = {gimpl->lutTexture};
        cfg.buffers = {
            {gimpl->signalUniformBuffer, Render::Program::Target::VERTEX |
                                         Render::Program::Target::FRAGMENT},
        };
        cfg.enableAlphaBlending = true;
        JST_CHECK(window->build(gimpl->signalProgram, cfg));
    }

    // Font element.

    if (!window->hasFont("default_mono")) {
        JST_ERROR("Font 'default_mono' not found.");
        return Result::ERROR;
    }

    {
        Render::Components::Text::Config cfg;
        cfg.maxCharacters = 128;
        cfg.color = {1.0f, 1.0f, 1.0f, 1.0f};
        cfg.font = window->font("default_mono");
        cfg.elements = {
            {"amplitude", {1.0f, {1.0f, 1.0f}, {0, 0}, 0.0f, ""}},
            {"axis-x", {0.85f, {0.0f, -0.99f}, {1, 2}, 0.0f, "Frequency (MHz)"}},
            {"axis-y", {0.85f, {-0.99f, 0.0f}, {1, 0}, 90.0f, "Amplitude (dBFS)"}},
        };
        for (U64 i = 1; i < config.numberOfVerticalLines - 1; i++) {
            cfg.elements[jst::fmt::format("x{:02d}", i)] = {0.85f, {0.0f, 0.99f}, {1, 0}, 0.0f, "2.4G"};
        }
        JST_CHECK(window->build(gimpl->text, cfg));
        JST_CHECK(window->bind(gimpl->text));
    }

    // Surface.

    {
        Render::Texture::Config cfg;
        cfg.size = config.viewSize * config.scale;
        JST_CHECK(window->build(gimpl->framebufferTexture, cfg));
    }

    {
        Render::Surface::Config cfg;
        cfg.framebuffer = gimpl->framebufferTexture;
        cfg.kernels = {
            gimpl->gridKernel,
            gimpl->signalKernel
        };
        cfg.programs = {
            gimpl->gridProgram,
            gimpl->signalProgram,
            gimpl->cursorProgram,
        };
        cfg.multisampled = true;
        cfg.clearColor = {0.1f, 0.1f, 0.1f, 1.0f};
        JST_CHECK(gimpl->text->surface(cfg));
        JST_CHECK(window->build(gimpl->surface, cfg));
        JST_CHECK(window->bind(gimpl->surface));
    }

    // Generate resources.

    gimpl->generateGridPoints(*this);

    // Initialize variables.

    gimpl->updateState(*this);

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Lineplot<D, T>::destroyPresent() {
    JST_CHECK(window->unbind(gimpl->surface));
    JST_CHECK(window->unbind(gimpl->text));
    JST_CHECK(window->unbind(gimpl->lutTexture));
    JST_CHECK(window->unbind(gimpl->signalPointsBuffer));
    JST_CHECK(window->unbind(gimpl->signalVerticesBuffer));
    JST_CHECK(window->unbind(gimpl->signalUniformBuffer));
    JST_CHECK(window->unbind(gimpl->gridPointsBuffer));
    JST_CHECK(window->unbind(gimpl->gridVerticesBuffer));
    JST_CHECK(window->unbind(gimpl->gridUniformBuffer));
    JST_CHECK(window->unbind(gimpl->cursorUniformBuffer));
    JST_CHECK(window->unbind(gimpl->cursorVerticesBuffer));
    JST_CHECK(window->unbind(gimpl->cursorIndicesBuffer));

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Lineplot<D, T>::present() {
    if (gimpl->updateGridPointsFlag) {
        gimpl->gridPointsBuffer->update();
        gimpl->gridKernel->update();
        gimpl->updateCursorState(*this);
        gimpl->updateGridPointsFlag = false;
    }

    if (gimpl->updateSignalPointsFlag) {
        gimpl->signalPointsBuffer->update();
        gimpl->signalKernel->update();
        gimpl->updateCursorState(*this);
        gimpl->updateSignalPointsFlag = false;
    }

    if (gimpl->updateGridUniformBufferFlag) {
        gimpl->gridUniformBuffer->update();
        gimpl->gridKernel->update();
        gimpl->updateGridUniformBufferFlag = false;
    }

    if (gimpl->updateSignalUniformBufferFlag) {
        gimpl->signalUniformBuffer->update();
        gimpl->signalKernel->update();
        gimpl->updateSignalUniformBufferFlag = false;
    }

    if (gimpl->updateCursorUniformBufferFlag) {
        gimpl->cursorUniformBuffer->update();
        gimpl->updateCursorUniformBufferFlag = false;
    }

    JST_CHECK(gimpl->text->present());

    return Result::SUCCESS;
}

template<Device D, typename T>
const Extent2D<U64>& Lineplot<D, T>::viewSize(const Extent2D<U64>& viewSize) {
    if (gimpl->surface->size(viewSize * config.scale) != this->viewSize() * config.scale) {
        config.viewSize = gimpl->surface->size() / config.scale;
        gimpl->updateState(*this);
    }

    return this->viewSize();
}

template<Device D, typename T>
std::pair<F32, F32> Lineplot<D, T>::zoom(const Extent2D<F32>& mouse_pos, const F32& zoom) {
    if (zoom < 1.0f) {
        config.zoom = 1.0f;
        config.translation = 0.0f;
    } else {
        const auto& before_mouse_x = mouse_pos.x + config.translation;
        const auto& after_mouse_x = (before_mouse_x * config.zoom) / zoom;
        config.zoom = zoom;
        config.translation += after_mouse_x - before_mouse_x;
    }

    gimpl->updateState(*this);

    return {
        config.zoom,
        config.translation
    };
}

template<Device D, typename T>
const F32& Lineplot<D, T>::translation(const F32& translation) {
    config.translation = translation;

    gimpl->updateState(*this);

    return config.translation;
}

template<Device D, typename T>
const U64& Lineplot<D, T>::averaging(const U64& averaging) {
    config.averaging = averaging;

    return config.averaging;
}

template<Device D, typename T>
const F32& Lineplot<D, T>::scale(const F32& scale) {
    if (scale < 0.0f) {
        JST_ERROR("Invalid scale ({}). It should be positive.", scale);
        return config.scale;
    }

    if (scale != config.scale) {
        config.scale = scale;
        gimpl->updateState(*this);
    }

    return config.scale;
}

template<Device D, typename T>
const Extent2D<F32>& Lineplot<D, T>::cursor(const Extent2D<F32>& cursorPos) {
    gimpl->cursorPos = cursorPos;

    gimpl->updateCursorState(*this);

    return gimpl->cursorPos;
}

template<Device D, typename T>
void Lineplot<D, T>::GImpl::updateState(Lineplot<D, T>& m) {
    const F32 maxTranslation = std::abs((1.0f / m.config.zoom) - 1.0f);
    m.config.translation = std::clamp(m.config.translation, -maxTranslation, maxTranslation);

    // Update global pixel size and scale.

    pixelSize = {
        2.0f / m.config.viewSize.x / m.config.scale,
        2.0f / m.config.viewSize.y / m.config.scale
    };

    const auto PadSize = (11.0f + 5.0f + 10.0f) * 2.0f;
    paddingScale = {
        1.0f - pixelSize.x * PadSize,
        1.0f - pixelSize.y * PadSize,
    };

    // Update the transform.

    // The transform matrix is initialized as an identity matrix.

    auto signalTransform = glm::mat4(1.0f);
    auto gridTransform = glm::mat4(1.0f);

    // Apply the translation according to the mouse position.

    signalTransform = glm::translate(signalTransform, glm::vec3(m.config.translation * m.config.zoom, 0.0f, 0.0f));

    // Scale everything to accomodate axis.

    signalTransform = glm::scale(signalTransform, glm::vec3(paddingScale.x, paddingScale.y, 1.0f));
    gridTransform = glm::scale(gridTransform, glm::vec3(paddingScale.x, paddingScale.y, 1.0f));

    // Update the signal and grid uniform buffers.

    signalUniforms.transform = signalTransform;
    signalUniforms.thickness[0] = pixelSize.x * m.config.thickness * 3.0f;
    signalUniforms.thickness[1] = pixelSize.y * m.config.thickness * 3.0f;
    signalUniforms.zoom = m.config.zoom;
    signalUniforms.numberOfPoints = numberOfElements;

    gridUniforms.transform = gridTransform;
    gridUniforms.thickness[0] = pixelSize.x * m.config.thickness * 3.0f;
    gridUniforms.thickness[1] = pixelSize.y * m.config.thickness * 3.0f;
    gridUniforms.zoom = 1.0f;
    gridUniforms.numberOfLines = m.config.numberOfVerticalLines + m.config.numberOfHorizontalLines;

    // Update the cursor.

    updateCursorState(m);

    // Schedule the uniform buffers for update.

    updateGridUniformBufferFlag = true;
    updateSignalUniformBufferFlag = true;
}

template<Device D, typename T>
void Lineplot<D, T>::GImpl::updateCursorState(Lineplot<D, T>& m) {
    // Fetch closest cursor plot value.
    // TODO: Implement interpolation.

    const auto stepX = (2.0f * paddingScale.x) / numberOfElements;
    const U64 cursorIndex = std::clamp((cursorPos.x + paddingScale.x) / stepX, 0.0f, numberOfElements - 1.0f);

    Tensor<D, F32> signalPointSlice = signalPoints;
    signalPointSlice.slice({cursorIndex, {}});
    Memory::Copy(cursorSignalPoint, signalPointSlice);

    const auto cursorValueX = cursorSignalPoint[0] * paddingScale.x;
    const auto cursorValueY = cursorSignalPoint[1] * paddingScale.y;

    // The transform matrix is initialized as an identity matrix.

    auto transform = glm::mat4(1.0f);

    // Translate the cursor to the cursor position.

    transform = glm::translate(transform, glm::vec3((cursorValueX + m.config.translation) * m.config.zoom, cursorValueY, 0.0f));

    // Scale cursor respecting aspect ratio.

    {
        const auto x = pixelSize.x * m.config.thickness * 15.0f;
        const auto y = pixelSize.y * m.config.thickness * 15.0f;
        transform = glm::scale(transform, glm::vec3(x, y, 1.0f));
    }

    // Update the cursor uniform buffer.

    cursorUniforms.transform = transform;

    // Schedule the uniform buffers for update.

    updateCursorUniformBufferFlag = true;

    // Update the text element.

    text->updatePixelSize(pixelSize);

    for (U64 i = 1; i < m.config.numberOfVerticalLines - 1; i++) {
        auto element = text->get(jst::fmt::format("x{:02d}", i));
        element.position = {(((2.0f * paddingScale.x) / (m.config.numberOfVerticalLines - 1)) * i) - paddingScale.x, 1.0f - pixelSize.y * 5.0f};
        element.fill = jst::fmt::format("{:.02f}", (element.position.x / m.config.zoom) - m.config.translation);
        text->update(jst::fmt::format("x{:02d}", i), element);
    }

    {
        auto element = text->get("axis-x");
        element.position = {0.0f, -1.0f + pixelSize.y * 5.0f};
        text->update("axis-x", element);
    }

    {
        auto element = text->get("axis-y");
        element.position = {-1.0f + pixelSize.x * 5.0f, 0.0f};
        text->update("axis-y", element);
    }

    {
        auto element = text->get("amplitude");
        element.fill = jst::fmt::format("({:.05}, {:.05})", cursorValueX, cursorValueY);
        element.position = {(cursorValueX + m.config.translation) * m.config.zoom + 0.05f, cursorValueY - 0.05f};
        text->update("amplitude", element);
    }
}

template<Device D, typename T>
void Lineplot<D, T>::GImpl::generateGridPoints(Lineplot<D, T>& m) {
    const U64& num_cols = m.config.numberOfVerticalLines;
    const U64& num_rows = m.config.numberOfHorizontalLines;

    const F32 x_step  = +2.0f / (num_cols - 1);
    const F32 y_step  = +2.0f / (num_rows - 1);
    const F32 x_start = -1.0f;
    const F32 y_start = -1.0f;
    const F32 x_end   = +1.0f;
    const F32 y_end   = +1.0f;

    for (U64 row = 0; row < num_rows; row++) {
        const F32 y = y_start + row * y_step;

        gridPoints[{row, 0, 0}] = x_start;
        gridPoints[{row, 0, 1}] = y;

        gridPoints[{row, 1, 0}] = x_end;
        gridPoints[{row, 1, 1}] = y;
    }

    for (U64 col = 0; col < num_cols; col++) {
        const F32 x = x_start + col * x_step;

        gridPoints[{col + num_rows, 0, 0}] = x;
        gridPoints[{col + num_rows, 0, 1}] = y_start;

        gridPoints[{col + num_rows, 1, 0}] = x;
        gridPoints[{col + num_rows, 1, 1}] = y_end;
    }

    updateGridPointsFlag = true;
}

template<Device D, typename T>
Render::Texture& Lineplot<D, T>::getTexture() {
    return *gimpl->framebufferTexture;
};

template<Device D, typename T>
const Extent2D<F32>& Lineplot<D, T>::cursor() const {
    return gimpl->cursorPos;
}

}  // namespace Jetstream
