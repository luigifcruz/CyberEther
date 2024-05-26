#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "jetstream/backend/devices/cpu/helpers.hh"
#include "jetstream/modules/lineplot.hh"
#include "jetstream/render/utils.hh"

#include "shaders/lineplot_shaders.hh"
#include "shaders/global_shaders.hh"

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
    numberOfElements = input.buffer.shape()[last_axis];
    numberOfBatches = (input.buffer.rank() == 2) ? input.buffer.shape()[0] : 1;
    normalizationFactor = 1.0f / (0.5f * numberOfBatches);

    // Check shape.

    if (numberOfElements < 2) {
        JST_ERROR("Invalid number of elements ({}). It should be at least '2'.", numberOfElements);
        return Result::ERROR;
    }

    // Allocate internal buffers.

    signalPoints = Tensor<D, F32>({numberOfElements, 2});
    signalVertices = Tensor<D, F32>({numberOfElements - 1, 4, 2});

    gridPoints = Tensor<Device::CPU, F32>({config.numberOfVerticalLines + config.numberOfHorizontalLines, 2, 2});
    gridVertices = Tensor<D, F32>({config.numberOfVerticalLines + config.numberOfHorizontalLines, 6, 4});    

    return Result::SUCCESS;
}

template<Device D, typename T>
void Lineplot<D, T>::info() const {
    JST_DEBUG("  Averaging: {}", config.averaging);
    JST_DEBUG("  Number of Vertical Lines: {}", config.numberOfVerticalLines);
    JST_DEBUG("  Number of Horizontal Lines: {}", config.numberOfHorizontalLines);
    JST_DEBUG("  Size: [{}, {}]", config.viewSize.width, config.viewSize.height);
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
        JST_CHECK(window->build(gridUniformBuffer, cfg));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = gridPoints.data();
        cfg.elementByteSize = sizeof(F32);
        cfg.size = gridPoints.size();
        cfg.target = Render::Buffer::Target::STORAGE;
        cfg.enableZeroCopy = false;
        JST_CHECK(window->build(gridPointsBuffer, cfg));
    }

    {
        auto [buffer, enableZeroCopy] = ConvertToOptimalStorage(window, gridVertices);

        Render::Buffer::Config cfg;
        cfg.buffer = buffer;
        cfg.elementByteSize = sizeof(F32);
        cfg.size = gridVertices.size();
        cfg.target = Render::Buffer::Target::VERTEX | Render::Buffer::Target::STORAGE;
        cfg.enableZeroCopy = enableZeroCopy;
        JST_CHECK(window->build(gridVerticesBuffer, cfg));
    }

    {
        Render::Kernel::Config cfg;
        cfg.gridSize = {config.numberOfVerticalLines + config.numberOfHorizontalLines, 1, 1};
        cfg.kernels = GlobalKernelsPackage["thicklines"];
        cfg.buffers = {gridUniformBuffer, gridPointsBuffer, gridVerticesBuffer};
        JST_CHECK(window->build(gridKernel, cfg));
    }

    {
        Render::Vertex::Config cfg;
        cfg.buffers = {
            {gridVerticesBuffer, 4},
        };
        JST_CHECK(window->build(gridVertex, cfg));
    }

    {
        Render::Draw::Config cfg;
        cfg.buffer = gridVertex;
        cfg.mode = Render::Draw::Mode::TRIANGLES;
        JST_CHECK(window->build(drawGridVertex, cfg));
    }

    {
        Render::Program::Config cfg;
        cfg.shaders = ShadersPackage["grid"];
        cfg.draw = drawGridVertex;
        cfg.buffers = {
            {gridUniformBuffer, Render::Program::Target::VERTEX | 
                                Render::Program::Target::FRAGMENT},
        };
        JST_CHECK(window->build(gridProgram, cfg));
    }

    // Signal element.

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &gimpl->signalUniforms;
        cfg.elementByteSize = sizeof(gimpl->signalUniforms);
        cfg.size = 1;
        cfg.target = Render::Buffer::Target::UNIFORM;
        JST_CHECK(window->build(signalUniformBuffer, cfg));
    }

    {
        auto [buffer, enableZeroCopy] = ConvertToOptimalStorage(window, signalPoints);

        Render::Buffer::Config cfg;
        cfg.buffer = buffer;
        cfg.elementByteSize = sizeof(F32);
        cfg.size = signalPoints.size();
        cfg.target = Render::Buffer::Target::STORAGE;
        cfg.enableZeroCopy = enableZeroCopy;
        JST_CHECK(window->build(signalPointsBuffer, cfg));
    }

    {
        auto [buffer, enableZeroCopy] = ConvertToOptimalStorage(window, signalVertices);

        Render::Buffer::Config cfg;
        cfg.buffer = buffer;
        cfg.elementByteSize = sizeof(F32);
        cfg.size = signalVertices.size();
        cfg.target = Render::Buffer::Target::VERTEX | Render::Buffer::Target::STORAGE;
        cfg.enableZeroCopy = enableZeroCopy;
        JST_CHECK(window->build(signalVerticesBuffer, cfg));
    }

    {
        Render::Kernel::Config cfg;
        cfg.gridSize = {numberOfElements - 1, 1, 1};
        cfg.kernels = GlobalKernelsPackage["thicklinestrip"];
        cfg.buffers = {signalUniformBuffer, signalPointsBuffer, signalVerticesBuffer};
        JST_CHECK(window->build(signalKernel, cfg));
    }

    {
        Render::Vertex::Config cfg;
        cfg.buffers = {
            {signalVerticesBuffer, 2},
        };
        JST_CHECK(window->build(signalVertex, cfg));
    }

    {
        Render::Draw::Config cfg;
        cfg.buffer = signalVertex;
        cfg.mode = Render::Draw::Mode::TRIANGLE_STRIP;
        JST_CHECK(window->build(drawSignalVertex, cfg));
    }
    
    {
        Render::Texture::Config cfg;
        cfg.size = {256, 1};
        cfg.buffer = (uint8_t*)Render::Extras::TurboLutBytes;
        JST_CHECK(window->build(lutTexture, cfg));
    }

    {
        Render::Program::Config cfg;
        cfg.shaders = ShadersPackage["signal"];
        cfg.draw = drawSignalVertex;
        cfg.textures = {lutTexture};
        cfg.buffers = {
            {signalUniformBuffer, Render::Program::Target::VERTEX | 
                                  Render::Program::Target::FRAGMENT},
        };
        JST_CHECK(window->build(signalProgram, cfg));
    }

    // Surface.

    {
        Render::Texture::Config cfg;
        cfg.size = config.viewSize * config.scale;
        JST_CHECK(window->build(framebufferTexture, cfg));
    }

    {
        Render::Surface::Config cfg;
        cfg.framebuffer = framebufferTexture;
        cfg.kernels = {gridKernel, signalKernel};
        cfg.programs = {gridProgram, signalProgram};
        cfg.buffers = {
            signalPointsBuffer,
            signalVerticesBuffer,
            gridPointsBuffer,
            gridVerticesBuffer,
            gridUniformBuffer,
            signalUniformBuffer
        };
        cfg.multisampled = true;
        JST_CHECK(window->build(surface, cfg));
        JST_CHECK(window->bind(surface));
    }

    // Generate resources.

    generateGridPoints();

    // Initialize variables.

    updateState();

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Lineplot<D, T>::destroyPresent() {
    JST_CHECK(window->unbind(surface));

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Lineplot<D, T>::present() {
    if (updateGridPointsFlag) {
        gridPointsBuffer->update();
        updateGridPointsFlag = false;
    }

    if (updateSignalPointsFlag) {
        signalPointsBuffer->update();
        updateSignalPointsFlag = false;
    }

    if (updateGridUniformBufferFlag) {
        gridUniformBuffer->update();
        updateGridUniformBufferFlag = false;
    }

    if (updateSignalUniformBufferFlag) {
        signalUniformBuffer->update();
        updateSignalUniformBufferFlag = false;
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
const Size2D<U64>& Lineplot<D, T>::viewSize(const Size2D<U64>& viewSize) {
    if (surface->size(viewSize * config.scale) != this->viewSize()) {
        config.viewSize = surface->size() / config.scale;
    }

    updateState();

    return this->viewSize();
}

template<Device D, typename T>
std::pair<F32, F32> Lineplot<D, T>::zoom(const std::pair<F32, F32>& mouse_pos, const F32& zoom) {
    if (zoom < 1.0f) {
        config.zoom = 1.0f;
        config.translation = 0.0f;
    } else {
        const auto& before_mouse_x = mouse_pos.first;
        const auto& after_mouse_x = (before_mouse_x * config.zoom) / zoom;
        config.zoom = zoom;
        config.translation += after_mouse_x - before_mouse_x;
    }

    updateState();

    return {config.zoom, config.translation};
}

template<Device D, typename T>
const F32& Lineplot<D, T>::translation(const F32& translation) {
    config.translation = translation;

    updateState();

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
    }

    return config.scale;
}

template<Device D, typename T>
void Lineplot<D, T>::updateState() {
    const F32 maxTranslation = std::abs((1.0f / config.zoom) - 1.0f);
    config.translation = std::clamp(config.translation, -maxTranslation, maxTranslation);

    // Update thickness.

    auto& [x, y] = thickness;
    x = ((2.0f / config.viewSize.width) * config.thickness) / 2.0f;
    y = ((2.0f / config.viewSize.height) * config.thickness) / 2.0f;

    // Update the transform.

    // The transform matrix is initialized as an identity matrix.
    auto transform = glm::mat4(1.0f);

    // Apply the translation according to the mouse position.
    transform = glm::translate(transform, glm::vec3(config.translation * config.zoom, 0.0f, 0.0f));

    // Update the signal and grid uniform buffers.

    gimpl->signalUniforms.transform = transform;
    gimpl->signalUniforms.thickness[0] = thickness.first;
    gimpl->signalUniforms.thickness[1] = thickness.second;
    gimpl->signalUniforms.zoom = config.zoom;
    gimpl->signalUniforms.numberOfPoints = numberOfElements;

    gimpl->gridUniforms.transform = transform;
    gimpl->gridUniforms.thickness[0] = thickness.first;
    gimpl->gridUniforms.thickness[1] = thickness.second;
    gimpl->gridUniforms.zoom = config.zoom;
    gimpl->gridUniforms.numberOfLines = config.numberOfVerticalLines + config.numberOfHorizontalLines;

    // Schedule the uniform buffers for update.

    updateGridUniformBufferFlag = true;
    updateSignalUniformBufferFlag = true;
}

template<Device D, typename T>
void Lineplot<D, T>::generateGridPoints() {
    const U64& num_cols = config.numberOfVerticalLines;
    const U64& num_rows = config.numberOfHorizontalLines;

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
    return *framebufferTexture;
};

}  // namespace Jetstream
