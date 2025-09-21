#include <span>

#include <glm/mat4x4.hpp>

#include "jetstream/render/base.hh"
#include "jetstream/render/utils.hh"
#include "jetstream/render/components/shapes.hh"

#include "jetstream/types.hh"
#include "resources/shaders/global_shaders.hh"

namespace Jetstream::Render::Components {

Shapes::Shapes(const Config& config) {
    this->config = config;
    this->pimpl = std::make_unique<Impl>(this->config);
}

Shapes::~Shapes() {
    pimpl.reset();
}

struct Shapes::Impl {
    // Shader Types.

    struct UniformBuffer {
        glm::vec2 pixelSize;
        glm::vec2 _padding;
    };

    struct ShapeProperties {
        glm::vec4 borderColor;
        glm::vec4 shapeParams;
    };

    struct InstanceData {
        glm::mat4 transform;
        glm::vec4 fillColor;
        glm::vec4 borderColor;
        glm::vec4 shapeParams; // x=type, y=borderWidth, z=cornerRadius, w=unused
    };

    // Internal Types.

    struct Element {
        const ElementConfig& config;
        std::span<ColorRGBA<F32>> instanceColorsBuffer;
        std::span<F32> instanceRotationsBuffer;
        std::span<Extent2D<F32>> instancePositionsBuffer;
        std::span<Extent2D<F32>> instanceSizesBuffer;
    };

    // Variables.

    const Config& config;
    UniformBuffer uniforms;
    U64 totalNumberOfInstances = 0;
    U64 totalNumberOfVertices = 0;
    U64 totalNumberOfIndices = 0;
    std::vector<Element> elements;
    std::unordered_map<std::string, U64> elementIndex;

    // Render.

    bool updateUniformBufferFlag = false;
    bool updateVerticesBufferFlag = false;
    bool updateIndicesBufferFlag = false;

    bool updatePropertiesBufferFlag = false;
    bool updateColorBufferFlag = false;
    bool updateRotationBufferFlag = false;
    bool updateSizeBufferFlag = false;
    bool updatePositionBufferFlag = false;

    bool computeInstanceBufferFlag = false;

    Tensor<Device::CPU, glm::vec2> vertices;
    Tensor<Device::CPU, InstanceData> instances;
    Tensor<Device::CPU, U32> indices;
    Tensor<Device::CPU, ShapeProperties> properties;

    Tensor<Device::CPU, ColorRGBA<F32>> instanceColors;
    Tensor<Device::CPU, F32> instanceRotations;
    Tensor<Device::CPU, Extent2D<F32>> instancePositions;
    Tensor<Device::CPU, Extent2D<F32>> instanceSizes;

    std::shared_ptr<Render::Buffer> uniformBuffer;
    std::shared_ptr<Render::Buffer> verticesBuffer;
    std::shared_ptr<Render::Buffer> instanceBuffer;
    std::shared_ptr<Render::Buffer> indicesBuffer;

    // Compute kernel buffers
    std::shared_ptr<Render::Buffer> propertiesBuffer;
    std::shared_ptr<Render::Buffer> colorsBuffer;
    std::shared_ptr<Render::Buffer> rotationsBuffer;
    std::shared_ptr<Render::Buffer> sizesBuffer;
    std::shared_ptr<Render::Buffer> positionsBuffer;
    std::shared_ptr<Render::Buffer> borderWidthsBuffer;
    std::shared_ptr<Render::Buffer> borderColorsBuffer;

    std::shared_ptr<Render::Vertex> vertex;
    std::shared_ptr<Render::Draw> draw;
    std::shared_ptr<Render::Program> program;
    std::shared_ptr<Render::Kernel> transformKernel;

    // Methods.

    Result updateUniforms();
    Result updateVertices();
    Result updateIndices();

    Result generateElementsProperties();
    Result generateElementsVertices();

    // Constructor.

    Impl(const Config& config) : config(config) {}
};

Result Shapes::create(Window* window) {
    JST_DEBUG("[SHAPES] Loading new shapes.");

    // Calculate geometry requirements.

    pimpl->totalNumberOfVertices = 4;
    pimpl->totalNumberOfIndices = 6;

    for (const auto& [_, elementConfig] : config.elements) {
        pimpl->totalNumberOfInstances += elementConfig.numberOfInstances;
    }

    // Reserve memory.

    pimpl->vertices = Tensor<Device::CPU, glm::vec2>({pimpl->totalNumberOfVertices});
    pimpl->indices = Tensor<Device::CPU, U32>({pimpl->totalNumberOfIndices});
    pimpl->instances = Tensor<Device::CPU, Impl::InstanceData>({pimpl->totalNumberOfInstances});
    pimpl->properties = Tensor<Device::CPU, Impl::ShapeProperties>({pimpl->totalNumberOfInstances});

    pimpl->instanceColors = Tensor<Device::CPU, ColorRGBA<F32>>({pimpl->totalNumberOfInstances});
    pimpl->instanceRotations = Tensor<Device::CPU, F32>({pimpl->totalNumberOfInstances});
    pimpl->instancePositions = Tensor<Device::CPU, Extent2D<F32>>({pimpl->totalNumberOfInstances});
    pimpl->instanceSizes = Tensor<Device::CPU, Extent2D<F32>>({pimpl->totalNumberOfInstances});

    // Debug information.

    JST_DEBUG("[SHAPES] Total number of vertices: {}", pimpl->totalNumberOfVertices);
    JST_DEBUG("[SHAPES] Total number of indices: {}", pimpl->totalNumberOfIndices);
    JST_DEBUG("[SHAPES] Total number of instances: {}", pimpl->totalNumberOfInstances);

    // Create render surface buffers.

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &pimpl->uniforms;
        cfg.elementByteSize = sizeof(pimpl->uniforms);
        cfg.size = 1;
        cfg.target = Render::Buffer::Target::UNIFORM;
        JST_CHECK(window->build(pimpl->uniformBuffer, cfg));
        JST_CHECK(window->bind(pimpl->uniformBuffer));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = pimpl->vertices.data();
        cfg.elementByteSize = sizeof(glm::vec2);
        cfg.size = pimpl->vertices.size();
        cfg.target = Render::Buffer::Target::VERTEX;
        JST_CHECK(window->build(pimpl->verticesBuffer, cfg));
        JST_CHECK(window->bind(pimpl->verticesBuffer));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = pimpl->instances.data();
        cfg.elementByteSize = sizeof(Shapes::Impl::InstanceData);
        cfg.size = pimpl->instances.size();
        cfg.target = Render::Buffer::Target::VERTEX | Render::Buffer::Target::STORAGE;
        JST_CHECK(window->build(pimpl->instanceBuffer, cfg));
        JST_CHECK(window->bind(pimpl->instanceBuffer));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = pimpl->indices.data();
        cfg.elementByteSize = sizeof(U32);
        cfg.size = pimpl->indices.size();
        cfg.target = Render::Buffer::Target::VERTEX_INDICES;
        JST_CHECK(window->build(pimpl->indicesBuffer, cfg));
        JST_CHECK(window->bind(pimpl->indicesBuffer));
    }

    // Create compute kernel buffers.

    {
        Render::Buffer::Config cfg;
        cfg.buffer = pimpl->properties.data();
        cfg.elementByteSize = sizeof(Impl::ShapeProperties);
        cfg.size = pimpl->properties.size();
        cfg.target = Render::Buffer::Target::STORAGE;
        JST_CHECK(window->build(pimpl->propertiesBuffer, cfg));
        JST_CHECK(window->bind(pimpl->propertiesBuffer));
    }

    {
        auto [buffer, enableZeroCopy] = ConvertToOptimalStorage(window, pimpl->instanceColors);

        Render::Buffer::Config cfg;
        cfg.buffer = buffer;
        cfg.elementByteSize = sizeof(glm::vec4);
        cfg.size = pimpl->instanceColors.size();
        cfg.enableZeroCopy = enableZeroCopy;
        cfg.target = Render::Buffer::Target::STORAGE;
        JST_CHECK(window->build(pimpl->colorsBuffer, cfg));
        JST_CHECK(window->bind(pimpl->colorsBuffer));
    }

    {
        auto [buffer, enableZeroCopy] = ConvertToOptimalStorage(window, pimpl->instanceRotations);

        Render::Buffer::Config cfg;
        cfg.buffer = buffer;
        cfg.elementByteSize = sizeof(F32);
        cfg.size = pimpl->instanceRotations.size();
        cfg.enableZeroCopy = enableZeroCopy;
        cfg.target = Render::Buffer::Target::STORAGE;
        JST_CHECK(window->build(pimpl->rotationsBuffer, cfg));
        JST_CHECK(window->bind(pimpl->rotationsBuffer));
    }

    {
        auto [buffer, enableZeroCopy] = ConvertToOptimalStorage(window, pimpl->instanceSizes);

        Render::Buffer::Config cfg;
        cfg.buffer = buffer;
        cfg.elementByteSize = sizeof(glm::vec2);
        cfg.size = pimpl->instanceSizes.size();
        cfg.enableZeroCopy = enableZeroCopy;
        cfg.target = Render::Buffer::Target::STORAGE;
        JST_CHECK(window->build(pimpl->sizesBuffer, cfg));
        JST_CHECK(window->bind(pimpl->sizesBuffer));
    }

    {
        auto [buffer, enableZeroCopy] = ConvertToOptimalStorage(window, pimpl->instancePositions);

        Render::Buffer::Config cfg;
        cfg.buffer = buffer;
        cfg.elementByteSize = sizeof(glm::vec2);
        cfg.size = pimpl->instancePositions.size();
        cfg.enableZeroCopy = enableZeroCopy;
        cfg.target = Render::Buffer::Target::STORAGE;
        JST_CHECK(window->build(pimpl->positionsBuffer, cfg));
        JST_CHECK(window->bind(pimpl->positionsBuffer));
    }

    {
        Render::Vertex::Config cfg;
        cfg.vertices = {
            {pimpl->verticesBuffer, 2},
        };
        cfg.instances = {
            {pimpl->instanceBuffer, sizeof(Shapes::Impl::InstanceData) / sizeof(F32)},
        };
        cfg.indices = pimpl->indicesBuffer;
        JST_CHECK(window->build(pimpl->vertex, cfg));
    }

    {
        Render::Draw::Config cfg;
        cfg.numberOfDraws = 1;
        cfg.numberOfInstances = pimpl->totalNumberOfInstances;
        cfg.buffer = pimpl->vertex;
        cfg.mode = Render::Draw::Mode::TRIANGLES;
        JST_CHECK(window->build(pimpl->draw, cfg));
    }

    // Create transform compute kernel
    {
        Render::Kernel::Config cfg;
        cfg.gridSize = {pimpl->totalNumberOfInstances, 1, 1};
        cfg.kernels = GlobalKernelsPackage["shapes"];
        cfg.buffers = {
            {pimpl->uniformBuffer, Render::Kernel::AccessMode::READ},
            {pimpl->propertiesBuffer, Render::Kernel::AccessMode::READ},
            {pimpl->colorsBuffer, Render::Kernel::AccessMode::READ},
            {pimpl->rotationsBuffer, Render::Kernel::AccessMode::READ},
            {pimpl->sizesBuffer, Render::Kernel::AccessMode::READ},
            {pimpl->positionsBuffer, Render::Kernel::AccessMode::READ},
            {pimpl->instanceBuffer, Render::Kernel::AccessMode::WRITE},
        };
        JST_CHECK(window->build(pimpl->transformKernel, cfg));
    }

    {
        Render::Program::Config cfg;
        cfg.shaders = GlobalShadersPackage["shapes"];
        cfg.draws = {
            pimpl->draw,
        };
        cfg.buffers = {
            {pimpl->uniformBuffer, Render::Program::Target::VERTEX |
                                   Render::Program::Target::FRAGMENT},
        };
        cfg.enableAlphaBlending = true;
        JST_CHECK(window->build(pimpl->program, cfg));
    }

    // Create element data.

    U64 currentInstanceOffset = 0;
    for (const auto& [id, elementConfig] : config.elements) {
        const U64 instanceCount = elementConfig.numberOfInstances;

        const auto& element = Impl::Element{
            .config = elementConfig,
            .instanceColorsBuffer = std::span<ColorRGBA<F32>>({pimpl->instanceColors.data(), pimpl->instanceColors.size()}).subspan(currentInstanceOffset, instanceCount),
            .instanceRotationsBuffer = std::span<F32>({pimpl->instanceRotations.data(), pimpl->instanceRotations.size()}).subspan(currentInstanceOffset, instanceCount),
            .instancePositionsBuffer = std::span<Extent2D<F32>>({pimpl->instancePositions.data(), pimpl->instancePositions.size()}).subspan(currentInstanceOffset, instanceCount),
            .instanceSizesBuffer = std::span<Extent2D<F32>>({pimpl->instanceSizes.data(), pimpl->instanceSizes.size()}).subspan(currentInstanceOffset, instanceCount),
        };

        for (U64 i = 0; i < instanceCount; ++i) {
            element.instanceColorsBuffer[i] = elementConfig.color;
        }
        pimpl->updateColorBufferFlag = true;

        for (U64 i = 0; i < instanceCount; ++i) {
            element.instanceRotationsBuffer[i] = elementConfig.rotation;
        }
        pimpl->updateRotationBufferFlag = true;

        for (U64 i = 0; i < instanceCount; ++i) {
            element.instancePositionsBuffer[i] = elementConfig.position;
        }
        pimpl->updatePositionBufferFlag = true;

        for (U64 i = 0; i < instanceCount; ++i) {
            element.instanceSizesBuffer[i] = elementConfig.size;
        }
        pimpl->updateSizeBufferFlag = true;

        pimpl->elementIndex[id] = pimpl->elements.size();
        pimpl->elements.push_back(element);

        currentInstanceOffset += instanceCount;
    }

    JST_CHECK(pimpl->generateElementsVertices());
    JST_CHECK(pimpl->generateElementsProperties());

    // Load static state.

    JST_CHECK(pimpl->updateUniforms());
    JST_CHECK(pimpl->updateVertices());
    JST_CHECK(pimpl->updateIndices());

    return Result::SUCCESS;
}

Result Shapes::destroy(Window* window) {
    JST_DEBUG("[SHAPES] Unloading shapes.");

    JST_CHECK(window->unbind(pimpl->uniformBuffer));
    JST_CHECK(window->unbind(pimpl->verticesBuffer));
    JST_CHECK(window->unbind(pimpl->instanceBuffer));
    JST_CHECK(window->unbind(pimpl->indicesBuffer));

    // Unbind compute buffers
    JST_CHECK(window->unbind(pimpl->propertiesBuffer));
    JST_CHECK(window->unbind(pimpl->colorsBuffer));
    JST_CHECK(window->unbind(pimpl->rotationsBuffer));
    JST_CHECK(window->unbind(pimpl->sizesBuffer));
    JST_CHECK(window->unbind(pimpl->positionsBuffer));

    return Result::SUCCESS;
}

Result Shapes::surface(Render::Surface::Config& config) {
    JST_DEBUG("[SHAPES] Binding shapes to surface.");

    config.programs.push_back(pimpl->program);
    config.kernels.push_back(pimpl->transformKernel);

    return Result::SUCCESS;
}

Result Shapes::Impl::generateElementsProperties() {
    // Set element properties for all instances.

    U64 currentInstanceOffset = 0;
    for (auto& element : elements) {
        // Fill properties for all instances of this element
        for (U64 i = 0; i < element.config.numberOfInstances; ++i) {
            auto& prop = properties[currentInstanceOffset + i];

            prop.borderColor = glm::vec4(element.config.borderColor.r,
                                         element.config.borderColor.g,
                                         element.config.borderColor.b,
                                         element.config.borderColor.a);
            prop.shapeParams = glm::vec4(element.config.type,
                                         element.config.borderWidth,
                                         element.config.cornerRadius,
                                         0.0f);
        }

        currentInstanceOffset += element.config.numberOfInstances;
    }

    // Set flag to update buffer.
    updatePropertiesBufferFlag = true;

    return Result::SUCCESS;
}

Result Shapes::Impl::generateElementsVertices() {
    glm::vec2* vertexData = vertices.data();
    U32* indexData = indices.data();

    vertexData[0] = {-1.0f, -1.0f}; // Bottom left
    vertexData[1] = { 1.0f, -1.0f}; // Bottom right
    vertexData[2] = { 1.0f,  1.0f}; // Top right
    vertexData[3] = {-1.0f,  1.0f}; // Top left

    indexData[0] = 0;
    indexData[1] = 1;
    indexData[2] = 2;
    indexData[3] = 2;
    indexData[4] = 3;
    indexData[5] = 0;

    // Set flags to update buffers.
    updateVerticesBufferFlag = true;
    updateIndicesBufferFlag = true;

    return Result::SUCCESS;
}

Result Shapes::getColors(const std::string& elementId, std::span<ColorRGBA<F32>>& colors) const {
    if (!pimpl->elementIndex.contains(elementId)) {
        JST_ERROR("[SHAPES] Element not found (color).");
        return Result::ERROR;
    }
    colors = pimpl->elements.at(pimpl->elementIndex.at(elementId)).instanceColorsBuffer;
    return Result::SUCCESS;
}

Result Shapes::updateColors(const std::string&) {
    // TODO: Implement element specific update.
    pimpl->updateColorBufferFlag = true;
    return Result::SUCCESS;
}

Result Shapes::getRotations(const std::string& elementId, std::span<F32>& rotations) const {
    if (!pimpl->elementIndex.contains(elementId)) {
        JST_ERROR("[SHAPES] Element not found (rotation).");
        return Result::ERROR;
    }
    rotations = pimpl->elements.at(pimpl->elementIndex.at(elementId)).instanceRotationsBuffer;
    return Result::SUCCESS;
}

Result Shapes::updateRotations(const std::string&) {
    // TODO: Implement element specific update.
    pimpl->updateRotationBufferFlag = true;
    return Result::SUCCESS;
}

Result Shapes::getPositions(const std::string& elementId, std::span<Extent2D<F32>>& positions) const {
    if (!pimpl->elementIndex.contains(elementId)) {
        JST_ERROR("[SHAPES] Element not found (position).");
        return Result::ERROR;
    }
    positions = pimpl->elements.at(pimpl->elementIndex.at(elementId)).instancePositionsBuffer;
    return Result::SUCCESS;
}

Result Shapes::updatePositions(const std::string&) {
    // TODO: Implement element specific update.
    pimpl->updatePositionBufferFlag = true;
    return Result::SUCCESS;
}

Result Shapes::getSizes(const std::string& elementId, std::span<Extent2D<F32>>& sizes) const {
    if (!pimpl->elementIndex.contains(elementId)) {
        JST_ERROR("[SHAPES] Element not found (size).");
        return Result::ERROR;
    }
    sizes = pimpl->elements.at(pimpl->elementIndex.at(elementId)).instanceSizesBuffer;
    return Result::SUCCESS;
}

Result Shapes::updateSizes(const std::string&) {
    // TODO: Implement element specific update.
    pimpl->updateSizeBufferFlag = true;
    return Result::SUCCESS;
}

Result Shapes::Impl::updateUniforms() {
    // Set data.
    uniforms.pixelSize = glm::vec2(config.pixelSize.x, config.pixelSize.y);

    // Set flag to update buffer.
    updateUniformBufferFlag = true;

    return Result::SUCCESS;
}

Result Shapes::Impl::updateVertices() {
    // Set flag to update buffer.
    updateVerticesBufferFlag = true;

    return Result::SUCCESS;
}

Result Shapes::Impl::updateIndices() {
    // Set flag to update buffer.
    updateIndicesBufferFlag = true;

    return Result::SUCCESS;
}

Result Shapes::updatePixelSize(const Extent2D<F32>& pixelSize) {
    // Check if pixel size has changed.
    bool shouldUpdateUniforms = config.pixelSize != pixelSize;

    // Update pixel size.
    config.pixelSize = pixelSize;

    // Update uniforms.
    if (shouldUpdateUniforms) {
        JST_CHECK(pimpl->updateUniforms());
    }

    return Result::SUCCESS;
}

Result Shapes::present() {
    // Update render buffers.

    if (pimpl->updateVerticesBufferFlag) {
        pimpl->verticesBuffer->update();
        pimpl->updateVerticesBufferFlag = false;
    }

    if (pimpl->updateIndicesBufferFlag) {
        pimpl->indicesBuffer->update();
        pimpl->updateIndicesBufferFlag = false;
    }

    if (pimpl->updateUniformBufferFlag) {
        pimpl->uniformBuffer->update();
        pimpl->updateUniformBufferFlag = false;
        pimpl->computeInstanceBufferFlag = true;
    }

    // Update compute buffers.

    if (pimpl->updatePropertiesBufferFlag) {
        pimpl->propertiesBuffer->update();
        pimpl->updatePropertiesBufferFlag = false;
        pimpl->computeInstanceBufferFlag = true;
    }

    if (pimpl->updateColorBufferFlag) {
        pimpl->colorsBuffer->update();
        pimpl->updateColorBufferFlag = false;
        pimpl->computeInstanceBufferFlag = true;
    }

    if (pimpl->updateRotationBufferFlag) {
        pimpl->rotationsBuffer->update();
        pimpl->updateRotationBufferFlag = false;
        pimpl->computeInstanceBufferFlag = true;
    }

    if (pimpl->updatePositionBufferFlag) {
        pimpl->positionsBuffer->update();
        pimpl->updatePositionBufferFlag = false;
        pimpl->computeInstanceBufferFlag = true;
    }

    if (pimpl->updateSizeBufferFlag) {
        pimpl->sizesBuffer->update();
        pimpl->updateSizeBufferFlag = false;
        pimpl->computeInstanceBufferFlag = true;
    }

    // Compute instance buffer.

    if (pimpl->computeInstanceBufferFlag) {
        pimpl->transformKernel->update();
        pimpl->computeInstanceBufferFlag = false;
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render::Components
