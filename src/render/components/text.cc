#include <span>

#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "jetstream/render/base.hh"
#include "jetstream/render/components/text.hh"

#include "resources/shaders/global_shaders.hh"

namespace Jetstream::Render::Components {

Text::Text(const Config& config) {
    this->config = config;
    this->pimpl = std::make_unique<Impl>(this->config);
}

Text::~Text() {
    pimpl.reset();
}

struct Text::Impl {
    struct UniformBuffer {
        glm::vec3 color;
        F32 sharpness;
    };

    struct Element {
        const ElementConfig& config;

        Extent2D<I32> bounds;
        std::span<glm::vec2> posVertices;
        std::span<glm::vec2> fillVertices;
        std::span<glm::mat4> instances;
    };

    // Variables.

    const Config& config;
    UniformBuffer uniforms;
    std::unordered_map<std::string, Element> elements;

    // Render.

    bool updateFontUniformBufferFlag = false;
    bool updateFontPosVerticesBufferFlag = false;
    bool updateFontFillVerticesBufferFlag = false;
    bool updateFontInstanceBufferFlag = false;
    bool updateFontIndicesBufferFlag = false;

    std::vector<glm::vec2> posVertices;
    std::vector<glm::vec2> fillVertices;
    std::vector<glm::mat4> instances;
    std::vector<U32> indices;

    std::shared_ptr<Render::Buffer> fontUniformBuffer;
    std::shared_ptr<Render::Buffer> fontPosVerticesBuffer;
    std::shared_ptr<Render::Buffer> fontFillVerticesBuffer;
    std::shared_ptr<Render::Buffer> fontInstanceBuffer;
    std::shared_ptr<Render::Buffer> fontIndicesBuffer;

    std::shared_ptr<Render::Vertex> fontVertex;

    std::shared_ptr<Render::Draw> drawFont;

    std::shared_ptr<Render::Program> fontProgram;

    // Methods.

    Result updateUniforms();
    Result updateVertices();
    Result updateInstances();
    Result updateIndices();

    Result updateElementVertex(Element& element);
    Result updateElementInstance(Element& element);

    // Constructor.

    Impl(const Config& config) : config(config) {}
};

Result Text::create(Window* window) {
    JST_DEBUG("[TEXT] Loading new text.");

    // Check if there is any element.
    if (config.elements.empty()) {
        JST_ERROR("[TEXT] No elements to render.");
        return Result::ERROR;
    }

    // Check if font is loaded.
    if (!config.font) {
        JST_ERROR("[TEXT] Font not loaded.");
        return Result::ERROR;
    }

    // Calculate constants.
    const U64 numberOfVertices = config.maxCharacters * 4;
    const U64 numberOfInstances = config.elements.size();
    const U64 numberOfIndices = config.maxCharacters * 6;

    // Reserve memory.
    pimpl->posVertices.resize(numberOfVertices * numberOfInstances);
    pimpl->fillVertices.resize(numberOfVertices * numberOfInstances);
    pimpl->instances.resize(numberOfInstances);
    pimpl->indices.resize(numberOfIndices);

    // Debug information.
    JST_DEBUG("[TEXT] Number of vertices: {}", numberOfVertices);
    JST_DEBUG("[TEXT] Number of instances: {}", numberOfInstances);
    JST_DEBUG("[TEXT] Number of indices: {}", numberOfIndices);

    // Create render surface.

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &pimpl->uniforms;
        cfg.elementByteSize = sizeof(pimpl->uniforms);
        cfg.size = 1;
        cfg.target = Render::Buffer::Target::UNIFORM;
        JST_CHECK(window->build(pimpl->fontUniformBuffer, cfg));
        JST_CHECK(window->bind(pimpl->fontUniformBuffer));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = pimpl->posVertices.data();
        cfg.elementByteSize = sizeof(glm::vec2);
        cfg.size = pimpl->posVertices.size();
        cfg.target = Render::Buffer::Target::VERTEX;
        JST_CHECK(window->build(pimpl->fontPosVerticesBuffer, cfg));
        JST_CHECK(window->bind(pimpl->fontPosVerticesBuffer));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = pimpl->fillVertices.data();
        cfg.elementByteSize = sizeof(glm::vec2);
        cfg.size = pimpl->fillVertices.size();
        cfg.target = Render::Buffer::Target::VERTEX;
        JST_CHECK(window->build(pimpl->fontFillVerticesBuffer, cfg));
        JST_CHECK(window->bind(pimpl->fontFillVerticesBuffer));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = pimpl->instances.data();
        cfg.elementByteSize = sizeof(glm::mat4);
        cfg.size = pimpl->instances.size();
        cfg.target = Render::Buffer::Target::VERTEX;
        JST_CHECK(window->build(pimpl->fontInstanceBuffer, cfg));
        JST_CHECK(window->bind(pimpl->fontInstanceBuffer));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = pimpl->indices.data();
        cfg.elementByteSize = sizeof(U32);
        cfg.size = pimpl->indices.size();
        cfg.target = Render::Buffer::Target::VERTEX_INDICES;
        JST_CHECK(window->build(pimpl->fontIndicesBuffer, cfg));
        JST_CHECK(window->bind(pimpl->fontIndicesBuffer));
    }

    {
        Render::Vertex::Config cfg;
        cfg.vertices = {
            {pimpl->fontPosVerticesBuffer, 2},
            {pimpl->fontFillVerticesBuffer, 2},
        };
        cfg.instances = {
            {pimpl->fontInstanceBuffer, 16},
        };
        cfg.indices = pimpl->fontIndicesBuffer;
        JST_CHECK(window->build(pimpl->fontVertex, cfg));
    }

    {
        Render::Draw::Config cfg;
        cfg.numberOfDraws = numberOfInstances;
        cfg.numberOfInstances = 1;
        cfg.buffer = pimpl->fontVertex;
        cfg.mode = Render::Draw::Mode::TRIANGLES;
        JST_CHECK(window->build(pimpl->drawFont, cfg));
    }

    {
        Render::Program::Config cfg;
        cfg.shaders = GlobalShadersPackage["text"];
        cfg.draws = {
            pimpl->drawFont,
        };
        cfg.textures = {
            config.font->atlas(),
        };
        cfg.buffers = {
            {pimpl->fontUniformBuffer, Render::Program::Target::VERTEX |
                                       Render::Program::Target::FRAGMENT},
        };
        cfg.enableAlphaBlending = true;
        JST_CHECK(window->build(pimpl->fontProgram, cfg));
    }

    // Create element data.

    U32 i = 0;
    for (const auto& [id, _] : config.elements) {
        pimpl->elements.emplace(id, Text::Impl::Element{
            .config = config.elements[id],
            .bounds = {0, 0},
            .posVertices = std::span{pimpl->posVertices}.subspan(numberOfVertices * i, numberOfVertices),
            .fillVertices = std::span{pimpl->fillVertices}.subspan(numberOfVertices * i, numberOfVertices),
            .instances = std::span{pimpl->instances}.subspan(i++, 1),
        });
    }

    // Load static state.
    JST_CHECK(pimpl->updateUniforms());
    JST_CHECK(pimpl->updateVertices());
    JST_CHECK(pimpl->updateInstances());
    JST_CHECK(pimpl->updateIndices());

    return Result::SUCCESS;
}

Result Text::destroy(Window* window) {
    JST_DEBUG("[TEXT] Unloading text.");

    JST_CHECK(window->unbind(pimpl->fontUniformBuffer));
    JST_CHECK(window->unbind(pimpl->fontPosVerticesBuffer));
    JST_CHECK(window->unbind(pimpl->fontFillVerticesBuffer));
    JST_CHECK(window->unbind(pimpl->fontInstanceBuffer));
    JST_CHECK(window->unbind(pimpl->fontIndicesBuffer));

    return Result::SUCCESS;
}

Result Text::surface(Render::Surface::Config& config) {
    JST_DEBUG("[TEXT] Binding text to surface.");

    config.programs.push_back(pimpl->fontProgram);

    return Result::SUCCESS;
}

const Text::ElementConfig& Text::get(const std::string& elementId) const {
    // Check if element exists.
    if (!config.elements.contains(elementId)) {
        JST_ERROR("[TEXT] Element '{}' not found.", elementId);
        JST_CHECK_THROW(Result::ERROR);
    }

    // Copy element data to output.
    return config.elements.at(elementId);
}

Result Text::update(const std::string& elementId, const ElementConfig& elementConfig) {
    // Check if element exists.
    if (!config.elements.contains(elementId)) {
        JST_ERROR("[TEXT] Element '{}' not found.", elementId);
        return Result::ERROR;
    }

    // Get element reference.
    auto& currentElement = config.elements[elementId];
    auto& updatedElement = elementConfig;

    // Check if element data has changed.
    bool shouldUpdateVertices = updatedElement.fill != currentElement.fill;

    // Update element data.
    currentElement = updatedElement;

    // Load element data.
    auto& element = pimpl->elements.at(elementId);

    // Update element vertex.
    if (shouldUpdateVertices) {
        JST_CHECK(pimpl->updateElementVertex(element));

        // Set flag to update buffers.
        pimpl->updateFontPosVerticesBufferFlag = true;
        pimpl->updateFontFillVerticesBufferFlag = true;
    }

    // Update element instance.
    JST_CHECK(pimpl->updateElementInstance(element));

    // Set flag to update buffer.
    pimpl->updateFontInstanceBufferFlag = true;

    return Result::SUCCESS;
}

Result Text::updatePixelSize(const Extent2D<F32>& pixelSize) {
    // Check if pixel size has changed.
    bool shouldUpdateVertices = config.pixelSize != pixelSize;

    // Update pixel size.
    config.pixelSize = pixelSize;

    // Update elements vertices and instances.
    if (shouldUpdateVertices) {
        JST_CHECK(pimpl->updateVertices());
        JST_CHECK(pimpl->updateInstances());
    }

    return Result::SUCCESS;
}

Result Text::Impl::updateUniforms() {
    // Set data.
    uniforms.color = glm::vec3(config.color.r, config.color.g, config.color.b);
    uniforms.sharpness = config.sharpness;

    // Set flag to update buffer.
    updateFontUniformBufferFlag = true;

    return Result::SUCCESS;
}

Result Text::Impl::updateVertices() {
    // Update elements vertex.
    for (auto& [_, element] : elements) {
        JST_CHECK(updateElementVertex(element));
    }

    // Set flag to update buffers.
    updateFontPosVerticesBufferFlag = true;
    updateFontFillVerticesBufferFlag = true;

    return Result::SUCCESS;
}

Result Text::Impl::updateInstances() {
    // Update elements instane.
    for (auto& [_, element] : elements) {
        JST_CHECK(updateElementInstance(element));
    }

    // Set flag to update buffer.
    updateFontInstanceBufferFlag = true;

    return Result::SUCCESS;
}

Result Text::Impl::updateIndices() {
    // Populate indices.
    for (U64 i = 0; i < config.maxCharacters; i++) {
        indices[(i * 6) + 0] = (i * 4);
        indices[(i * 6) + 1] = (i * 4) + 1;
        indices[(i * 6) + 2] = (i * 4) + 2;
        indices[(i * 6) + 3] = (i * 4) + 2;
        indices[(i * 6) + 4] = (i * 4) + 3;
        indices[(i * 6) + 5] = (i * 4);
    }

    // Set flag to update buffer.
    updateFontIndicesBufferFlag = true;

    return Result::SUCCESS;
}

Result Text::Impl::updateElementInstance(Element& element) {
    // Reference transform.
    auto& transform = element.instances[0];

    // Reset transform.
    transform = glm::mat4(1.0f);

    // Translate to screen position.
    transform = glm::translate(transform, glm::vec3(element.config.position.x, element.config.position.y, 0.0f));

    // Scale to pixel size.
    transform = glm::scale(transform, glm::vec3(config.pixelSize.x, config.pixelSize.y, 1.0f));

    // Scale font.
    transform = glm::scale(transform, glm::vec3(element.config.scale, element.config.scale, 1.0f));

    // Rotate.
    transform = glm::rotate(transform, glm::radians(element.config.rotationDeg), glm::vec3(0.0f, 0.0f, 1.0f));

    // Horizontal center.
    if (element.config.alignment.x) {
        if (element.config.alignment.x == 1) {
            transform = glm::translate(transform, glm::vec3(-element.bounds.x / 2.0f, 0.0f, 0.0f));
        }
        if (element.config.alignment.x == 2) {
            transform = glm::translate(transform, glm::vec3(-element.bounds.x, 0.0f, 0.0f));
        }
    }

    // Vertical center.
    if (element.config.alignment.y) {
        if (element.config.alignment.y == 1) {
            transform = glm::translate(transform, glm::vec3(0.0f, element.bounds.y / 2.0f, 0.0f));
        }
        if (element.config.alignment.y == 2) {
            transform = glm::translate(transform, glm::vec3(0.0f, element.bounds.y, 0.0f));
        }
    }

    return Result::SUCCESS;
}

Result Text::Impl::updateElementVertex(Element& element) {
    // Check config.

    if (element.config.fill.empty()) {
        return Result::SUCCESS;
    }

    if (element.config.fill.size() > config.maxCharacters) {
        JST_ERROR("[TEXT] Text too long ({} characters). Increase the max size.", element.config.fill.size());
        return Result::ERROR;
    }

    // Clear buffers.
    std::fill(element.posVertices.begin(), element.posVertices.end(), glm::vec2(0.0f));
    std::fill(element.fillVertices.begin(), element.fillVertices.end(), glm::vec2(0.0f));

    // Recalculate vertex buffer.

    F32 x = 0.0f;
    F32 y = 0.0f;

    I32 minx = 0;
    I32 miny = 0;

    for (const auto& c : element.config.fill) {
        if (c >= 32 && c < 127) {
            if (c == ' ') {
                continue;
            }

            // TODO: Check if there is no better way to do this.
            const auto& b = config.font->glyph(c - 32);
            minx = std::min(minx, static_cast<I32>(x + b.xOffset));
            miny = std::max(miny, static_cast<I32>(y - b.yOffset));
        }
    }

    for (U64 i = 0; i < element.config.fill.size(); i++) {
        const auto& fontSize = config.font->getConfig().size;
        const auto& atlasSize = config.font->atlasSize();
        const auto& c = element.config.fill[i];

        if (c >= 32 && c < 127) {
            if (c == ' ') {
                x += (fontSize / 2.0f);
                continue;
            }

            const auto& b = config.font->glyph(c - 32);

            F32 x0 = x + b.xOffset - minx;
            F32 y0 = y - b.yOffset - miny;
            F32 x1 = x0 + (b.x1 - b.x0);
            F32 y1 = y0 - (b.y1 - b.y0);

            // Add positions.

            element.posVertices[(i * 4) + 0] = glm::vec2(x0, y0);
            element.posVertices[(i * 4) + 1] = glm::vec2(x1, y0);
            element.posVertices[(i * 4) + 2] = glm::vec2(x1, y1);
            element.posVertices[(i * 4) + 3] = glm::vec2(x0, y1);

            // Normalize texture coordinates.

            F32 s0 = b.x0 / static_cast<F32>(atlasSize.x);
            F32 t0 = b.y0 / static_cast<F32>(atlasSize.y);
            F32 s1 = b.x1 / static_cast<F32>(atlasSize.x);
            F32 t1 = b.y1 / static_cast<F32>(atlasSize.y);

            // Add texture coordinates.

            element.fillVertices[(i * 4) + 0] = glm::vec2(s0, t0);
            element.fillVertices[(i * 4) + 1] = glm::vec2(s1, t0);
            element.fillVertices[(i * 4) + 2] = glm::vec2(s1, t1);
            element.fillVertices[(i * 4) + 3] = glm::vec2(s0, t1);

            // Update horizontal position.

            x += b.xAdvance;

            // Save text height.

            element.bounds.y = std::max(element.bounds.y, static_cast<I32>(b.y1 - b.y0));
        }
    }

    element.bounds.x = x;

    return Result::SUCCESS;
}

Result Text::present() {
    if (pimpl->updateFontFillVerticesBufferFlag) {
        pimpl->fontFillVerticesBuffer->update();
        pimpl->updateFontFillVerticesBufferFlag = false;
    }

    if (pimpl->updateFontPosVerticesBufferFlag) {
        pimpl->fontPosVerticesBuffer->update();
        pimpl->updateFontPosVerticesBufferFlag = false;
    }

    if (pimpl->updateFontIndicesBufferFlag) {
        pimpl->fontIndicesBuffer->update();
        pimpl->updateFontIndicesBufferFlag = false;
    }

    if (pimpl->updateFontUniformBufferFlag) {
        pimpl->fontUniformBuffer->update();
        pimpl->updateFontUniformBufferFlag = false;
    }

    if (pimpl->updateFontInstanceBufferFlag) {
        pimpl->fontInstanceBuffer->update();
        pimpl->updateFontInstanceBufferFlag = false;
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render::Components
