#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>

#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "jetstream/render/base.hh"
#include "jetstream/render/components/text.hh"
#include "jetstream/tools/numeric.hh"

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
    static std::vector<F32> GlyphAdvances(const Font* font, const std::string& fill, F32 scale) {
        std::vector<F32> result(fill.size(), 0.0f);
        if (!font) {
            return result;
        }
        for (U64 i = 0; i < fill.size(); ++i) {
            const char c = fill[i];
            if (c < 32 || c >= 127) {
                continue;
            }
            result[i] = std::round(font->glyph(c - 32).xAdvance * scale);
        }
        return result;
    }

    static F32 ScaledLineHeight(const Font* font, F32 scale) {
        if (!font) {
            return 0.0f;
        }
        return std::max(1.0f, std::round(static_cast<F32>(font->lineHeight()) * scale));
    }

    struct UniformBuffer {
        glm::vec3 color;
        F32 sharpness;
        F32 atlasPixelRange;
        F32 padding[3];
    };

    struct InstanceData {
        glm::mat4 transform;
        glm::vec4 color;
    };

    struct Element {
        U64 characterCount = 0;
        const ElementConfig& config;

        Extent2D<I32> bounds;
        std::span<glm::vec2> posVertices;
        std::span<glm::vec2> fillVertices;
        std::span<InstanceData> instances;
    };

    // Variables.

    const Config& config;
    UniformBuffer uniforms;
    std::unordered_map<std::string, Element> elements;
    U64 vertexCount = 0;

    // Render.

    bool updateFontUniformBufferFlag = false;
    bool updateFontPosVerticesBufferFlag = false;
    bool updateFontFillVerticesBufferFlag = false;
    bool updateFontInstanceBufferFlag = false;
    bool updateFontIndicesBufferFlag = false;
    bool updateVertexCountFlag = false;

    std::vector<glm::vec2> posVertices;
    std::vector<glm::vec2> fillVertices;
    std::vector<InstanceData> instances;
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
    Result refreshVertexCount();

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
    U64 numberOfVertices = 0;
    const U64 numberOfInstances = config.elements.size();
    U64 numberOfIndices = 0;
    U64 totalVertexCount = 0;
    if (!detail::CheckedMultiply(config.maxCharacters, 4,
                                 numberOfVertices) ||
        !detail::CheckedMultiply(config.maxCharacters, 6,
                                 numberOfIndices) ||
        !detail::CheckedMultiply(numberOfVertices, numberOfInstances,
                                 totalVertexCount) ||
        numberOfVertices > std::numeric_limits<U32>::max() ||
        numberOfInstances > std::numeric_limits<U32>::max() ||
        totalVertexCount > pimpl->posVertices.max_size() ||
        totalVertexCount > pimpl->fillVertices.max_size() ||
        numberOfInstances > pimpl->instances.max_size() ||
        numberOfIndices > pimpl->indices.max_size()) {
        JST_ERROR("[TEXT] Geometry exceeds the supported rendering range.");
        return Result::ERROR;
    }

    // Reserve memory.
    try {
        pimpl->posVertices.resize(totalVertexCount);
        pimpl->fillVertices.resize(totalVertexCount);
        pimpl->instances.resize(numberOfInstances);
        pimpl->indices.resize(numberOfIndices);
    } catch (const std::bad_alloc&) {
        JST_ERROR("[TEXT] Failed to allocate text geometry.");
        return Result::ERROR;
    } catch (const std::length_error&) {
        JST_ERROR("[TEXT] Text geometry exceeds container capacity.");
        return Result::ERROR;
    }

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
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = pimpl->posVertices.data();
        cfg.elementByteSize = sizeof(glm::vec2);
        cfg.size = pimpl->posVertices.size();
        cfg.target = Render::Buffer::Target::VERTEX;
        JST_CHECK(window->build(pimpl->fontPosVerticesBuffer, cfg));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = pimpl->fillVertices.data();
        cfg.elementByteSize = sizeof(glm::vec2);
        cfg.size = pimpl->fillVertices.size();
        cfg.target = Render::Buffer::Target::VERTEX;
        JST_CHECK(window->build(pimpl->fontFillVerticesBuffer, cfg));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = pimpl->instances.data();
        cfg.elementByteSize = sizeof(Impl::InstanceData);
        cfg.size = pimpl->instances.size();
        cfg.target = Render::Buffer::Target::VERTEX;
        JST_CHECK(window->build(pimpl->fontInstanceBuffer, cfg));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = pimpl->indices.data();
        cfg.elementByteSize = sizeof(U32);
        cfg.size = pimpl->indices.size();
        cfg.target = Render::Buffer::Target::VERTEX_INDICES;
        JST_CHECK(window->build(pimpl->fontIndicesBuffer, cfg));
    }

    {
        Render::Vertex::Config cfg;
        cfg.vertices = {
            {pimpl->fontPosVerticesBuffer, 2},
            {pimpl->fontFillVerticesBuffer, 2},
        };
        cfg.instances = {
            {pimpl->fontInstanceBuffer, sizeof(Impl::InstanceData) / sizeof(F32)},
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

Result Text::destroy(Window*) {
    JST_DEBUG("[TEXT] Unloading text.");

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
    const bool shouldUpdateVertices = updatedElement.fill != currentElement.fill ||
                                      updatedElement.scale != currentElement.scale;
    const bool shouldUpdateInstance = shouldUpdateVertices ||
                                      updatedElement.scale != currentElement.scale ||
                                      updatedElement.position != currentElement.position ||
                                      updatedElement.alignment != currentElement.alignment ||
                                      updatedElement.rotationDeg != currentElement.rotationDeg ||
                                      updatedElement.color != currentElement.color;

    if (!shouldUpdateInstance) {
        return Result::SUCCESS;
    }

    // Update element data.
    currentElement = updatedElement;

    // Load element data.
    auto& element = pimpl->elements.at(elementId);

    // Update element vertex.
    if (shouldUpdateVertices) {
        JST_CHECK(pimpl->updateElementVertex(element));
        JST_CHECK(pimpl->refreshVertexCount());

        // Set flag to update buffers.
        pimpl->updateFontPosVerticesBufferFlag = true;
        pimpl->updateFontFillVerticesBufferFlag = true;
    }

    // Update element instance.
    if (shouldUpdateInstance) {
        JST_CHECK(pimpl->updateElementInstance(element));

        // Set flag to update buffer.
        pimpl->updateFontInstanceBufferFlag = true;
    }

    return Result::SUCCESS;
}

F32 Text::advance(const std::string& fill, F32 scale) const {
    const auto perGlyph = advances(fill, scale);

    F32 x = 0.0f;
    F32 maxWidth = 0.0f;
    for (U64 i = 0; i < fill.size(); ++i) {
        if (fill[i] == '\n') {
            maxWidth = std::max(maxWidth, x);
            x = 0.0f;
            continue;
        }
        x += perGlyph[i];
    }

    return std::max(maxWidth, x);
}

std::vector<F32> Text::advances(const std::string& fill, F32 scale) const {
    return Impl::GlyphAdvances(config.font.get(), fill, scale);
}

F32 Text::lineHeight(F32 scale) const {
    return Impl::ScaledLineHeight(config.font.get(), scale);
}

Result Text::updatePixelSize(const Extent2D<F32>& pixelSize) {
    // Check if pixel size has changed. Use epsilon to avoid tiny float jitter.
    constexpr F32 pixelSizeEpsilon = 1e-6f;
    const bool shouldUpdateVertices =
        std::abs(config.pixelSize.x - pixelSize.x) > pixelSizeEpsilon ||
        std::abs(config.pixelSize.y - pixelSize.y) > pixelSizeEpsilon;

    if (!shouldUpdateVertices) {
        return Result::SUCCESS;
    }

    // Update pixel size.
    config.pixelSize = pixelSize;

    // Update elements vertices and instances.
    if (shouldUpdateVertices) {
        JST_CHECK(pimpl->updateVertices());
        JST_CHECK(pimpl->updateInstances());
    }

    return Result::SUCCESS;
}

Result Text::updateScissorRect(const std::optional<Render::ScissorRect>& rect) {
    pimpl->fontProgram->scissorRect(rect);
    return Result::SUCCESS;
}

Result Text::Impl::updateUniforms() {
    // Set data.
    uniforms.color = glm::vec3(config.color.r, config.color.g, config.color.b);
    uniforms.sharpness = config.sharpness;
    uniforms.atlasPixelRange = config.font->atlasPixelRange();

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

    JST_CHECK(refreshVertexCount());

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

Result Text::Impl::refreshVertexCount() {
    U64 maxCharacterCount = 0;
    for (const auto& [_, element] : elements) {
        maxCharacterCount = std::max(maxCharacterCount, element.characterCount);
    }

    const U64 nextVertexCount = maxCharacterCount * 6;
    if (nextVertexCount == vertexCount) {
        return Result::SUCCESS;
    }

    vertexCount = nextVertexCount;
    updateVertexCountFlag = true;
    return Result::SUCCESS;
}

Result Text::Impl::updateElementInstance(Element& element) {
    auto& instance = element.instances[0];
    auto& transform = instance.transform;

    glm::vec2 alignment(0.0f, 0.0f);
    if (element.config.alignment.x == 1) {
        alignment.x = -static_cast<F32>(element.bounds.x) / 2.0f;
    } else if (element.config.alignment.x == 2) {
        alignment.x = -static_cast<F32>(element.bounds.x);
    }
    if (element.config.alignment.y == 1) {
        alignment.y = static_cast<F32>(element.bounds.y) / 2.0f;
    } else if (element.config.alignment.y == 2) {
        alignment.y = static_cast<F32>(element.bounds.y);
    }

    glm::vec2 origin(element.config.position.x, element.config.position.y);
    const bool snap = element.config.rotationDeg == 0.0f &&
                      config.pixelSize.x > 0.0f &&
                      config.pixelSize.y > 0.0f;
    if (snap) {
        const F32 column = std::round((origin.x + 1.0f) / config.pixelSize.x + alignment.x);
        const F32 row = std::round((1.0f - origin.y) / config.pixelSize.y - alignment.y);
        origin = glm::vec2(-1.0f + column * config.pixelSize.x, 1.0f - row * config.pixelSize.y);
        alignment = glm::vec2(0.0f, 0.0f);
    }

    transform = glm::mat4(1.0f);
    transform = glm::translate(transform, glm::vec3(origin, 0.0f));
    transform = glm::scale(transform, glm::vec3(config.pixelSize.x, config.pixelSize.y, 1.0f));
    transform = glm::rotate(transform, glm::radians(element.config.rotationDeg), glm::vec3(0.0f, 0.0f, 1.0f));
    transform = glm::translate(transform, glm::vec3(alignment, 0.0f));

    const auto color = element.config.color.value_or(config.color);
    instance.color = glm::vec4(color.r, color.g, color.b, color.a);

    return Result::SUCCESS;
}

Result Text::Impl::updateElementVertex(Element& element) {
    std::fill(element.posVertices.begin(), element.posVertices.end(), glm::vec2(0.0f));
    std::fill(element.fillVertices.begin(), element.fillVertices.end(), glm::vec2(0.0f));

    element.characterCount = 0;
    element.bounds = {0, 0};

    if (element.config.fill.empty()) {
        return Result::SUCCESS;
    }

    U64 renderableCharacterCount = 0;
    for (const auto c : element.config.fill) {
        if (c >= 32 && c < 127 && c != ' ') {
            ++renderableCharacterCount;
        }
    }

    if (renderableCharacterCount > config.maxCharacters) {
        JST_ERROR("[TEXT] Text too long ({} rendered characters). Increase the max size.",
                  renderableCharacterCount);
        return Result::ERROR;
    }

    const auto& font = *config.font;
    const auto& atlasSize = font.atlasSize();
    const F32 scale = element.config.scale;
    const F32 texel = scale / font.atlasScale();
    const F32 lineHeight = ScaledLineHeight(config.font.get(), scale);
    const F32 ascent = std::round(static_cast<F32>(font.ascent()) * scale);
    const auto perGlyph = GlyphAdvances(config.font.get(), element.config.fill, scale);

    std::vector<F32> lineWidths(1, 0.0f);
    for (U64 i = 0; i < element.config.fill.size(); ++i) {
        if (element.config.fill[i] == '\n') {
            lineWidths.push_back(0.0f);
        } else {
            lineWidths.back() += perGlyph[i];
        }
    }
    const F32 blockWidth = *std::max_element(lineWidths.begin(), lineWidths.end());
    const U64 lineCount = lineWidths.size();

    U64 lineIndex = 0;
    F32 x = std::round((blockWidth - lineWidths[lineIndex]) * 0.5f);
    F32 y = 0.0f;

    for (U64 i = 0; i < element.config.fill.size(); ++i) {
        const char c = element.config.fill[i];
        if (c == '\n') {
            ++lineIndex;
            x = std::round((blockWidth - lineWidths[lineIndex]) * 0.5f);
            y -= lineHeight;
            continue;
        }

        if (c < 32 || c >= 127) {
            continue;
        }

        const auto& b = font.glyph(c - 32);
        const F32 advance = perGlyph[i];

        if (c == ' ') {
            x += advance;
            continue;
        }

        const F32 x0 = x + static_cast<F32>(b.xOffset) * scale;
        const F32 y0 = y - ascent - static_cast<F32>(b.yOffset) * scale;
        const F32 x1 = x0 + static_cast<F32>(b.x1 - b.x0) * texel;
        const F32 y1 = y0 - static_cast<F32>(b.y1 - b.y0) * texel;
        const U64 base = element.characterCount * 4;

        element.posVertices[base + 0] = glm::vec2(x0, y0);
        element.posVertices[base + 1] = glm::vec2(x1, y0);
        element.posVertices[base + 2] = glm::vec2(x1, y1);
        element.posVertices[base + 3] = glm::vec2(x0, y1);

        const F32 s0 = static_cast<F32>(b.x0) / static_cast<F32>(atlasSize.x);
        const F32 t0 = static_cast<F32>(b.y0) / static_cast<F32>(atlasSize.y);
        const F32 s1 = static_cast<F32>(b.x1) / static_cast<F32>(atlasSize.x);
        const F32 t1 = static_cast<F32>(b.y1) / static_cast<F32>(atlasSize.y);

        element.fillVertices[base + 0] = glm::vec2(s0, t0);
        element.fillVertices[base + 1] = glm::vec2(s1, t0);
        element.fillVertices[base + 2] = glm::vec2(s1, t1);
        element.fillVertices[base + 3] = glm::vec2(s0, t1);

        x += advance;
        element.characterCount++;
    }

    element.bounds.x = static_cast<I32>(blockWidth);
    element.bounds.y = static_cast<I32>(lineHeight * static_cast<F32>(lineCount));

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

    if (pimpl->updateVertexCountFlag) {
        JST_TRACE("[TEXT] Vertex optimization: {}/{}.", pimpl->vertexCount,
                  config.maxCharacters * 6);
        JST_CHECK(pimpl->drawFont->updateVertexCount(pimpl->vertexCount));
        pimpl->updateVertexCountFlag = false;
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render::Components
