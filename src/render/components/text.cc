#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "jetstream/render/base.hh"
#include "jetstream/render/components/text.hh"

#include "shaders/global_shaders.hh"

namespace Jetstream::Render::Components {

Text::Text(const Config& config) {
    this->config = config;
    this->pimpl = std::make_unique<Impl>();
}

Text::~Text() {
    pimpl.reset();
}

struct Text::Impl {
    std::shared_ptr<Components::Font> font;

    bool fillDidChange = false;
    bool transformDidChange = false;

    bool updateFontUniformBufferFlag = false;
    bool updateFontPosVerticiesBufferFlag = false;
    bool updateFontFillVerticesBufferFlag = false;
    bool updateFontIndicesBufferFlag = false;

    I32 textWidth = 0;
    I32 textHeight = 0;

    struct {
        glm::mat4 transform;
        glm::vec3 color;
    } uniforms;

    std::vector<glm::vec2> posVertices;
    std::vector<glm::vec2> fillVertices;
    std::vector<U32> indices;

    std::shared_ptr<Render::Buffer> fontUniformBuffer;
    std::shared_ptr<Render::Buffer> fontPosVerticesBuffer;
    std::shared_ptr<Render::Buffer> fontFillVerticesBuffer;
    std::shared_ptr<Render::Buffer> fontIndicesBuffer;

    std::shared_ptr<Render::Vertex> fontVertex;

    std::shared_ptr<Render::Draw> drawFontVertex;

    std::shared_ptr<Render::Program> fontProgram;
};

Result Text::create(Window* window) {
    JST_DEBUG("[TEXT] Loading new text.");

    // Load variables.

    pimpl->font = config.font;

    // Check if font is loaded.

    if (!pimpl->font) {
        JST_ERROR("[TEXT] Font not loaded.");
        return Result::ERROR;
    }

    // Reserve memory.

    pimpl->posVertices.resize(config.maxCharacters * 4);
    pimpl->fillVertices.resize(config.maxCharacters * 4);
    pimpl->indices.resize(config.maxCharacters * 6);

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
        cfg.buffer = pimpl->indices.data();
        cfg.elementByteSize = sizeof(U32);
        cfg.size = pimpl->indices.size();
        cfg.target = Render::Buffer::Target::VERTEX_INDICES;
        JST_CHECK(window->build(pimpl->fontIndicesBuffer, cfg));
        JST_CHECK(window->bind(pimpl->fontIndicesBuffer));
    }

    {
        Render::Vertex::Config cfg;
        cfg.buffers = {
            {pimpl->fontPosVerticesBuffer, 2},
            {pimpl->fontFillVerticesBuffer, 2},
        };
        cfg.indices = pimpl->fontIndicesBuffer;
        JST_CHECK(window->build(pimpl->fontVertex, cfg));
    }

    {
        Render::Draw::Config cfg;
        cfg.buffer = pimpl->fontVertex;
        cfg.mode = Render::Draw::Mode::TRIANGLES;
        JST_CHECK(window->build(pimpl->drawFontVertex, cfg));
    }

    {
        Render::Program::Config cfg;
        cfg.shaders = GlobalShadersPackage["text"];
        cfg.draw = pimpl->drawFontVertex;
        cfg.textures = {
            pimpl->font->atlas(),
        };
        cfg.buffers = {
            {pimpl->fontUniformBuffer, Render::Program::Target::VERTEX |
                                       Render::Program::Target::FRAGMENT},
        };
        cfg.enableAlphaBlending = true;
        JST_CHECK(window->build(pimpl->fontProgram, cfg));
    }

    // Update.

    JST_CHECK(updateVertices());
    JST_CHECK(updateTransform());

    return Result::SUCCESS;
}

Result Text::destroy(Window* window) {
    JST_DEBUG("[TEXT] Unloading text.");

    JST_CHECK(window->unbind(pimpl->fontUniformBuffer));
    JST_CHECK(window->unbind(pimpl->fontPosVerticesBuffer));
    JST_CHECK(window->unbind(pimpl->fontFillVerticesBuffer));
    JST_CHECK(window->unbind(pimpl->fontIndicesBuffer));

    return Result::SUCCESS;
}

Result Text::surface(Render::Surface::Config& config) {
    JST_DEBUG("[TEXT] Binding text to surface.");

    config.programs.push_back(pimpl->fontProgram);

    return Result::SUCCESS;
}

const F32& Text::scale(const F32& scale) {
    if (config.scale != scale) {
        config.scale = scale;
        pimpl->transformDidChange = true;
    }

    return config.scale;
}

const Extent2D<F32>& Text::position(const Extent2D<F32>& position) {
    if (config.position != position) {
        config.position = position;
        pimpl->transformDidChange = true;
    }

    return config.position;
}

const Extent2D<F32>& Text::pixelSize(const Extent2D<F32>& pixelSize) {
    if (config.pixelSize != pixelSize) {
        config.pixelSize = pixelSize;
        pimpl->transformDidChange = true;
    }

    return config.pixelSize;
}

const Extent2D<bool>& Text::center(const Extent2D<bool>& center) {
    if (config.center != center) {
        config.center = center;
        pimpl->transformDidChange = true;
    }

    return config.center;
}

const ColorRGBA<F32>& Text::color(const ColorRGBA<F32>& color) {
    if (config.color != color) {
        config.color = color;
        pimpl->fillDidChange = true;
    }

    return config.color;
}

const F32& Text::rotationDeg(const F32& rotationDeg) {
    if (config.rotationDeg != rotationDeg) {
        config.rotationDeg = rotationDeg;
        pimpl->transformDidChange = true;
    }

    return config.rotationDeg;
}

const std::string& Text::fill(const std::string& text) {
    if (config.fill != text) {
        config.fill = text;
        pimpl->fillDidChange = true;
    }

    return config.fill;
}

Result Text::apply() {
    if (pimpl->fillDidChange) {
        JST_CHECK(updateVertices());

        if (config.center.x || config.center.y) {
            JST_CHECK(updateTransform());
            pimpl->transformDidChange = false;
        }

        pimpl->fillDidChange = false;
    }

    if (pimpl->transformDidChange) {
        JST_CHECK(updateTransform());

        pimpl->transformDidChange = false;
    }

    return Result::SUCCESS;
}

Result Text::updateTransform() {
    auto transform = glm::mat4(1.0f);

    // Translate to screen position.
    transform = glm::translate(transform, glm::vec3(config.position.x, config.position.y, 0.0f));

    // Scale font.
    transform = glm::scale(transform, glm::vec3(config.scale, config.scale, 1.0f));

    // Rotate.
    transform = glm::rotate(transform, glm::radians(config.rotationDeg), glm::vec3(0.0f, 0.0f, 1.0f));

    // Scale to pixel size.
    transform = glm::scale(transform, glm::vec3(config.pixelSize.x, config.pixelSize.y, 1.0f));

    // Horizontal center.
    if (config.center.x) {
        transform = glm::translate(transform, glm::vec3(-pimpl->textWidth / 2.0f, 0.0f, 0.0f));
    }

    // Vertical center.
    if (config.center.y) {
        transform = glm::translate(transform, glm::vec3(0.0f, pimpl->textHeight / 2.0f, 0.0f));
    }

    // Save transform.
    pimpl->uniforms.transform = transform;
    pimpl->uniforms.color = glm::vec3(config.color.r, config.color.g, config.color.b);

    // Update uniform buffer.
    pimpl->updateFontUniformBufferFlag = true;

    return Result::SUCCESS;
}

Result Text::updateVertices() {
    // Check config.

    if (config.fill.empty()) {
        return Result::SUCCESS;
    }

    if (config.fill.size() >= config.maxCharacters) {
        JST_ERROR("[TEXT] Text too long ({} characters). Increase the max size.", config.fill.size());
        return Result::ERROR;
    }

    // Clear buffers.

    std::fill(pimpl->posVertices.begin(), pimpl->posVertices.end(), glm::vec2(0.0f));
    std::fill(pimpl->fillVertices.begin(), pimpl->fillVertices.end(), glm::vec2(0.0f));
    std::fill(pimpl->indices.begin(), pimpl->indices.end(), 0);

    // Recalculate vertex buffer.

    F32 x = 0.0f;
    F32 y = 0.0f;

    I32 minx = 0;
    I32 miny = 0;

    for (const auto& c : config.fill) {
        if (c >= 32 && c < 127) {
            if (c == ' ') {
                continue;
            }

            const auto& b = pimpl->font->glyph(c - 32);
            minx = std::min(minx, static_cast<I32>(x + b.xOffset));
            miny = std::max(miny, static_cast<I32>(y - b.yOffset));
        }
    }

    JST_INFO("-> {}: {} {}", config.fill, minx, miny);

    for (U64 i = 0; i < config.fill.size(); i++) {
        const auto& fontSize = pimpl->font->getConfig().size;
        const auto& atlasSize = pimpl->font->atlasSize();
        const auto& c = config.fill[i];

        if (c >= 32 && c < 127) {
            if (c == ' ') {
                x += (fontSize / 2.0f);
                continue;
            }

            const auto& b = pimpl->font->glyph(c - 32);

            F32 x0 = x + b.xOffset - minx;
            F32 y0 = y - b.yOffset - miny;
            F32 x1 = x0 + (b.x1 - b.x0);
            F32 y1 = y0 - (b.y1 - b.y0);

            JST_INFO("-> {}: {} {} {}", c, x0, y0, b.xOffset);

            // Add positions.

            pimpl->posVertices[(i * 4) + 0] = glm::vec2(x0, y0);
            pimpl->posVertices[(i * 4) + 1] = glm::vec2(x1, y0);
            pimpl->posVertices[(i * 4) + 2] = glm::vec2(x1, y1);
            pimpl->posVertices[(i * 4) + 3] = glm::vec2(x0, y1);

            // Normalize texture coordinates.

            F32 s0 = b.x0 / static_cast<F32>(atlasSize.x);
            F32 t0 = b.y0 / static_cast<F32>(atlasSize.y);
            F32 s1 = b.x1 / static_cast<F32>(atlasSize.x);
            F32 t1 = b.y1 / static_cast<F32>(atlasSize.y);

            // Add texture coordinates.

            pimpl->fillVertices[(i * 4) + 0] = glm::vec2(s0, t0);
            pimpl->fillVertices[(i * 4) + 1] = glm::vec2(s1, t0);
            pimpl->fillVertices[(i * 4) + 2] = glm::vec2(s1, t1);
            pimpl->fillVertices[(i * 4) + 3] = glm::vec2(s0, t1);

            // Add indices for two triangles.

            pimpl->indices[(i * 6) + 0] = (i * 4);
            pimpl->indices[(i * 6) + 1] = (i * 4) + 1;
            pimpl->indices[(i * 6) + 2] = (i * 4) + 2;
            pimpl->indices[(i * 6) + 3] = (i * 4) + 2;
            pimpl->indices[(i * 6) + 4] = (i * 4) + 3;
            pimpl->indices[(i * 6) + 5] = (i * 4);

            // Update horizontal position.

            x += b.xAdvance;

            // Save text height.

            pimpl->textHeight = std::max(pimpl->textHeight, static_cast<I32>(b.y1 - b.y0));
        }
    }

    pimpl->textWidth = x;

    // Update buffers.

    pimpl->updateFontPosVerticiesBufferFlag = true;
    pimpl->updateFontFillVerticesBufferFlag = true;
    pimpl->updateFontIndicesBufferFlag = true;

    // TODO: Implement partial drawings.

    return Result::SUCCESS;
}

Result Text::present() {
    if (pimpl->updateFontFillVerticesBufferFlag) {
        pimpl->fontFillVerticesBuffer->update();
        pimpl->updateFontFillVerticesBufferFlag = false;
    }

    if (pimpl->updateFontPosVerticiesBufferFlag) {
        pimpl->fontPosVerticesBuffer->update();
        pimpl->updateFontPosVerticiesBufferFlag = false;
    }

    if (pimpl->updateFontIndicesBufferFlag) {
        pimpl->fontIndicesBuffer->update();
        pimpl->updateFontIndicesBufferFlag = false;
    }

    if (pimpl->updateFontUniformBufferFlag) {
        pimpl->fontUniformBuffer->update();
        pimpl->updateFontUniformBufferFlag = false;
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render::Components
