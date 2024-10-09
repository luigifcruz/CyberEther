#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

#define STBTT_STATIC
#define STB_TRUETYPE_IMPLEMENTATION
#include "jetstream/render/tools/imstb_truetype.h"

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include "jetstream/render/base.hh"
#include "jetstream/render/components/font.hh"

namespace Jetstream::Render::Components {

Font::Font(const Config& config) {
    this->config = config;
    this->pimpl = std::make_unique<Impl>();
}

Font::~Font() {
    pimpl.reset();
}

struct Font::Impl {
    Window* window;

    // Font.

    stbtt_fontinfo font;

    // Font atlas.

    I32 atlasPadding = 4;
    U8  atlasOneEdgeValue = 128;
    F32 atlasPixelDistScale = 16.0f;
    Extent2D<I32> atlasSize = {512, 512};
    I32 ascent, descent;

    // Texture.

    std::shared_ptr<Render::Texture> fontAtlasTexture;
};

Result Font::create(Window* window) {
    JST_DEBUG("[FONT] Loading new font.");

    // Load variables.

    pimpl->window = window;

    // Decompress font.

    const U8* compressedData = static_cast<const uint8_t*>(config.data);
    const U32 decompressedSize = stb_decompress_length(compressedData);
    std::vector<uint8_t> decompressedData(decompressedSize);
    stb_decompress(decompressedData.data(), compressedData, decompressedSize);

    // Load font.

    const auto fontOffset = stbtt_GetFontOffsetForIndex(decompressedData.data(), 0);
    if (!stbtt_InitFont(&pimpl->font, decompressedData.data(), fontOffset)) {
        JST_ERROR("[FONT] Failed to load font.");
        return Result::ERROR;
    }

    JST_DEBUG("[FONT] Loaded new font.");
    decompressedData.clear();

    // Calculate ascent and descent.

    stbtt_GetFontVMetrics(&pimpl->font, &pimpl->ascent, &pimpl->descent, nullptr);
    const auto scale = stbtt_ScaleForPixelHeight(&pimpl->font, config.size);
    pimpl->ascent = roundf(pimpl->ascent * scale);
    pimpl->descent = roundf(pimpl->descent * scale);

    // Create font atlas.

    std::vector<uint8_t> atlas(pimpl->atlasSize.x * pimpl->atlasSize.y);

    int x = 0;
    int y = 0;
    int maxHeight = 0;

    for (int ch = 32; ch < 128; ch++) {
        uint8_t* sdf;
        int width, height, xoffset, yoffset;

        if (!(sdf = stbtt_GetCodepointSDF(&pimpl->font,
                                          scale,
                                          ch,
                                          pimpl->atlasPadding,
                                          pimpl->atlasOneEdgeValue,
                                          pimpl->atlasPixelDistScale,
                                          &width,
                                          &height,
                                          &xoffset,
                                          &yoffset))) {
            continue;
        }

        if (x + width >= pimpl->atlasSize.x) {
            x = 0;
            y += maxHeight + 1;
            maxHeight = 0;
        }

        if (y + height >= pimpl->atlasSize.y) {
            break;
        }

        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                atlas[(y + j) * pimpl->atlasSize.x + (x + i)] = sdf[j * width + i];
            }
        }

        glyphs[ch - 32] = {
            .x0 = x,
            .y0 = y,
            .x1 = x + width,
            .y1 = y + height,
            .xOffset = static_cast<F32>(xoffset),
            .yOffset = static_cast<F32>(yoffset),
            .xAdvance = static_cast<F32>(width)
        };

        x += width + 1;
        if (height > maxHeight) {
            maxHeight = height;
        }

        stbtt_FreeSDF(sdf, nullptr);
    }

    JST_DEBUG("[FONT] Created font atlas.");

    // Create texture.

    {
        Render::Texture::Config cfg;
        cfg.size = {
            static_cast<U64>(pimpl->atlasSize.x),
            static_cast<U64>(pimpl->atlasSize.y),
        };
        cfg.buffer = atlas.data();
        cfg.dfmt = Render::Texture::DataFormat::UI8;
        cfg.pfmt = Render::Texture::PixelFormat::RED;
        cfg.ptype = Render::Texture::PixelType::UI8;
        JST_CHECK(window->build(pimpl->fontAtlasTexture, cfg));
        JST_CHECK(window->bind(pimpl->fontAtlasTexture));
    }

    JST_DEBUG("[FONT] Created font atlas texture.");

    return Result::SUCCESS;
}

Result Font::destroy(Window* window) {
    JST_DEBUG("[FONT] Destroying font.");

    JST_CHECK(window->unbind(pimpl->fontAtlasTexture));

    return Result::SUCCESS;
}

const Font::Glyph& Font::glyph(const I32& code) const {
    return glyphs.at(code);
}

const std::shared_ptr<Render::Texture>& Font::atlas() const {
    return pimpl->fontAtlasTexture;
}

const Extent2D<I32>& Font::atlasSize() const {
    return pimpl->atlasSize;
}

}  // namespace Jetstream::Render::Components
