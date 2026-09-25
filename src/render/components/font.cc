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

#include <algorithm>
#include <vector>

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

    F32 atlasOversample = 2.0f;
    I32 atlasPadding = 8;
    U8  atlasOneEdgeValue = 128;
    F32 atlasPixelDistScale = 8.0f;
    I32 atlasWidth = 1024;
    Extent2D<I32> atlasSize = {0, 0};
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

    const auto bakeScale = stbtt_ScaleForPixelHeight(&pimpl->font, config.size * pimpl->atlasOversample);

    struct Baked {
        I32 code;
        U8* sdf;
        I32 width;
        I32 height;
        I32 xOffset;
        I32 yOffset;
    };
    std::vector<Baked> baked;

    for (int ch = 32; ch < 128; ch++) {
        int advanceWidth, leftSideBearing;
        stbtt_GetCodepointHMetrics(&pimpl->font, ch, &advanceWidth, &leftSideBearing);

        glyphs[ch - 32] = {
            .x0 = 0,
            .y0 = 0,
            .x1 = 0,
            .y1 = 0,
            .xOffset = 0.0f,
            .yOffset = 0.0f,
            .xAdvance = static_cast<F32>(advanceWidth * scale)
        };

        Baked glyph{.code = ch - 32};
        glyph.sdf = stbtt_GetCodepointSDF(&pimpl->font,
                                          bakeScale,
                                          ch,
                                          pimpl->atlasPadding,
                                          pimpl->atlasOneEdgeValue,
                                          pimpl->atlasPixelDistScale,
                                          &glyph.width,
                                          &glyph.height,
                                          &glyph.xOffset,
                                          &glyph.yOffset);
        if (glyph.sdf) {
            baked.push_back(glyph);
        }
    }

    I32 atlasWidth = pimpl->atlasWidth;
    for (const auto& glyph : baked) {
        atlasWidth = std::max(atlasWidth, glyph.width + 1);
    }

    int x = 0;
    int y = 0;
    int maxHeight = 0;
    std::vector<Extent2D<I32>> placements(baked.size());

    for (U64 i = 0; i < baked.size(); i++) {
        const auto& glyph = baked[i];
        if (x + glyph.width >= atlasWidth) {
            x = 0;
            y += maxHeight + 1;
            maxHeight = 0;
        }
        placements[i] = {x, y};
        x += glyph.width + 1;
        maxHeight = std::max(maxHeight, glyph.height);
    }

    pimpl->atlasSize = {atlasWidth, y + maxHeight + 1};
    std::vector<uint8_t> atlas(static_cast<U64>(pimpl->atlasSize.x) * pimpl->atlasSize.y);

    for (U64 i = 0; i < baked.size(); i++) {
        const auto& glyph = baked[i];
        const auto& [gx, gy] = placements[i];

        for (int j = 0; j < glyph.height; ++j) {
            for (int k = 0; k < glyph.width; ++k) {
                atlas[(gy + j) * pimpl->atlasSize.x + (gx + k)] = glyph.sdf[j * glyph.width + k];
            }
        }

        glyphs[glyph.code] = {
            .x0 = gx,
            .y0 = gy,
            .x1 = gx + glyph.width,
            .y1 = gy + glyph.height,
            .xOffset = static_cast<F32>(glyph.xOffset) / pimpl->atlasOversample,
            .yOffset = static_cast<F32>(glyph.yOffset) / pimpl->atlasOversample,
            .xAdvance = glyphs[glyph.code].xAdvance
        };

        stbtt_FreeSDF(glyph.sdf, nullptr);
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
    // TODO: Implement full UTF-8 support.
    // The atlas only covers printable ASCII (codes 0..95 == chars 32..127); any
    // other byte (a UTF-8 multi-byte lead/continuation, or a control char) has no
    // glyph. Return a blank one instead of throwing so non-ASCII text renders as
    // empty space rather than crashing the renderer.
    static const Glyph kEmpty = {};
    const auto it = glyphs.find(code);
    return it != glyphs.end() ? it->second : kEmpty;
}

I32 Font::ascent() const {
    return pimpl->ascent;
}

I32 Font::descent() const {
    return pimpl->descent;
}

I32 Font::lineHeight() const {
    return pimpl->ascent - pimpl->descent;
}

F32 Font::atlasScale() const {
    return pimpl->atlasOversample;
}

F32 Font::atlasPixelRange() const {
    return 255.0f / pimpl->atlasPixelDistScale;
}

const std::shared_ptr<Render::Texture>& Font::atlas() const {
    return pimpl->fontAtlasTexture;
}

const Extent2D<I32>& Font::atlasSize() const {
    return pimpl->atlasSize;
}

}  // namespace Jetstream::Render::Components
