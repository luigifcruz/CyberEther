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
#include <cmath>
#include <thread>
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

    pimpl->atlasOversample = config.icons ? 1.0f : 2.0f;
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

    std::vector<I32> codepoints;
    const auto addRange = [&](I32 first, I32 last) {
        for (I32 code = first; code <= last; ++code) {
            if (code == 0xFFFD || stbtt_FindGlyphIndex(&pimpl->font, code) != 0) {
                codepoints.push_back(code);
            }
        }
    };
    addRange(0x0020, 0x007E);
    addRange(0x00A0, 0x017F);
    addRange(0x2010, 0x2027);
    addRange(0x2030, 0x203C);
    addRange(0x20A0, 0x20BF);
    addRange(0x2100, 0x214F);
    addRange(0x2190, 0x21FF);
    addRange(0x2200, 0x2265);
    addRange(0x25A0, 0x25CF);
    addRange(0x2605, 0x2606);
    addRange(0x2610, 0x2612);
    addRange(0x2713, 0x2718);
    if (config.icons) {
        addRange(0xE000, 0xF8FF);
    }
    addRange(0xFFFD, 0xFFFD);

    std::vector<Baked> results(codepoints.size());
    const U64 workers = std::max<U64>(1, std::min<U64>(std::thread::hardware_concurrency(), 8));
    std::vector<std::thread> threads;
    for (U64 t = 0; t < workers; ++t) {
        threads.emplace_back([&, t]() {
            for (U64 i = t; i < codepoints.size(); i += workers) {
                Baked glyph{.code = codepoints[i]};
                glyph.sdf = stbtt_GetCodepointSDF(&pimpl->font,
                                                  bakeScale,
                                                  glyph.code,
                                                  pimpl->atlasPadding,
                                                  pimpl->atlasOneEdgeValue,
                                                  pimpl->atlasPixelDistScale,
                                                  &glyph.width,
                                                  &glyph.height,
                                                  &glyph.xOffset,
                                                  &glyph.yOffset);
                results[i] = glyph;
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    U64 atlasArea = 0;
    for (U64 i = 0; i < codepoints.size(); i++) {
        int advanceWidth, leftSideBearing;
        stbtt_GetCodepointHMetrics(&pimpl->font, codepoints[i], &advanceWidth, &leftSideBearing);

        glyphs[codepoints[i]] = {
            .x0 = 0,
            .y0 = 0,
            .x1 = 0,
            .y1 = 0,
            .xOffset = 0.0f,
            .yOffset = 0.0f,
            .xAdvance = static_cast<F32>(advanceWidth * scale)
        };

        if (results[i].sdf) {
            baked.push_back(results[i]);
            atlasArea += static_cast<U64>(results[i].width + 1) * static_cast<U64>(results[i].height + 1);
        }
    }

    const I32 suggestedWidth = static_cast<I32>(std::sqrt(static_cast<F64>(atlasArea) * 1.1));
    I32 atlasWidth = std::max(pimpl->atlasWidth, (suggestedWidth + 255) / 256 * 256);
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
    static const Glyph kEmpty = {};
    auto it = glyphs.find(code);
    if (it == glyphs.end()) {
        it = glyphs.find(0xFFFD);
    }
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
