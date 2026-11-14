#ifndef JETSTREAM_RENDER_SAKURA_DETAIL_TERMINAL_SPRITES_HH
#define JETSTREAM_RENDER_SAKURA_DETAIL_TERMINAL_SPRITES_HH

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/polyline.hh>
#include <jetstream/types.hh>

#include <vector>

namespace Jetstream::Sakura {

struct TerminalSprites {
    std::vector<Box::Instance> rects;
    std::vector<Polyline::Instance> strokes;
    U64 rectCapacity = 0;
    U64 strokeCapacity = 0;

    static bool Covers(U32 codepoint);

    void clear();
    bool draw(U32 codepoint, const Rect& cell, const ColorRGBA<F32>& color, F32 light);

 private:
    struct Cell {
        F32 x = 0.0f;
        F32 y = 0.0f;
        I32 width = 0;
        I32 height = 0;
        I32 light = 1;
        ColorRGBA<F32> color;
    };

    void rect(const Cell& cell, I32 x0, I32 y0, I32 x1, I32 y1, F32 alpha = 1.0f);
    void stroke(const Cell& cell, std::vector<Extent2D<F32>> points, F32 thickness, bool filled = false);
    void lines(const Cell& cell, U8 spec);
    void dashes(const Cell& cell, bool vertical, I32 count, I32 thickness, I32 gap);
    void arc(const Cell& cell, bool top, bool left);
    void diagonal(const Cell& cell, bool rising);
    void block(const Cell& cell, U32 codepoint);
    void braille(const Cell& cell, U32 codepoint);
    void powerline(const Cell& cell, U32 codepoint);
    void arrow(const Cell& cell, U32 codepoint);
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_DETAIL_TERMINAL_SPRITES_HH
