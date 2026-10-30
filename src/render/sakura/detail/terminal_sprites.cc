#include "terminal_sprites.hh"

#include <algorithm>
#include <cmath>
#include <utility>

namespace Jetstream::Sakura {

namespace {

enum LineStyle : U8 {
    None = 0,
    Light = 1,
    Heavy = 2,
    Double = 3,
};

constexpr U8 kLines[128] = {
    0x44, 0x88, 0x11, 0x22, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x14, 0x18, 0x24, 0x28,
    0x50, 0x90, 0x60, 0xA0, 0x05, 0x09, 0x06, 0x0A, 0x41, 0x81, 0x42, 0x82, 0x15, 0x19, 0x16, 0x25,
    0x26, 0x1A, 0x29, 0x2A, 0x51, 0x91, 0x52, 0x61, 0x62, 0x92, 0xA1, 0xA2, 0x54, 0x94, 0x58, 0x98,
    0x64, 0xA4, 0x68, 0xA8, 0x45, 0x85, 0x49, 0x89, 0x46, 0x86, 0x4A, 0x8A, 0x55, 0x95, 0x59, 0x99,
    0x56, 0x65, 0x66, 0x96, 0x5A, 0xA5, 0x69, 0x9A, 0xA9, 0xA6, 0x6A, 0xAA, 0x00, 0x00, 0x00, 0x00,
    0xCC, 0x33, 0x1C, 0x34, 0x3C, 0xD0, 0x70, 0xF0, 0x0D, 0x07, 0x0F, 0xC1, 0x43, 0xC3, 0x1D, 0x37,
    0x3F, 0xD1, 0x73, 0xF3, 0xDC, 0x74, 0xFC, 0xCD, 0x47, 0xCF, 0xDD, 0x77, 0xFF, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x40, 0x01, 0x04, 0x10, 0x80, 0x02, 0x08, 0x20, 0x48, 0x21, 0x84, 0x12,
};

constexpr U8 kHorizontalLight = (Light << 2) | (Light << 6);
constexpr U8 kVerticalLight = Light | (Light << 4);
constexpr U64 kArcSegments = 8;
constexpr F32 kArcControl = 0.25f;
constexpr F32 kPointEpsilon = 0.01f;
constexpr F32 kArrowInset = 0.1f;
constexpr F32 kArrowHead = 0.36f;
constexpr F32 kArrowHeadLength = 1.2f;
constexpr F32 kArrowShaft = 0.12f;
constexpr F32 kArrowNotch = 0.3f;

constexpr U8 kQuadTopLeft = 1;
constexpr U8 kQuadTopRight = 2;
constexpr U8 kQuadBottomLeft = 4;
constexpr U8 kQuadBottomRight = 8;
constexpr U8 kQuadrants[10] = {
    kQuadBottomLeft,
    kQuadBottomRight,
    kQuadTopLeft,
    kQuadTopLeft | kQuadBottomLeft | kQuadBottomRight,
    kQuadTopLeft | kQuadBottomRight,
    kQuadTopLeft | kQuadTopRight | kQuadBottomLeft,
    kQuadTopLeft | kQuadTopRight | kQuadBottomRight,
    kQuadTopRight,
    kQuadTopRight | kQuadBottomLeft,
    kQuadTopRight | kQuadBottomLeft | kQuadBottomRight,
};

I32 Sub(I32 a, I32 b) {
    return std::max(0, a - b);
}

I32 Fraction(I32 size, F32 fraction) {
    return static_cast<I32>(std::round(static_cast<F32>(size) * fraction));
}

}  // namespace

bool TerminalSprites::Covers(U32 codepoint) {
    return (codepoint >= 0x2500 && codepoint <= 0x259F) ||
           (codepoint >= 0x2800 && codepoint <= 0x28FF) ||
           (codepoint >= 0xE0B0 && codepoint <= 0xE0B3) ||
           codepoint == 0x2794 || codepoint == 0x279C || codepoint == 0x279E || codepoint == 0x27A4;
}

void TerminalSprites::clear() {
    rects.clear();
    strokes.clear();
}

bool TerminalSprites::draw(U32 codepoint, const Rect& cellRect, const ColorRGBA<F32>& color, F32 light) {
    if (!Covers(codepoint)) {
        return false;
    }
    const Cell cell{
        .x = cellRect.x,
        .y = cellRect.y,
        .width = static_cast<I32>(std::round(cellRect.width)),
        .height = static_cast<I32>(std::round(cellRect.height)),
        .light = std::max(1, static_cast<I32>(std::round(light))),
        .color = color,
    };
    if (cell.width <= 0 || cell.height <= 0 || color.a <= 0.0f) {
        return true;
    }

    if (codepoint >= 0x2580 && codepoint <= 0x259F) {
        block(cell, codepoint);
        return true;
    }
    if (codepoint >= 0x2800 && codepoint <= 0x28FF) {
        braille(cell, codepoint);
        return true;
    }
    if (codepoint >= 0xE0B0) {
        powerline(cell, codepoint);
        return true;
    }
    if (codepoint >= 0x2794 && codepoint <= 0x27A4) {
        arrow(cell, codepoint);
        return true;
    }

    if (const U8 spec = kLines[codepoint - 0x2500]; spec != 0) {
        lines(cell, spec);
        return true;
    }
    const I32 lightPx = cell.light;
    const I32 heavyPx = 2 * cell.light;
    const I32 gap = std::max(4, lightPx);
    switch (codepoint) {
        case 0x2504: dashes(cell, false, 3, lightPx, gap); break;
        case 0x2505: dashes(cell, false, 3, heavyPx, gap); break;
        case 0x2506: dashes(cell, true, 3, lightPx, gap); break;
        case 0x2507: dashes(cell, true, 3, heavyPx, gap); break;
        case 0x2508: dashes(cell, false, 4, lightPx, gap); break;
        case 0x2509: dashes(cell, false, 4, heavyPx, gap); break;
        case 0x250A: dashes(cell, true, 4, lightPx, gap); break;
        case 0x250B: dashes(cell, true, 4, heavyPx, gap); break;
        case 0x254C: dashes(cell, false, 2, lightPx, lightPx); break;
        case 0x254D: dashes(cell, false, 2, heavyPx, heavyPx); break;
        case 0x254E: dashes(cell, true, 2, lightPx, heavyPx); break;
        case 0x254F: dashes(cell, true, 2, heavyPx, heavyPx); break;
        case 0x256D: arc(cell, false, false); break;
        case 0x256E: arc(cell, false, true); break;
        case 0x256F: arc(cell, true, true); break;
        case 0x2570: arc(cell, true, false); break;
        case 0x2571: diagonal(cell, true); break;
        case 0x2572: diagonal(cell, false); break;
        case 0x2573:
            diagonal(cell, true);
            diagonal(cell, false);
            break;
        default: break;
    }
    return true;
}

void TerminalSprites::rect(const Cell& cell, I32 x0, I32 y0, I32 x1, I32 y1, F32 alpha) {
    x0 = std::clamp(x0, 0, cell.width);
    x1 = std::clamp(x1, 0, cell.width);
    y0 = std::clamp(y0, 0, cell.height);
    y1 = std::clamp(y1, 0, cell.height);
    if (x1 <= x0 || y1 <= y0) {
        return;
    }
    ColorRGBA<F32> color = cell.color;
    color.a *= alpha;
    const Rect bounds = {
        cell.x + static_cast<F32>(x0),
        cell.y + static_cast<F32>(y0),
        static_cast<F32>(x1 - x0),
        static_cast<F32>(y1 - y0),
    };
    if (!rects.empty()) {
        auto& last = rects.back();
        if (last.rect.y == bounds.y && last.rect.height == bounds.height && last.backgroundColor == color &&
            bounds.x >= last.rect.x && bounds.x <= last.rect.right()) {
            last.rect.width = std::max(last.rect.right(), bounds.right()) - last.rect.x;
            return;
        }
        if (last.rect.x == bounds.x && last.rect.width == bounds.width && last.backgroundColor == color &&
            bounds.y >= last.rect.y && bounds.y <= last.rect.bottom()) {
            last.rect.height = std::max(last.rect.bottom(), bounds.bottom()) - last.rect.y;
            return;
        }
    }
    if (rects.size() >= rectCapacity) {
        return;
    }
    rects.push_back({.rect = bounds, .backgroundColor = color, .corners = 0});
}

void TerminalSprites::stroke(const Cell& cell, std::vector<Extent2D<F32>> points, F32 thickness, bool filled) {
    if (strokes.size() >= strokeCapacity) {
        return;
    }
    Polyline::Instance instance{
        .color = cell.color,
        .thickness = thickness,
        .filled = filled,
    };
    instance.points.reserve(points.size());
    for (const auto& point : points) {
        const Extent2D<F32> absolute = {cell.x + point.x, cell.y + point.y};
        if (!instance.points.empty()) {
            const auto& previous = instance.points.back();
            if (std::abs(previous.x - absolute.x) < kPointEpsilon && std::abs(previous.y - absolute.y) < kPointEpsilon) {
                continue;
            }
        }
        instance.points.push_back(absolute);
    }
    strokes.push_back(std::move(instance));
}

void TerminalSprites::lines(const Cell& cell, U8 spec) {
    const U8 up = spec & 0x3;
    const U8 right = (spec >> 2) & 0x3;
    const U8 down = (spec >> 4) & 0x3;
    const U8 left = (spec >> 6) & 0x3;
    const I32 width = cell.width;
    const I32 height = cell.height;
    const I32 lightPx = cell.light;
    const I32 heavyPx = 2 * cell.light;

    const I32 hLightTop = Sub(height, lightPx) / 2;
    const I32 hLightBottom = hLightTop + lightPx;
    const I32 hHeavyTop = Sub(height, heavyPx) / 2;
    const I32 hHeavyBottom = hHeavyTop + heavyPx;
    const I32 hDoubleTop = Sub(hLightTop, lightPx);
    const I32 hDoubleBottom = hLightBottom + lightPx;

    const I32 vLightLeft = Sub(width, lightPx) / 2;
    const I32 vLightRight = vLightLeft + lightPx;
    const I32 vHeavyLeft = Sub(width, heavyPx) / 2;
    const I32 vHeavyRight = vHeavyLeft + heavyPx;
    const I32 vDoubleLeft = Sub(vLightLeft, lightPx);
    const I32 vDoubleRight = vLightRight + lightPx;

    const I32 upBottom = (left == Heavy || right == Heavy) ? hHeavyBottom
                         : (left != right || down == up)    ? ((left == Double || right == Double) ? hDoubleBottom
                                                                                                   : hLightBottom)
                         : (left == None && right == None)  ? hLightBottom
                                                            : hLightTop;
    const I32 downTop = (left == Heavy || right == Heavy) ? hHeavyTop
                        : (left != right || up == down)    ? ((left == Double || right == Double) ? hDoubleTop
                                                                                                  : hLightTop)
                        : (left == None && right == None)  ? hLightTop
                                                           : hLightBottom;
    const I32 leftRight = (up == Heavy || down == Heavy) ? vHeavyRight
                          : (up != down || left == right) ? ((up == Double || down == Double) ? vDoubleRight
                                                                                              : vLightRight)
                          : (up == None && down == None)  ? vLightRight
                                                          : vLightLeft;
    const I32 rightLeft = (up == Heavy || down == Heavy) ? vHeavyLeft
                          : (up != down || right == left) ? ((up == Double || down == Double) ? vDoubleLeft
                                                                                              : vLightLeft)
                          : (up == None && down == None)  ? vLightLeft
                                                          : vLightRight;

    switch (up) {
        case Light: rect(cell, vLightLeft, 0, vLightRight, upBottom); break;
        case Heavy: rect(cell, vHeavyLeft, 0, vHeavyRight, upBottom); break;
        case Double:
            rect(cell, vDoubleLeft, 0, vLightLeft, left == Double ? hLightTop : upBottom);
            rect(cell, vLightRight, 0, vDoubleRight, right == Double ? hLightTop : upBottom);
            break;
        default: break;
    }
    switch (left) {
        case Light: rect(cell, 0, hLightTop, leftRight, hLightBottom); break;
        case Heavy: rect(cell, 0, hHeavyTop, leftRight, hHeavyBottom); break;
        case Double:
            rect(cell, 0, hDoubleTop, up == Double ? vLightLeft : leftRight, hLightTop);
            rect(cell, 0, hLightBottom, down == Double ? vLightLeft : leftRight, hDoubleBottom);
            break;
        default: break;
    }
    switch (right) {
        case Light: rect(cell, rightLeft, hLightTop, width, hLightBottom); break;
        case Heavy: rect(cell, rightLeft, hHeavyTop, width, hHeavyBottom); break;
        case Double:
            rect(cell, up == Double ? vLightRight : rightLeft, hDoubleTop, width, hLightTop);
            rect(cell, down == Double ? vLightRight : rightLeft, hLightBottom, width, hDoubleBottom);
            break;
        default: break;
    }
    switch (down) {
        case Light: rect(cell, vLightLeft, downTop, vLightRight, height); break;
        case Heavy: rect(cell, vHeavyLeft, downTop, vHeavyRight, height); break;
        case Double:
            rect(cell, vDoubleLeft, left == Double ? hLightBottom : downTop, vLightLeft, height);
            rect(cell, vLightRight, right == Double ? hLightBottom : downTop, vDoubleRight, height);
            break;
        default: break;
    }
}

void TerminalSprites::dashes(const Cell& cell, bool vertical, I32 count, I32 thickness, I32 gap) {
    const I32 length = vertical ? cell.height : cell.width;
    if (length < 2 * count) {
        lines(cell, vertical ? kVerticalLight : kHorizontalLight);
        return;
    }
    const I32 gapLength = std::min(gap, length / (2 * count));
    const I32 dashTotal = length - count * gapLength;
    const I32 dashLength = dashTotal / count;
    I32 extra = dashTotal % count;
    I32 position = vertical ? 0 : gapLength / 2;
    const I32 offset = Sub(vertical ? cell.width : cell.height, thickness) / 2;
    for (I32 index = 0; index < count; ++index) {
        I32 end = position + dashLength;
        if (extra > 0) {
            --extra;
            ++end;
        }
        if (vertical) {
            rect(cell, offset, position, offset + thickness, end);
        } else {
            rect(cell, position, offset, end, offset + thickness);
        }
        position = end + gapLength;
    }
}

void TerminalSprites::arc(const Cell& cell, bool top, bool left) {
    const F32 width = static_cast<F32>(cell.width);
    const F32 height = static_cast<F32>(cell.height);
    const F32 thickness = static_cast<F32>(cell.light);
    const F32 centerX = static_cast<F32>(Sub(cell.width, cell.light) / 2) + thickness * 0.5f;
    const F32 centerY = static_cast<F32>(Sub(cell.height, cell.light) / 2) + thickness * 0.5f;
    const F32 radius = std::min(width, height) * 0.5f;
    const F32 sx = left ? -1.0f : 1.0f;
    const F32 sy = top ? -1.0f : 1.0f;

    std::vector<Extent2D<F32>> points;
    points.push_back({centerX, top ? 0.0f : height});
    const auto curve = Polyline::Bezier({centerX, centerY + sy * radius},
                                        {centerX, centerY + sy * kArcControl * radius},
                                        {centerX + sx * kArcControl * radius, centerY},
                                        {centerX + sx * radius, centerY},
                                        kArcSegments);
    points.insert(points.end(), curve.begin(), curve.end());
    points.push_back({left ? 0.0f : width, centerY});
    stroke(cell, std::move(points), thickness);
}

void TerminalSprites::diagonal(const Cell& cell, bool rising) {
    const F32 width = static_cast<F32>(cell.width);
    const F32 height = static_cast<F32>(cell.height);
    const F32 slopeX = std::min(1.0f, width / height) * 0.5f;
    const F32 slopeY = std::min(1.0f, height / width) * 0.5f;
    if (rising) {
        stroke(cell, {{width + slopeX, -slopeY}, {-slopeX, height + slopeY}}, static_cast<F32>(cell.light));
    } else {
        stroke(cell, {{-slopeX, -slopeY}, {width + slopeX, height + slopeY}}, static_cast<F32>(cell.light));
    }
}

void TerminalSprites::block(const Cell& cell, U32 codepoint) {
    const I32 width = cell.width;
    const I32 height = cell.height;
    const auto upper = [&](F32 fraction) {
        rect(cell, 0, 0, width, Fraction(height, fraction));
    };
    const auto lower = [&](F32 fraction) {
        rect(cell, 0, height - Fraction(height, fraction), width, height);
    };
    const auto leftBlock = [&](F32 fraction) {
        rect(cell, 0, 0, Fraction(width, fraction), height);
    };
    const auto rightBlock = [&](F32 fraction) {
        rect(cell, width - Fraction(width, fraction), 0, width, height);
    };

    if (codepoint == 0x2580) {
        upper(0.5f);
    } else if (codepoint <= 0x2587) {
        lower(static_cast<F32>(codepoint - 0x2580) / 8.0f);
    } else if (codepoint == 0x2588) {
        rect(cell, 0, 0, width, height);
    } else if (codepoint <= 0x258F) {
        leftBlock(static_cast<F32>(0x2590 - codepoint) / 8.0f);
    } else if (codepoint == 0x2590) {
        rightBlock(0.5f);
    } else if (codepoint <= 0x2593) {
        rect(cell, 0, 0, width, height, static_cast<F32>(codepoint - 0x2590) * 0.25f);
    } else if (codepoint == 0x2594) {
        upper(0.125f);
    } else if (codepoint == 0x2595) {
        rightBlock(0.125f);
    } else {
        const U8 quads = kQuadrants[codepoint - 0x2596];
        const I32 midX = Fraction(width, 0.5f);
        const I32 midY = Fraction(height, 0.5f);
        if (quads & kQuadTopLeft) {
            rect(cell, 0, 0, midX, midY);
        }
        if (quads & kQuadTopRight) {
            rect(cell, midX, 0, width, midY);
        }
        if (quads & kQuadBottomLeft) {
            rect(cell, 0, midY, midX, height);
        }
        if (quads & kQuadBottomRight) {
            rect(cell, midX, midY, width, height);
        }
    }
}

void TerminalSprites::braille(const Cell& cell, U32 codepoint) {
    const I32 width = cell.width;
    const I32 height = cell.height;
    I32 dot = std::min(width / 4, height / 8);
    I32 spacingX = width / 4;
    I32 spacingY = height / 8;
    I32 marginX = spacingX / 2;
    I32 marginY = spacingY / 2;
    I32 leftX = width - 2 * marginX - spacingX - 2 * dot;
    I32 leftY = height - 2 * marginY - 3 * spacingY - 4 * dot;

    if (leftX >= 2 && leftY >= 4 && dot == 0) {
        dot += 1;
        leftX -= 2;
        leftY -= 4;
    }
    if (leftX >= 2 && marginX == 0) {
        marginX = 1;
        leftX -= 2;
    }
    if (leftY >= 2 && marginY == 0) {
        marginY = 1;
        leftY -= 2;
    }
    if (leftX >= 1) {
        spacingX += 1;
        leftX -= 1;
    }
    if (leftY >= 3) {
        spacingY += 1;
        leftY -= 3;
    }
    if (leftX >= 2) {
        marginX += 1;
        leftX -= 2;
    }
    if (leftY >= 2) {
        marginY += 1;
        leftY -= 2;
    }
    if (leftX >= 2 && leftY >= 4) {
        dot += 1;
    }
    if (dot <= 0) {
        return;
    }

    const I32 x[2] = {marginX, marginX + dot + spacingX};
    I32 y[4] = {marginY, 0, 0, 0};
    for (U64 index = 1; index < 4; ++index) {
        y[index] = y[index - 1] + dot + spacingY;
    }
    struct Dot {
        U8 column;
        U8 row;
    };
    constexpr Dot kDots[8] = {{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}, {0, 3}, {1, 3}};
    const U32 pattern = codepoint - 0x2800;
    for (U8 row = 0; row < 4; ++row) {
        for (U8 bit = 0; bit < 8; ++bit) {
            if ((pattern & (1u << bit)) && kDots[bit].row == row) {
                const I32 left = x[kDots[bit].column];
                rect(cell, left, y[row], left + dot, y[row] + dot);
            }
        }
    }
}

void TerminalSprites::arrow(const Cell& cell, U32 codepoint) {
    const F32 width = static_cast<F32>(cell.width);
    const F32 middle = static_cast<F32>(Sub(cell.height, cell.light) / 2) + static_cast<F32>(cell.light) * 0.5f;
    const F32 head = std::min(width * kArrowHead, static_cast<F32>(cell.height) * kArrowHead);
    const F32 left = width * kArrowInset;
    const F32 tip = width - left;
    if (codepoint == 0x27A4) {
        const F32 notch = left + (tip - left) * kArrowNotch;
        stroke(cell, {{left, middle - head}, {tip, middle}, {notch, middle}}, 0.0f, true);
        stroke(cell, {{notch, middle}, {tip, middle}, {left, middle + head}}, 0.0f, true);
        return;
    }
    const F32 base = tip - head * kArrowHeadLength;
    const F32 shaft = std::max(2.0f * static_cast<F32>(cell.light), std::round(static_cast<F32>(cell.height) *
                                                                                kArrowShaft));
    stroke(cell, {{left, middle}, {base + shaft * 0.5f, middle}}, shaft);
    stroke(cell, {{base, middle - head}, {tip, middle}, {base, middle + head}}, 0.0f, true);
}

void TerminalSprites::powerline(const Cell& cell, U32 codepoint) {
    const F32 width = static_cast<F32>(cell.width);
    const F32 height = static_cast<F32>(cell.height);
    const F32 middle = height * 0.5f;
    const F32 thickness = static_cast<F32>(cell.light);
    switch (codepoint) {
        case 0xE0B0: stroke(cell, {{0.0f, 0.0f}, {width, middle}, {0.0f, height}}, thickness, true); break;
        case 0xE0B1: stroke(cell, {{0.0f, 0.0f}, {width, middle}, {0.0f, height}}, thickness); break;
        case 0xE0B2: stroke(cell, {{width, height}, {0.0f, middle}, {width, 0.0f}}, thickness, true); break;
        case 0xE0B3: stroke(cell, {{width, 0.0f}, {0.0f, middle}, {width, height}}, thickness); break;
        default: break;
    }
}

}  // namespace Jetstream::Sakura
