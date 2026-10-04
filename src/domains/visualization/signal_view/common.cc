#include "common.hh"

#include <algorithm>
#include <any>
#include <array>
#include <span>

#include <glm/gtc/matrix_transform.hpp>

#include "lineplot.hh"
#include "waterfall.hh"

namespace Jetstream::Modules {

namespace {

constexpr ColorRGBA<F32> kAccentColor = {1.0f, 0.85f, 0.0f, 1.0f};
constexpr ColorRGBA<F32> kCursorLineColor = {1.0f, 1.0f, 1.0f, 0.35f};
constexpr ColorRGBA<F32> kCursorHaloColor = {0.0f, 0.0f, 0.0f, 0.65f};
constexpr ColorRGBA<F32> kCursorPillColor = {0.07f, 0.07f, 0.07f, 0.9f};
constexpr ColorRGBA<F32> kCursorPillEdgeColor = {1.0f, 1.0f, 1.0f, 0.22f};
constexpr std::array<ColorRGBA<F32>, detail::MaxMarkers> kMarkerPalette = {{
    {0.31f, 0.76f, 0.97f, 1.0f},
    {1.00f, 0.44f, 0.26f, 1.0f},
    {0.40f, 0.73f, 0.42f, 1.0f},
    {0.67f, 0.28f, 0.74f, 1.0f},
    {1.00f, 0.93f, 0.35f, 1.0f},
    {0.15f, 0.78f, 0.85f, 1.0f},
    {0.94f, 0.33f, 0.31f, 1.0f},
    {0.93f, 0.25f, 0.48f, 1.0f},
    {0.55f, 0.43f, 0.39f, 1.0f},
    {0.61f, 0.80f, 0.40f, 1.0f},
    {0.26f, 0.65f, 0.96f, 1.0f},
    {1.00f, 0.65f, 0.15f, 1.0f},
    {0.49f, 0.34f, 0.76f, 1.0f},
    {0.15f, 0.65f, 0.60f, 1.0f},
    {0.83f, 0.88f, 0.34f, 1.0f},
    {0.74f, 0.74f, 0.74f, 1.0f},
}};

constexpr ColorRGBA<F32> MarkerLineColor(const U64 index) {
    auto color = kMarkerPalette[index];
    color.a = 0.6f;
    return color;
}

constexpr ColorRGBA<F32> MarkerTagTextColor(const U64 index) {
    const auto& color = kMarkerPalette[index];
    const F32 luminance = 0.2126f * color.r + 0.7152f * color.g + 0.0722f * color.b;
    return luminance > 0.5f ? ColorRGBA<F32>{0.05f, 0.05f, 0.05f, 1.0f}
                            : ColorRGBA<F32>{1.0f, 1.0f, 1.0f, 1.0f};
}
constexpr ColorRGBA<F32> kMarkerSpanColor = {1.0f, 1.0f, 1.0f, 0.6f};
constexpr F32 kLabelScale = 0.85f;
constexpr F32 kMarkerPickRadiusPx = 8.0f;
constexpr F32 kMarkerSpanThicknessPx = 3.0f;
constexpr F32 kMarkerSpanArrowArmPx = 11.0f;
constexpr F32 kMarkerSpanArrowAngleDeg = 30.0f;
constexpr F32 kMarkerSpanLabelGapPx = 10.0f;
constexpr F32 kMarkerSpanTagGapPx = 5.0f;
constexpr F32 kMarkerTableGapPx = 12.0f;

enum CursorInstance : U64 {
    kCursorLine = 0,
    kCursorHalo,
    kCursorMarker,
    kCursorPillEdge,
    kCursorPill,
    kCursorInstances,
};

enum MarkerGroup : U64 {
    kMarkerLine = 0,
    kMarkerHalo,
    kMarkerDot,
    kMarkerGroups,
};

enum MarkerTableGroup : U64 {
    kMarkerPillEdge = 0,
    kMarkerPill,
    kMarkerTableGroups,
};

enum MarkerSpanSegment : U64 {
    kSpanLeadSegment = 0,
    kSpanTrailSegment,
    kSpanSegments,
};

enum MarkerSpanArrow : U64 {
    kSpanLeadUpperArm = 0,
    kSpanLeadLowerArm,
    kSpanTrailUpperArm,
    kSpanTrailLowerArm,
    kSpanArrows,
};

constexpr U64 MarkerInstance(const U64 group, const U64 index) {
    return group * detail::MaxMarkers + index;
}

constexpr U64 SpanInstance(const U64 span, const U64 segment) {
    return span * kSpanSegments + segment;
}

constexpr U64 ArrowInstance(const U64 span, const U64 arm) {
    return span * kSpanArrows + arm;
}

std::string MarkerElement(const U64 index, const char* suffix) {
    return jst::fmt::format("marker-{}-{}", index, suffix);
}

std::string SpanElement(const U64 index, const char* suffix) {
    return jst::fmt::format("span-{}-{}", index, suffix);
}

F32 PanTranslation(const SurfaceInteractionState& interaction) {
    const F32 maxTranslation = std::abs((1.0f / interaction.zoom) - 1.0f);
    return std::clamp(-2.0f * interaction.offset, -maxTranslation, maxTranslation);
}

}  // namespace

SignalViewFrequency SignalViewFrequencyOf(const Tensor& input) {
    if (!input.hasAttribute("frequency") || !input.hasAttribute("sampleRate")) {
        return {};
    }
    return {
        .valid = true,
        .center = std::any_cast<F32>(input.attribute("frequency")),
        .sampleRate = std::any_cast<F32>(input.attribute("sampleRate")),
    };
}

void SignalViewCanvas::reset(const SignalView& config) {
    splitter = {};
    splitter.ratio = config.splitRatio;
    displayHeld = false;
    cursor = {};
    markerPositions = config.markers;
    updateMarkersFlag = false;
    markerDrag = {};
    tagBounds.fill({});
    applyPins(config.pins);
}

Result SignalViewCanvas::create(const std::shared_ptr<Render::Window>& window,
                                const Context& context) {
    const auto& config = context.config;
    const bool lineplot = context.lineplot != nullptr;
    const bool waterfall = context.waterfall != nullptr;
    const bool combined = lineplot && waterfall;

    {
        Render::Components::Axis::Config cfg;
        cfg.thickness = detail::kSignalViewLineThickness;
        cfg.showInteriorGrid = lineplot;
        cfg.verticalScale = combined ? config.splitRatio : 1.0f;
        cfg.showFrameTicks = lineplot;
        cfg.font = window->font("default_mono");
        cfg.xTitle = config.xLabel;
        cfg.yTitle = combined ? "" : (lineplot
            ? config.amplitudeLabel
            : config.waterfallLabel);
        cfg.yLabelOnRight = lineplot;
        cfg.gridColor = {0.12f, 0.12f, 0.12f, 1.0f};
        cfg.majorGridColor = {0.5f, 0.5f, 0.5f, 1.0f};
        JST_CHECK(window->build(axis, cfg));
        JST_CHECK(window->bind(axis));
    }

    // Text labels (header, zoom, and cursor readouts).

    {
        Render::Components::Text::Config cfg;
        cfg.maxCharacters = 256;
        cfg.color = {1.0f, 1.0f, 1.0f, 1.0f};
        cfg.font = window->font("default_mono");
        cfg.elements = {
            {"header",
             {.scale = kLabelScale,
              .position = {-1.0f, 1.0f},
              .alignment = {0, 0}}},
            {"zoom",
             {.scale = kLabelScale,
              .position = {0.0f, 1.0f},
              .alignment = {1, 0}}},
            {"hold",
             {.scale = kLabelScale,
              .position = {-1.0f, 1.0f},
              .alignment = {0, 0},
              .color = kAccentColor}},
            {"amplitude-title",
             {.scale = kLabelScale,
              .position = {-1.0f, 0.5f},
              .alignment = {1, 0},
              .rotationDeg = 90.0f}},
            {"waterfall-title",
             {.scale = kLabelScale,
              .position = {-1.0f, -0.5f},
              .alignment = {1, 0},
              .rotationDeg = 90.0f}},
        };
        JST_CHECK(window->build(text, cfg));
        JST_CHECK(window->bind(text));
    }

    // Cursor overlay (line, trace marker, and readout pill).

    {
        Render::Components::Shapes::Config cfg;
        cfg.pixelSize = {
            2.0f / interaction.viewSize.x,
            2.0f / interaction.viewSize.y,
        };
        cfg.elements["cursor"] = {
            .type = Render::Components::Shapes::Type::RECT,
            .numberOfInstances = kCursorInstances,
            .position = {-2.0f, -2.0f},
            .size = {0.0f, 0.0f},
            .cornerRadius = 1.0e4f,
        };
        JST_CHECK(window->build(cursorShapes, cfg));
        JST_CHECK(window->bind(cursorShapes));

        std::span<ColorRGBA<F32>> colors;
        JST_CHECK(cursorShapes->getColors("cursor", colors));
        colors[kCursorLine] = kCursorLineColor;
        colors[kCursorHalo] = kCursorHaloColor;
        colors[kCursorMarker] = kAccentColor;
        colors[kCursorPillEdge] = kCursorPillEdgeColor;
        colors[kCursorPill] = kCursorPillColor;
        JST_CHECK(cursorShapes->updateColors("cursor"));
    }

    {
        Render::Components::Text::Config cfg;
        cfg.maxCharacters = 64;
        cfg.color = {1.0f, 1.0f, 1.0f, 1.0f};
        cfg.font = window->font("default_mono");
        cfg.elements = {
            {"cursor-x",
             {.scale = kLabelScale,
              .position = {-2.0f, -2.0f},
              .alignment = {0, 1}}},
            {"cursor-y",
             {.scale = kLabelScale,
              .position = {-2.0f, -2.0f},
              .alignment = {0, 1},
              .color = kAccentColor}},
        };
        JST_CHECK(window->build(cursorText, cfg));
        JST_CHECK(window->bind(cursorText));
    }

    // Marker overlay (lines, trace dots, and readout table).

    {
        Render::Components::Shapes::Config cfg;
        cfg.pixelSize = {
            2.0f / interaction.viewSize.x,
            2.0f / interaction.viewSize.y,
        };
        cfg.elements["markers"] = {
            .type = Render::Components::Shapes::Type::RECT,
            .numberOfInstances = kMarkerGroups * detail::MaxMarkers,
            .position = {-2.0f, -2.0f},
            .size = {0.0f, 0.0f},
            .cornerRadius = 1.0e4f,
        };
        JST_CHECK(window->build(markerShapes, cfg));
        JST_CHECK(window->bind(markerShapes));

        std::span<ColorRGBA<F32>> colors;
        JST_CHECK(markerShapes->getColors("markers", colors));
        for (U64 i = 0; i < detail::MaxMarkers; ++i) {
            colors[MarkerInstance(kMarkerLine, i)] = MarkerLineColor(i);
            colors[MarkerInstance(kMarkerHalo, i)] = kCursorHaloColor;
            colors[MarkerInstance(kMarkerDot, i)] = kMarkerPalette[i];
        }
        JST_CHECK(markerShapes->updateColors("markers"));
    }

    {
        Render::Components::Shapes::Config cfg;
        cfg.pixelSize = {
            2.0f / interaction.viewSize.x,
            2.0f / interaction.viewSize.y,
        };
        cfg.elements["tags"] = {
            .type = Render::Components::Shapes::Type::RECT,
            .numberOfInstances = detail::MaxMarkers,
            .position = {-2.0f, -2.0f},
            .size = {0.0f, 0.0f},
            .cornerRadius = 4.0f,
        };
        JST_CHECK(window->build(markerTagShapes, cfg));
        JST_CHECK(window->bind(markerTagShapes));

        std::span<ColorRGBA<F32>> colors;
        JST_CHECK(markerTagShapes->getColors("tags", colors));
        for (U64 i = 0; i < detail::MaxMarkers; ++i) {
            colors[i] = kMarkerPalette[i];
        }
        JST_CHECK(markerTagShapes->updateColors("tags"));
    }

    {
        Render::Components::Shapes::Config cfg;
        cfg.pixelSize = {
            2.0f / interaction.viewSize.x,
            2.0f / interaction.viewSize.y,
        };
        cfg.elements["spans"] = {
            .type = Render::Components::Shapes::Type::RECT,
            .numberOfInstances = kSpanSegments * detail::MarkerSpans,
            .color = kMarkerSpanColor,
            .position = {-2.0f, -2.0f},
            .size = {0.0f, 0.0f},
            .cornerRadius = kMarkerSpanThicknessPx * 0.5f,
        };
        cfg.elements["arrows"] = {
            .type = Render::Components::Shapes::Type::RECT,
            .numberOfInstances = kSpanArrows * detail::MarkerSpans,
            .color = kMarkerSpanColor,
            .position = {-2.0f, -2.0f},
            .size = {0.0f, 0.0f},
            .cornerRadius = kMarkerSpanThicknessPx * 0.5f,
        };
        JST_CHECK(window->build(markerSpanShapes, cfg));
        JST_CHECK(window->bind(markerSpanShapes));
    }

    {
        Render::Components::Shapes::Config cfg;
        cfg.pixelSize = {
            2.0f / interaction.viewSize.x,
            2.0f / interaction.viewSize.y,
        };
        cfg.elements["table"] = {
            .type = Render::Components::Shapes::Type::RECT,
            .numberOfInstances = kMarkerTableGroups * detail::MaxMarkers,
            .position = {-2.0f, -2.0f},
            .size = {0.0f, 0.0f},
            .cornerRadius = 1.0e4f,
        };
        JST_CHECK(window->build(markerTableShapes, cfg));
        JST_CHECK(window->bind(markerTableShapes));

        std::span<ColorRGBA<F32>> colors;
        JST_CHECK(markerTableShapes->getColors("table", colors));
        for (U64 i = 0; i < detail::MaxMarkers; ++i) {
            colors[MarkerInstance(kMarkerPillEdge, i)] = kCursorPillEdgeColor;
            colors[MarkerInstance(kMarkerPill, i)] = kCursorPillColor;
        }
        JST_CHECK(markerTableShapes->updateColors("table"));
    }

    {
        Render::Components::Text::Config cfg;
        cfg.maxCharacters = 1024;
        cfg.color = {1.0f, 1.0f, 1.0f, 1.0f};
        cfg.font = window->font("default_mono");
        for (U64 i = 0; i < detail::MaxMarkers; ++i) {
            cfg.elements[MarkerElement(i, "x")] = {
                .scale = kLabelScale,
                .position = {-2.0f, -2.0f},
                .alignment = {0, 1},
            };
            cfg.elements[MarkerElement(i, "y")] = {
                .scale = kLabelScale,
                .position = {-2.0f, -2.0f},
                .alignment = {0, 1},
            };
        }
        for (U64 i = 0; i < detail::MarkerSpans; ++i) {
            cfg.elements[SpanElement(i, "label")] = {
                .scale = kLabelScale,
                .position = {-2.0f, -2.0f},
                .alignment = {1, 1},
                .color = kMarkerSpanColor,
            };
        }
        JST_CHECK(window->build(markerText, cfg));
        JST_CHECK(window->bind(markerText));
    }

    {
        Render::Components::Text::Config cfg;
        cfg.maxCharacters = 64;
        cfg.color = {1.0f, 1.0f, 1.0f, 1.0f};
        cfg.font = window->hasFont("default_mono_bold")
            ? window->font("default_mono_bold")
            : window->font("default_mono");
        for (U64 i = 0; i < detail::MaxMarkers; ++i) {
            cfg.elements[MarkerElement(i, "id")] = {
                .scale = kLabelScale,
                .position = {-2.0f, -2.0f},
                .alignment = {0, 1},
                .color = kMarkerPalette[i],
            };
        }
        JST_CHECK(window->build(markerBadgeText, cfg));
        JST_CHECK(window->bind(markerBadgeText));
    }

    {
        Render::Components::Text::Config cfg;
        cfg.maxCharacters = 64;
        cfg.color = {1.0f, 1.0f, 1.0f, 1.0f};
        cfg.font = window->hasFont("default_mono_bold")
            ? window->font("default_mono_bold")
            : window->font("default_mono");
        for (U64 i = 0; i < detail::MaxMarkers; ++i) {
            cfg.elements[MarkerElement(i, "tag")] = {
                .scale = kLabelScale,
                .position = {-2.0f, -2.0f},
                .alignment = {1, 1},
                .color = MarkerTagTextColor(i),
            };
        }
        JST_CHECK(window->build(markerTagText, cfg));
        JST_CHECK(window->bind(markerTagText));
    }

    // Framebuffer texture.

    {
        Render::Texture::Config cfg;
        cfg.size = interaction.viewSize;
        JST_CHECK(window->build(framebufferTexture, cfg));
    }

    // Surface.

    {
        Render::Surface::Config cfg;
        cfg.framebuffer = framebufferTexture;
        cfg.multisampled = lineplot;
        cfg.clearColor = {0.0f, 0.0f, 0.0f, 1.0f};
        if (waterfall) {
            context.waterfall->attach(cfg);
        }
        JST_CHECK(axis->surfaceUnderlay(cfg));
        if (lineplot) {
            context.lineplot->attach(cfg);
        }
        JST_CHECK(markerShapes->surface(cfg));
        JST_CHECK(markerSpanShapes->surface(cfg));
        JST_CHECK(markerTagShapes->surface(cfg));
        JST_CHECK(markerTagText->surface(cfg));
        JST_CHECK(markerTableShapes->surface(cfg));
        JST_CHECK(markerText->surface(cfg));
        JST_CHECK(markerBadgeText->surface(cfg));
        JST_CHECK(axis->surfaceOverlay(cfg));
        JST_CHECK(text->surface(cfg));
        JST_CHECK(cursorShapes->surface(cfg));
        JST_CHECK(cursorText->surface(cfg));
        JST_CHECK(window->build(renderSurface, cfg));
        JST_CHECK(window->bind(renderSurface));
    }

    updateState(context);

    return Result::SUCCESS;
}

Result SignalViewCanvas::destroy(const std::shared_ptr<Render::Window>& window) {
    if (renderSurface) {
        JST_CHECK(window->unbind(renderSurface));
    }
    if (cursorText) {
        JST_CHECK(window->unbind(cursorText));
    }
    if (cursorShapes) {
        JST_CHECK(window->unbind(cursorShapes));
    }
    if (markerTagText) {
        JST_CHECK(window->unbind(markerTagText));
    }
    if (markerBadgeText) {
        JST_CHECK(window->unbind(markerBadgeText));
    }
    if (markerText) {
        JST_CHECK(window->unbind(markerText));
    }
    if (markerTableShapes) {
        JST_CHECK(window->unbind(markerTableShapes));
    }
    if (markerTagShapes) {
        JST_CHECK(window->unbind(markerTagShapes));
    }
    if (markerSpanShapes) {
        JST_CHECK(window->unbind(markerSpanShapes));
    }
    if (markerShapes) {
        JST_CHECK(window->unbind(markerShapes));
    }
    if (text) {
        JST_CHECK(window->unbind(text));
    }
    if (axis) {
        JST_CHECK(window->unbind(axis));
    }
    return Result::SUCCESS;
}

Result SignalViewCanvas::processSurfaceEvents(std::vector<SurfaceEvent>&& events) {
    interaction = ProcessSurfaceInteraction(interaction, std::move(events), {});
    // Resize the axis before hit-testing so input and rendering use the same
    // padded plot rectangle, including on a resize-and-click frame.
    return axis->updatePixelSize({
        (2.0f * interaction.scale) / interaction.viewSize.x,
        (2.0f * interaction.scale) / interaction.viewSize.y,
    });
}

SurfaceCursor SignalViewCanvas::processInputEvents(std::vector<InputEvent>&& events,
                                                   const Context& context,
                                                   const Edits& edits) {
    const auto& paddingScale = axis->paddingScale();
    const F32 splitRatio = context.config.splitRatio;
    const F32 previousRatio = splitter.ratio;
    if (!splitter.dragging && !edits.pending()) {
        splitter.ratio = splitRatio;
    }

    syncMarkers(context, edits);

    std::vector<InputEvent> plotEvents;
    for (const auto& input : events) {
        if (const auto* key = std::get_if<KeyEvent>(&input)) {
            if (key->type == KeyEventType::Press && key->key == KeyCode::Space && !key->repeat) {
                displayHeld = !displayHeld;
            }
            if (key->type == KeyEventType::Press && key->key == KeyCode::Escape &&
                std::any_of(pinned.begin(), pinned.end(), [](const bool flag) { return flag; })) {
                pinned.fill(false);
                commitMarkers(context, edits);
            }
        }
        const auto mouse = SurfaceMouseEvent(input);
        if (!mouse) continue;
        const auto& event = *mouse;
        if (event.type == MouseEventType::Move) {
            cursor.inside = true;
            cursor.position = event.position;
        } else if (event.type == MouseEventType::Leave) {
            cursor.inside = false;
        }
        if (event.type == MouseEventType::Click && event.modifiers.shift && !splitter.dragging) {
            cursor.inside = true;
            cursor.position = event.position;
            if (event.button == MouseButton::Left) {
                toggleMarker(context, edits);
            } else if (event.button == MouseButton::Right) {
                clearMarkers(context, edits);
            }
            continue;
        }
        if (markerDrag.index && *markerDrag.index >= markerPositions.size()) {
            markerDrag = {};
        }
        if (markerDrag.index) {
            const bool released = event.type == MouseEventType::Release &&
                                  event.button == MouseButton::Left;
            if (event.type == MouseEventType::Move || released) {
                cursor.inside = true;
                cursor.position = event.position;
                const F32 travel = std::abs(event.position.x - markerDrag.origin.x) *
                                   static_cast<F32>(interaction.viewSize.x);
                if (markerDrag.moved || travel > 3.0f * interaction.scale) {
                    markerDrag.moved = true;
                    markerPositions[*markerDrag.index] = pointAtX(event.position.x);
                }
                if (!released) {
                    continue;
                }
            }
            if (released || event.type == MouseEventType::Leave) {
                if (event.type == MouseEventType::Leave) {
                    cursor.inside = false;
                }
                if (markerDrag.moved) {
                    commitMarkers(context, edits);
                }
                markerDrag = {};
                continue;
            }
        }
        if (event.type == MouseEventType::Click && event.button == MouseButton::Left &&
            !splitter.dragging) {
            cursor.inside = true;
            cursor.position = event.position;
            if (const auto tag = tagAt(event.position)) {
                pinned[*tag] = !pinned[*tag];
                commitMarkers(context, edits);
                continue;
            }
            if (const auto hit = markerAt(context, event.position)) {
                markerDrag = {.index = hit, .origin = event.position};
                continue;
            }
        }
        const auto layout = detail::CalculateSignalViewPanels(paddingScale,
                                                               interaction.viewSize,
                                                               splitter.ratio);
        bool commit = false;
        if (splitter.process(event, layout, interaction.viewSize,
                              interaction.scale, edits.splitEnabled, commit)) {
            // Splitter capture must never start or continue a horizontal pan.
            interaction.dragging = false;
            if (commit && (splitter.ratio != splitRatio || edits.pending())) {
                Parser::Map edit;
                edit["splitRatio"] = splitter.ratio;
                if (edits.request(edit) != Result::SUCCESS) {
                    splitter.ratio = splitRatio;
                }
            } else if (event.type == MouseEventType::Leave) {
                splitter.ratio = splitRatio;
            }
        } else {
            plotEvents.push_back(event);
        }
    }
    const bool viewChanged = interaction.viewChanged || previousRatio != splitter.ratio;
    interaction = ProcessSurfaceInteraction(interaction, {}, std::move(plotEvents));
    interaction.viewChanged |= viewChanged;

    SurfaceCursor shape = SurfaceCursor::Default;
    if (markerDrag.index) {
        shape = SurfaceCursor::ResizeEW;
    } else if (splitter.dragging) {
        shape = SurfaceCursor::ResizeNS;
    } else if (cursor.inside) {
        const auto layout = detail::CalculateSignalViewPanels(paddingScale,
                                                               interaction.viewSize,
                                                               splitter.ratio);
        if (tagAt(cursor.position)) {
            shape = SurfaceCursor::Default;
        } else if (markerAt(context, cursor.position)) {
            shape = SurfaceCursor::ResizeEW;
        } else if (splitter.hovered(cursor.position, layout, interaction.viewSize,
                                    interaction.scale, edits.splitEnabled)) {
            shape = SurfaceCursor::ResizeNS;
        }
    }
    return shape;
}

void SignalViewCanvas::resize() {
    renderSurface->size(interaction.viewSize);
    renderSurface->clearColor(interaction.backgroundColor);
}

void SignalViewCanvas::updateState(const Context& context) {
    const F32 translation = PanTranslation(interaction);

    // Update global pixel size.

    pixelSize = {
        (2.0f * interaction.scale) / interaction.viewSize.x,
        (2.0f * interaction.scale) / interaction.viewSize.y
    };

    // Update axis component (computes paddingScale internally).

    axis->updatePixelSize(pixelSize);
    const auto& paddingScale = axis->paddingScale();

    const bool combined = context.lineplot && context.waterfall;
    const auto panels = detail::CalculateSignalViewPanels(paddingScale,
                                                           interaction.viewSize,
                                                           splitter.ratio);
    const F32 linePanelScale = combined ? panels.lineFraction : 1.0f;
    axis->updateVerticalScale(linePanelScale);

    // Lay out the lineplot and waterfall inside the plot area.

    if (context.lineplot) {
        auto signalTransform = glm::mat4(1.0f);

        const F32 linePanelOffset = paddingScale.y * (1.0f - linePanelScale);
        signalTransform = glm::translate(signalTransform,
                                         glm::vec3(translation *
                                                       paddingScale.x *
                                                       interaction.zoom,
                                                   linePanelOffset, 0.0f));
        signalTransform = glm::scale(signalTransform,
                                     glm::vec3(paddingScale.x,
                                               paddingScale.y * linePanelScale,
                                               1.0f));

        context.lineplot->layout(signalTransform, pixelSize, linePanelScale,
                                 interaction.zoom, combined ? panels.line : panels.plot);
    }

    if (context.waterfall) {
        if (combined) {
            context.waterfall->layout(panels.waterfall,
                                      paddingScale.x,
                                      paddingScale.y * (1.0f - linePanelScale),
                                      -paddingScale.y * linePanelScale);
        } else {
            context.waterfall->layout(panels.plot, paddingScale.x, paddingScale.y, 0.0f);
        }
    }

    const auto& vs = interaction.viewSize;
    axis->updateScissorRect({0, 0,
                             static_cast<U32>(vs.x),
                             static_cast<U32>(vs.y)});

    updateLabels(context);
}

Result SignalViewCanvas::present(const Context& context) {
    updateLabels(context);
    JST_CHECK(updateMarkers(context));
    JST_CHECK(updateCursor(context));

    JST_CHECK(axis->present());
    if (text) {
        JST_CHECK(text->present());
    }
    if (markerShapes) {
        JST_CHECK(markerShapes->present());
    }
    if (markerSpanShapes) {
        JST_CHECK(markerSpanShapes->present());
    }
    if (markerTagShapes) {
        JST_CHECK(markerTagShapes->present());
    }
    if (markerTableShapes) {
        JST_CHECK(markerTableShapes->present());
    }
    if (markerText) {
        JST_CHECK(markerText->present());
    }
    if (markerBadgeText) {
        JST_CHECK(markerBadgeText->present());
    }
    if (markerTagText) {
        JST_CHECK(markerTagText->present());
    }
    if (cursorShapes) {
        JST_CHECK(cursorShapes->present());
    }
    if (cursorText) {
        JST_CHECK(cursorText->present());
    }

    return Result::SUCCESS;
}

void SignalViewCanvas::updateLabels(const Context& context) {
    const auto& config = context.config;
    const auto& paddingScale = axis->paddingScale();
    const bool lineplot = context.lineplot != nullptr;
    const bool combined = lineplot && context.waterfall;
    const auto& frequency = context.frequency;
    const F32 translation = PanTranslation(interaction);

    // Update tick labels via axis component.

    if (axis) {
        const bool ticksVisible = lineplot &&
            interaction.placement != SurfacePlacementType::Attached;
        axis->setShowFrameTicks(ticksVisible);

        auto xFormatter = [frequency, lineplot,
                           zoom = interaction.zoom, translation](const F32 position) {
            const F32 normalizedPos = position / zoom - translation;
            if (frequency.valid) {
                const F32 freq =
                    (frequency.center + normalizedPos * frequency.sampleRate / 2.0f) / 1e6f;
                return jst::fmt::format("{:.02f}", freq);
            }
            const F32 value = lineplot
                ? normalizedPos
                : (normalizedPos + 1.0f) * 0.5f;
            return jst::fmt::format("{:.02f}", value);
        };

        Render::Components::Axis::TickFormatter yFormatter;
        if (lineplot) {
            yFormatter = [min = config.rangeMin, max = config.rangeMax](const F32 position) {
                return detail::LineplotAmplitudeLabel(position, min, max);
            };
        }

        axis->updateTickFormatters(std::move(xFormatter), std::move(yFormatter));
    }

    if (text) {
        text->updatePixelSize(pixelSize);
    }
    if (cursorText) {
        cursorText->updatePixelSize(pixelSize);
    }
    if (markerText) {
        markerText->updatePixelSize(pixelSize);
    }
    if (markerBadgeText) {
        markerBadgeText->updatePixelSize(pixelSize);
    }
    if (markerTagText) {
        markerTagText->updatePixelSize(pixelSize);
    }

    if (lineplot && text) {
        const F32 tickOffset = axis->getConfig().majorTickLengthPx + 4.0f;

        auto holdLabel = text->get("hold");
        const F32 lineHeight = text->getConfig().font
            ? static_cast<F32>(text->getConfig().font->lineHeight()) * holdLabel.scale
            : 0.0f;
        holdLabel.position = {-paddingScale.x + pixelSize.x * tickOffset,
                              paddingScale.y - pixelSize.y * (tickOffset + lineHeight)};
        holdLabel.fill = displayHeld ? "HOLD" : " ";
        text->update("hold", holdLabel);

        auto header = text->get("header");
        if (interaction.placement == SurfacePlacementType::Attached) {
            header.fill = " ";
        } else {
            header.position = {-paddingScale.x + pixelSize.x * tickOffset,
                               paddingScale.y - pixelSize.y * tickOffset};
            if (frequency.valid) {
                header.fill = jst::fmt::format("CENTER {:.3f} MHz   SPAN {:.3f} MHz",
                                               frequency.center / 1e6f,
                                               frequency.sampleRate / 1e6f);
            } else {
                header.fill = "CENTER 0.000   SPAN 1.000";
            }
        }
        text->update("header", header);

        auto zoomLabel = text->get("zoom");
        zoomLabel.position = {
            paddingScale.x -
                pixelSize.x * (axis->getConfig().majorTickLengthPx + 4.0f),
            paddingScale.y -
                pixelSize.y * (axis->getConfig().majorTickLengthPx + 4.0f),
        };
        zoomLabel.alignment = {2, 0};
        if (splitter.dragging) {
            zoomLabel.fill = jst::fmt::format("SPLIT {:.0f}%", splitter.ratio * 100.0f);
        } else if (interaction.placement == SurfacePlacementType::Attached) {
            zoomLabel.fill = " ";
        } else if (std::abs(interaction.zoom - 1.0f) > 0.01f) {
            zoomLabel.fill = jst::fmt::format("ZOOM {:.1f}x", interaction.zoom);
        } else {
            zoomLabel.fill = " ";
        }
        text->update("zoom", zoomLabel);

        auto amplitudeTitle = text->get("amplitude-title");
        const F32 lineFraction = axis->getConfig().verticalScale;
        amplitudeTitle.position = {-1.0f + pixelSize.x * 3.0f,
                                   paddingScale.y * (1.0f - lineFraction)};
        amplitudeTitle.fill = combined ? config.amplitudeLabel : " ";
        text->update("amplitude-title", amplitudeTitle);

        auto waterfallTitle = text->get("waterfall-title");
        waterfallTitle.position = {-1.0f + pixelSize.x * 3.0f,
                                   -paddingScale.y * lineFraction};
        waterfallTitle.fill = combined ? config.waterfallLabel : " ";
        text->update("waterfall-title", waterfallTitle);
    }
}

F32 SignalViewCanvas::viewTranslation() const {
    return PanTranslation(interaction);
}

std::optional<F32> SignalViewCanvas::cursorPoint(const Context& context) const {
    const bool visible = cursor.inside &&
                         !splitter.dragging &&
                         interaction.placement != SurfacePlacementType::Attached &&
                         insidePlot(cursor.position) &&
                         context.numberOfElements >= 2;
    if (!visible) {
        return std::nullopt;
    }
    return pointAtX(cursor.position.x);
}

F32 SignalViewCanvas::projectPointX(const F32 xPoint) const {
    return (xPoint + viewTranslation()) * interaction.zoom * axis->paddingScale().x;
}

std::optional<F32> SignalViewCanvas::displayedAmplitude(const Context& context,
                                                        const F32 xPoint) const {
    if (!context.lineplot) {
        return std::nullopt;
    }
    return context.lineplot->sample(xPoint);
}

F32 SignalViewCanvas::amplitudeToNdc(const Context& context, const F32 yPoint) const {
    const auto& padding = axis->paddingScale();
    const bool combined = context.lineplot && context.waterfall;
    const F32 lineFraction = combined ? axis->getConfig().verticalScale : 1.0f;
    return padding.y * (1.0f - lineFraction) +
           padding.y * lineFraction * std::clamp(yPoint, -1.0f, 1.0f);
}

std::string SignalViewCanvas::formatPointX(const Context& context, const F32 xPoint) const {
    const auto& frequency = context.frequency;
    if (frequency.valid) {
        return jst::fmt::format("{:.4f} MHz",
                                (frequency.center + xPoint * frequency.sampleRate / 2.0f) / 1e6f);
    }
    return jst::fmt::format("{:.4f}", context.lineplot ? xPoint : (xPoint + 1.0f) * 0.5f);
}

std::string SignalViewCanvas::formatSpanX(const Context& context, const F32 delta) const {
    if (context.frequency.valid) {
        return detail::FormatFrequencySpan(std::abs(delta) * context.frequency.sampleRate / 2.0f);
    }
    return jst::fmt::format("{:.4f}", std::abs(context.lineplot ? delta : delta * 0.5f));
}

std::string SignalViewCanvas::formatAmplitude(const Context& context, const F32 yPoint) const {
    const auto& config = context.config;
    const auto value = detail::LineplotAmplitudeValue(yPoint, config.rangeMin, config.rangeMax);
    if (!value) {
        return {};
    }
    const auto unit = detail::LabelUnit(config.amplitudeLabel);
    return unit.empty()
        ? jst::fmt::format("{:.1f}", *value)
        : jst::fmt::format("{:.1f} {}", *value, unit);
}

void SignalViewCanvas::syncMarkers(const Context& context, const Edits& edits) {
    if (markerDrag.index || edits.pending()) {
        return;
    }
    if (updateMarkersFlag || edits.enabled("markers")) {
        markerPositions = context.config.markers;
    }
    if (updateMarkersFlag || edits.enabled("pins")) {
        applyPins(context.config.pins);
    }
    updateMarkersFlag = false;
}

void SignalViewCanvas::applyPins(const std::vector<U64>& pins) {
    pinned.fill(false);
    for (const auto index : pins) {
        if (index < markerPositions.size() && index < pinned.size()) {
            pinned[index] = true;
        }
    }
}

std::vector<U64> SignalViewCanvas::pinnedIndices() const {
    std::vector<U64> indices;
    for (U64 i = 0; i < markerPositions.size() && i < pinned.size(); ++i) {
        if (pinned[i]) {
            indices.push_back(i);
        }
    }
    return indices;
}

bool SignalViewCanvas::insidePlot(const Extent2D<F32>& position) const {
    const auto& padding = axis->paddingScale();
    const F32 u = (position.x - 0.5f) / std::max(padding.x, 1e-6f) + 0.5f;
    const F32 v = (position.y - 0.5f) / std::max(padding.y, 1e-6f) + 0.5f;
    return u >= 0.0f && u <= 1.0f && v >= 0.0f && v <= 1.0f;
}

F32 SignalViewCanvas::pointAtX(const F32 x) const {
    const F32 u = (x - 0.5f) / std::max(axis->paddingScale().x, 1e-6f) + 0.5f;
    return std::clamp((u * 2.0f - 1.0f) / interaction.zoom - viewTranslation(), -1.0f, 1.0f);
}

std::optional<U64> SignalViewCanvas::markerAt(const Context& context,
                                              const Extent2D<F32>& position) const {
    if (interaction.placement == SurfacePlacementType::Attached ||
        context.numberOfElements < 2 || !insidePlot(position)) {
        return std::nullopt;
    }
    const F32 pointerNdc = position.x * 2.0f - 1.0f;
    F32 best = kMarkerPickRadiusPx * (2.0f * interaction.scale) /
               static_cast<F32>(std::max<U64>(interaction.viewSize.x, 1));
    std::optional<U64> nearest;
    for (U64 i = 0; i < markerPositions.size(); ++i) {
        const F32 distance = std::abs(projectPointX(markerPositions[i]) - pointerNdc);
        if (distance < best) {
            best = distance;
            nearest = i;
        }
    }
    return nearest;
}

std::optional<U64> SignalViewCanvas::tagAt(const Extent2D<F32>& position) const {
    const Extent2D<F32> pointer = {position.x * 2.0f - 1.0f, 1.0f - position.y * 2.0f};
    for (U64 i = 0; i < tagBounds.size(); ++i) {
        const auto& bounds = tagBounds[i];
        if (bounds.active &&
            std::abs(pointer.x - bounds.center.x) <= bounds.halfSize.x &&
            std::abs(pointer.y - bounds.center.y) <= bounds.halfSize.y) {
            return i;
        }
    }
    return std::nullopt;
}

void SignalViewCanvas::toggleMarker(const Context& context, const Edits& edits) {
    const auto point = cursorPoint(context);
    if (!point) {
        return;
    }

    if (const auto nearest = markerAt(context, cursor.position)) {
        const U64 removed = *nearest;
        markerPositions.erase(markerPositions.begin() + removed);
        std::copy(pinned.begin() + removed + 1, pinned.end(), pinned.begin() + removed);
        pinned.back() = false;
    } else if (markerPositions.size() < detail::MaxMarkers) {
        markerPositions.push_back(*point);
    } else {
        return;
    }
    commitMarkers(context, edits);
}

void SignalViewCanvas::clearMarkers(const Context& context, const Edits& edits) {
    if (markerPositions.empty()) {
        return;
    }
    markerPositions.clear();
    pinned.fill(false);
    commitMarkers(context, edits);
}

void SignalViewCanvas::commitMarkers(const Context& context, const Edits& edits) {
    if (!edits.enabled("markers")) {
        return;
    }
    Parser::Map edit;
    edit["markers"] = markerPositions;
    if (edits.enabled("pins")) {
        edit["pins"] = pinnedIndices();
    }
    if (edits.request(edit) != Result::SUCCESS) {
        markerPositions = context.config.markers;
        applyPins(context.config.pins);
    }
}

Result SignalViewCanvas::updateCursor(const Context& context) {
    const auto& padding = axis->paddingScale();
    const auto point = cursorPoint(context);
    const bool visible = point.has_value();

    std::string xLabelText;
    std::string yLabelText;
    F32 xNdc = 0.0f;
    F32 yNdc = 0.0f;
    bool hasMarker = false;

    if (visible) {
        cursor.point = *point;
        xNdc = std::clamp((*point + viewTranslation()) * interaction.zoom,
                          -1.0f, 1.0f) * padding.x;
        xLabelText = formatPointX(context, *point);
        if (const auto yPoint = displayedAmplitude(context, *point)) {
            yLabelText = formatAmplitude(context, *yPoint);
            if (!yLabelText.empty()) {
                yNdc = amplitudeToNdc(context, *yPoint);
                hasMarker = true;
            }
        }
    }

    cursor.visible = visible;
    cursor.marker = hasMarker;
    cursor.plot = {xNdc, yNdc};

    Extent2D<F32> xLabelPosition = {-2.0f, -2.0f};
    Extent2D<F32> yLabelPosition = {-2.0f, -2.0f};

    if (cursorShapes) {
        JST_CHECK(cursorShapes->updatePixelSize({
            2.0f / interaction.viewSize.x,
            2.0f / interaction.viewSize.y,
        }));

        std::span<Extent2D<F32>> positions;
        JST_CHECK(cursorShapes->getPositions("cursor", positions));
        std::span<Extent2D<F32>> sizes;
        JST_CHECK(cursorShapes->getSizes("cursor", sizes));
        for (U64 i = 0; i < kCursorInstances; ++i) {
            positions[i] = {-2.0f, -2.0f};
            sizes[i] = {0.0f, 0.0f};
        }

        if (visible) {
            const F32 scale = interaction.scale;
            const F32 toPixelsX = static_cast<F32>(interaction.viewSize.x) * 0.5f;
            const F32 toPixelsY = static_cast<F32>(interaction.viewSize.y) * 0.5f;

            if (!cursor.overMarker) {
                positions[kCursorLine] = {xNdc, 0.0f};
                sizes[kCursorLine] = {2.0f * scale, padding.y * 2.0f * toPixelsY};
            }

            if (hasMarker && !cursor.overMarker) {
                positions[kCursorHalo] = {xNdc, yNdc};
                sizes[kCursorHalo] = {16.0f * scale, 16.0f * scale};
                positions[kCursorMarker] = {xNdc, yNdc};
                sizes[kCursorMarker] = {11.0f * scale, 11.0f * scale};
            }

            if (cursorText && !cursor.overMarker) {
                const F32 lineHeight = cursorText->lineHeight(kLabelScale);
                const F32 xWidth = cursorText->advance(xLabelText, kLabelScale);
                const F32 yWidth = yLabelText.empty()
                    ? 0.0f
                    : cursorText->advance(yLabelText, kLabelScale);
                const F32 gap = yLabelText.empty() ? 0.0f : 10.0f;
                const F32 padX = 9.0f;
                const F32 padY = 4.0f;
                const F32 pillWidth = (padX * 2.0f + xWidth + gap + yWidth) * pixelSize.x;
                const F32 pillHeight = (padY * 2.0f + lineHeight) * pixelSize.y;
                const F32 nudge = 14.0f * pixelSize.x;

                F32 left = xNdc + nudge;
                if (left + pillWidth > padding.x) {
                    left = xNdc - nudge - pillWidth;
                }
                const F32 headerOffset = axis->getConfig().majorTickLengthPx + 4.0f;
                const F32 top = padding.y - (headerOffset + lineHeight + 8.0f) * pixelSize.y;
                const F32 centerY = top - pillHeight * 0.5f;
                const F32 centerX = left + pillWidth * 0.5f;

                positions[kCursorPillEdge] = {centerX, centerY};
                sizes[kCursorPillEdge] = {pillWidth * toPixelsX + 2.0f * scale,
                                          pillHeight * toPixelsY + 2.0f * scale};
                positions[kCursorPill] = {centerX, centerY};
                sizes[kCursorPill] = {pillWidth * toPixelsX, pillHeight * toPixelsY};

                xLabelPosition = {left + padX * pixelSize.x, centerY};
                yLabelPosition = {left + (padX + xWidth + gap) * pixelSize.x, centerY};
            }
        }

        JST_CHECK(cursorShapes->updatePositions("cursor"));
        JST_CHECK(cursorShapes->updateSizes("cursor"));
    }

    if (cursorText) {
        const bool readout = visible && !cursor.overMarker;
        auto xLabelElement = cursorText->get("cursor-x");
        xLabelElement.position = xLabelPosition;
        xLabelElement.fill = readout ? xLabelText : " ";
        JST_CHECK(cursorText->update("cursor-x", xLabelElement));

        auto yLabelElement = cursorText->get("cursor-y");
        yLabelElement.position = yLabelPosition;
        yLabelElement.fill = (readout && !yLabelText.empty()) ? yLabelText : " ";
        JST_CHECK(cursorText->update("cursor-y", yLabelElement));
    }

    return Result::SUCCESS;
}

Result SignalViewCanvas::updateMarkers(const Context& context) {
    const auto& padding = axis->paddingScale();
    const bool shown = context.numberOfElements >= 2;
    const bool table = interaction.placement != SurfacePlacementType::Attached;

    struct Row {
        std::string tag;
        std::string id;
        std::string x;
        std::string y;
        F32 xNdc = 0.0f;
        F32 yNdc = 0.0f;
        bool inView = false;
        bool dot = false;
        bool tagged = false;
        F32 tagWidth = 0.0f;
        Extent2D<F32> tagPosition = {-2.0f, -2.0f};
        Extent2D<F32> idPosition = {-2.0f, -2.0f};
        Extent2D<F32> xPosition = {-2.0f, -2.0f};
        Extent2D<F32> yPosition = {-2.0f, -2.0f};
    };
    struct Span {
        bool active = false;
        F32 left = 0.0f;
        F32 right = 0.0f;
        std::string label;
        Extent2D<F32> labelPosition = {-2.0f, -2.0f};
    };
    std::array<Row, detail::MaxMarkers> rows;
    std::array<Span, detail::MarkerSpans> spans;
    const U64 count = shown ? std::min<U64>(markerPositions.size(), detail::MaxMarkers) : 0;
    Render::Components::Text* const badgeText =
        markerBadgeText ? markerBadgeText.get() : markerText.get();
    Render::Components::Text* const tagText =
        markerTagText ? markerTagText.get() : badgeText;

    for (U64 i = 0; i < count; ++i) {
        auto& row = rows[i];
        const F32 marker = markerPositions[i];
        row.xNdc = projectPointX(marker);
        row.inView = std::abs(row.xNdc) <= padding.x;
        row.id = jst::fmt::format("M{}", i + 1);
        row.tag = row.id;
        row.x = formatPointX(context, marker);
        if (const auto yPoint = displayedAmplitude(context, marker)) {
            row.y = formatAmplitude(context, *yPoint);
            if (!row.y.empty()) {
                row.yNdc = amplitudeToNdc(context, *yPoint);
                row.dot = row.inView;
            }
        }
    }

    const F32 scale = interaction.scale;
    const F32 toPixelsX = static_cast<F32>(interaction.viewSize.x) * 0.5f;
    const F32 toPixelsY = static_cast<F32>(interaction.viewSize.y) * 0.5f;
    const F32 lineHeight = markerText ? markerText->lineHeight(kLabelScale) : 0.0f;
    const F32 tagLineHeight = tagText ? tagText->lineHeight(kLabelScale) : 0.0f;
    const F32 headerOffset = axis->getConfig().majorTickLengthPx + 4.0f;
    const F32 headerHeight = table ? tagLineHeight + 8.0f : 0.0f;
    const F32 tagPadX = 6.0f;
    const F32 tagPadY = 2.0f;
    const F32 tagHeight = (tagPadY * 2.0f + tagLineHeight) * pixelSize.y;
    const F32 cursorRowTop = padding.y - (headerOffset + headerHeight) * pixelSize.y;
    const F32 cursorRowCenter = cursorRowTop - (8.0f + tagLineHeight) * pixelSize.y * 0.5f;

    tagBounds.fill({});
    for (U64 i = 0; i < count; ++i) {
        auto& row = rows[i];
        if (!row.inView || !markerText) {
            continue;
        }
        row.tagWidth = (tagPadX * 2.0f + tagText->advance(row.tag, kLabelScale)) * pixelSize.x;
        // Center oversized tags instead of clamping with inverted bounds.
        const F32 tagLimit = std::max(0.0f, padding.x - row.tagWidth * 0.5f);
        row.tagPosition = {
            std::clamp(row.xNdc, -tagLimit, tagLimit),
            cursorRowCenter,
        };
        row.tagged = true;
        tagBounds[i] = {
            .active = true,
            .center = row.tagPosition,
            .halfSize = {row.tagWidth * 0.5f, tagHeight * 0.5f},
        };
    }

    std::optional<U64> hovered;
    if (cursor.inside && !splitter.dragging && shown) {
        hovered = tagAt(cursor.position);
    }
    cursor.overMarker = hovered.has_value() || markerDrag.index.has_value() ||
                        (cursor.inside && !splitter.dragging && markerAt(context, cursor.position));

    std::vector<U64> focused;
    if (hovered) {
        focused.push_back(*hovered);
    }
    for (U64 i = 0; i < count; ++i) {
        if (pinned[i] && rows[i].tagged && hovered != i) {
            focused.push_back(i);
        }
    }

    if (!focused.empty()) {
        std::vector<U64> order;
        for (U64 i = 0; i < count; ++i) {
            if (rows[i].tagged) {
                order.push_back(i);
            }
        }
        std::sort(order.begin(), order.end(), [&](const U64 a, const U64 b) {
            return rows[a].xNdc < rows[b].xNdc;
        });
        std::array<bool, detail::MarkerSpans> gaps{};
        for (const U64 index : focused) {
            const U64 rank = std::find(order.begin(), order.end(), index) - order.begin();
            if (rank > 0) {
                gaps[rank - 1] = true;
            }
            if (rank + 1 < order.size()) {
                gaps[rank] = true;
            }
        }
        const F32 arrowLength = kMarkerSpanArrowArmPx *
                                std::cos(glm::radians(kMarkerSpanArrowAngleDeg)) * pixelSize.x;
        const F32 labelGap = kMarkerSpanLabelGapPx * pixelSize.x;
        const F32 tagGap = kMarkerSpanTagGapPx * pixelSize.x;
        const auto fillSpan = [&](Span& span, const U64 leftIndex, const U64 rightIndex) {
            const auto& leftRow = rows[leftIndex];
            const auto& rightRow = rows[rightIndex];
            span.left = leftRow.tagPosition.x + leftRow.tagWidth * 0.5f + tagGap;
            span.right = rightRow.tagPosition.x - rightRow.tagWidth * 0.5f - tagGap;
            span.label = formatSpanX(context, markerPositions[rightIndex] - markerPositions[leftIndex]);
            const F32 labelWidth = markerText->advance(span.label, kLabelScale) * pixelSize.x;
            span.active = span.right - span.left >= (arrowLength + labelGap) * 2.0f + labelWidth;
            span.labelPosition = {(span.left + span.right) * 0.5f, cursorRowCenter};
        };
        for (U64 gap = 0; gap < detail::MarkerSpans; ++gap) {
            if (gaps[gap]) {
                fillSpan(spans[gap], order[gap], order[gap + 1]);
            }
        }
    }

    if (markerShapes) {
        JST_CHECK(markerShapes->updatePixelSize({
            2.0f / interaction.viewSize.x,
            2.0f / interaction.viewSize.y,
        }));

        std::span<Extent2D<F32>> positions;
        JST_CHECK(markerShapes->getPositions("markers", positions));
        std::span<Extent2D<F32>> sizes;
        JST_CHECK(markerShapes->getSizes("markers", sizes));
        for (U64 i = 0; i < kMarkerGroups * detail::MaxMarkers; ++i) {
            positions[i] = {-2.0f, -2.0f};
            sizes[i] = {0.0f, 0.0f};
        }

        std::span<Extent2D<F32>> tablePositions;
        std::span<Extent2D<F32>> tableSizes;
        if (markerTableShapes) {
            JST_CHECK(markerTableShapes->updatePixelSize({
                2.0f / interaction.viewSize.x,
                2.0f / interaction.viewSize.y,
            }));
            JST_CHECK(markerTableShapes->getPositions("table", tablePositions));
            JST_CHECK(markerTableShapes->getSizes("table", tableSizes));
            for (U64 i = 0; i < kMarkerTableGroups * detail::MaxMarkers; ++i) {
                tablePositions[i] = {-2.0f, -2.0f};
                tableSizes[i] = {0.0f, 0.0f};
            }
        }

        const F32 padX = 9.0f;
        const F32 padY = 4.0f;
        const F32 rowGap = 4.0f;
        const F32 pillHeight = (padY * 2.0f + lineHeight) * pixelSize.y;
        const F32 right = padding.x - kMarkerTableGapPx * pixelSize.x;
        const F32 bottom = -padding.y + kMarkerTableGapPx * pixelSize.y;

        for (U64 i = 0; i < count; ++i) {
            auto& row = rows[i];

            if (row.inView) {
                positions[MarkerInstance(kMarkerLine, i)] = {row.xNdc, 0.0f};
                sizes[MarkerInstance(kMarkerLine, i)] = {2.0f * scale, padding.y * 2.0f * toPixelsY};
            }

            if (row.dot) {
                positions[MarkerInstance(kMarkerHalo, i)] = {row.xNdc, row.yNdc};
                sizes[MarkerInstance(kMarkerHalo, i)] = {13.0f * scale, 13.0f * scale};
                positions[MarkerInstance(kMarkerDot, i)] = {row.xNdc, row.yNdc};
                sizes[MarkerInstance(kMarkerDot, i)] = {8.0f * scale, 8.0f * scale};
            }

            if (markerText && table) {
                const F32 idWidth = badgeText->advance(row.id, kLabelScale);
                const F32 idGap = 8.0f;
                const F32 xWidth = markerText->advance(row.x, kLabelScale);
                const F32 yWidth = row.y.empty() ? 0.0f : markerText->advance(row.y, kLabelScale);
                const F32 gap = row.y.empty() ? 0.0f : 10.0f;
                const F32 pillWidth =
                    (padX * 2.0f + idWidth + idGap + xWidth + gap + yWidth) * pixelSize.x;
                const F32 left = right - pillWidth;
                const F32 centerY = bottom + (pillHeight + rowGap * pixelSize.y) * (count - 1 - i) +
                                    pillHeight * 0.5f;
                const F32 centerX = left + pillWidth * 0.5f;

                if (markerTableShapes) {
                    tablePositions[MarkerInstance(kMarkerPillEdge, i)] = {centerX, centerY};
                    tableSizes[MarkerInstance(kMarkerPillEdge, i)] = {
                        pillWidth * toPixelsX + 2.0f * scale,
                        pillHeight * toPixelsY + 2.0f * scale,
                    };
                    tablePositions[MarkerInstance(kMarkerPill, i)] = {centerX, centerY};
                    tableSizes[MarkerInstance(kMarkerPill, i)] = {pillWidth * toPixelsX,
                                                                  pillHeight * toPixelsY};
                }

                row.idPosition = {left + padX * pixelSize.x, centerY};
                row.xPosition = {left + (padX + idWidth + idGap) * pixelSize.x, centerY};
                row.yPosition = {left + (padX + idWidth + idGap + xWidth + gap) * pixelSize.x,
                                 centerY};
            }
        }

        JST_CHECK(markerShapes->updatePositions("markers"));
        JST_CHECK(markerShapes->updateSizes("markers"));
        if (markerTableShapes) {
            JST_CHECK(markerTableShapes->updatePositions("table"));
            JST_CHECK(markerTableShapes->updateSizes("table"));
        }
    }

    if (markerTagShapes) {
        JST_CHECK(markerTagShapes->updatePixelSize({
            2.0f / interaction.viewSize.x,
            2.0f / interaction.viewSize.y,
        }));
        JST_CHECK(markerTagShapes->updateProperties("tags", 4.0f * interaction.scale, 0.0f, {}));

        std::span<Extent2D<F32>> positions;
        JST_CHECK(markerTagShapes->getPositions("tags", positions));
        std::span<Extent2D<F32>> sizes;
        JST_CHECK(markerTagShapes->getSizes("tags", sizes));
        for (U64 i = 0; i < detail::MaxMarkers; ++i) {
            positions[i] = {-2.0f, -2.0f};
            sizes[i] = {0.0f, 0.0f};
        }

        for (U64 i = 0; i < count; ++i) {
            const auto& row = rows[i];
            if (!row.tagged) {
                continue;
            }
            positions[i] = row.tagPosition;
            sizes[i] = {row.tagWidth * toPixelsX, tagHeight * toPixelsY};
        }

        JST_CHECK(markerTagShapes->updatePositions("tags"));
        JST_CHECK(markerTagShapes->updateSizes("tags"));
    }

    if (markerSpanShapes) {
        JST_CHECK(markerSpanShapes->updatePixelSize({
            2.0f / interaction.viewSize.x,
            2.0f / interaction.viewSize.y,
        }));

        const F32 thickness = kMarkerSpanThicknessPx * scale;
        JST_CHECK(markerSpanShapes->updateProperties("spans", thickness * 0.5f, 0.0f, {}));
        JST_CHECK(markerSpanShapes->updateProperties("arrows", thickness * 0.5f, 0.0f, {}));

        std::span<Extent2D<F32>> positions;
        JST_CHECK(markerSpanShapes->getPositions("spans", positions));
        std::span<Extent2D<F32>> sizes;
        JST_CHECK(markerSpanShapes->getSizes("spans", sizes));
        for (U64 i = 0; i < kSpanSegments * detail::MarkerSpans; ++i) {
            positions[i] = {-2.0f, -2.0f};
            sizes[i] = {0.0f, 0.0f};
        }
        std::span<Extent2D<F32>> arrowPositions;
        JST_CHECK(markerSpanShapes->getPositions("arrows", arrowPositions));
        std::span<Extent2D<F32>> arrowSizes;
        JST_CHECK(markerSpanShapes->getSizes("arrows", arrowSizes));
        std::span<F32> arrowRotations;
        JST_CHECK(markerSpanShapes->getRotations("arrows", arrowRotations));
        for (U64 i = 0; i < kSpanArrows * detail::MarkerSpans; ++i) {
            arrowPositions[i] = {-2.0f, -2.0f};
            arrowSizes[i] = {0.0f, 0.0f};
            arrowRotations[i] = 0.0f;
        }

        for (U64 i = 0; i < detail::MarkerSpans; ++i) {
            const auto& span = spans[i];
            if (!span.active || !markerText) {
                continue;
            }
            const F32 arrowAngle = glm::radians(kMarkerSpanArrowAngleDeg);
            const F32 armReach = (kMarkerSpanArrowArmPx - kMarkerSpanThicknessPx) * 0.5f;
            const F32 armDx = armReach * std::cos(arrowAngle) * pixelSize.x;
            const F32 armDy = armReach * std::sin(arrowAngle) * pixelSize.y;
            const F32 labelGap = kMarkerSpanLabelGapPx * pixelSize.x;
            const F32 labelHalf = markerText->advance(span.label, kLabelScale) * pixelSize.x * 0.5f;
            const F32 leadStart = span.left;
            const F32 trailEnd = span.right;
            const F32 leadEnd = std::max(span.labelPosition.x - labelHalf - labelGap, leadStart);
            const F32 trailStart = std::min(span.labelPosition.x + labelHalf + labelGap, trailEnd);

            const auto arm = [&](const U64 arrow, const F32 tipX, const F32 dx, const F32 dy, const F32 degrees) {
                arrowPositions[ArrowInstance(i, arrow)] = {tipX + dx, cursorRowCenter + dy};
                arrowSizes[ArrowInstance(i, arrow)] = {kMarkerSpanArrowArmPx * scale, thickness};
                arrowRotations[ArrowInstance(i, arrow)] = degrees;
            };
            arm(kSpanLeadUpperArm, span.left, armDx, armDy, kMarkerSpanArrowAngleDeg);
            arm(kSpanLeadLowerArm, span.left, armDx, -armDy, -kMarkerSpanArrowAngleDeg);
            arm(kSpanTrailUpperArm, span.right, -armDx, armDy, -kMarkerSpanArrowAngleDeg);
            arm(kSpanTrailLowerArm, span.right, -armDx, -armDy, kMarkerSpanArrowAngleDeg);

            positions[SpanInstance(i, kSpanLeadSegment)] = {(leadStart + leadEnd) * 0.5f, cursorRowCenter};
            sizes[SpanInstance(i, kSpanLeadSegment)] = {(leadEnd - leadStart) * toPixelsX, thickness};
            positions[SpanInstance(i, kSpanTrailSegment)] = {(trailStart + trailEnd) * 0.5f, cursorRowCenter};
            sizes[SpanInstance(i, kSpanTrailSegment)] = {(trailEnd - trailStart) * toPixelsX, thickness};
        }

        JST_CHECK(markerSpanShapes->updatePositions());
        JST_CHECK(markerSpanShapes->updateSizes());
        JST_CHECK(markerSpanShapes->updateRotations());
    }

    if (markerText) {
        for (U64 i = 0; i < detail::MaxMarkers; ++i) {
            const auto& row = rows[i];
            const bool active = i < count;

            auto tag = tagText->get(MarkerElement(i, "tag"));
            tag.position = row.tagPosition;
            tag.fill = (active && row.inView) ? row.tag : " ";
            JST_CHECK(tagText->update(MarkerElement(i, "tag"), tag));

            auto id = badgeText->get(MarkerElement(i, "id"));
            id.position = row.idPosition;
            id.fill = (active && table) ? row.id : " ";
            JST_CHECK(badgeText->update(MarkerElement(i, "id"), id));

            auto x = markerText->get(MarkerElement(i, "x"));
            x.position = row.xPosition;
            x.fill = (active && table) ? row.x : " ";
            JST_CHECK(markerText->update(MarkerElement(i, "x"), x));

            auto y = markerText->get(MarkerElement(i, "y"));
            y.position = row.yPosition;
            y.fill = (active && table && !row.y.empty()) ? row.y : " ";
            JST_CHECK(markerText->update(MarkerElement(i, "y"), y));
        }

        for (U64 i = 0; i < detail::MarkerSpans; ++i) {
            const auto& span = spans[i];

            auto label = markerText->get(SpanElement(i, "label"));
            label.position = span.labelPosition;
            label.fill = span.active ? span.label : " ";
            JST_CHECK(markerText->update(SpanElement(i, "label"), label));
        }
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
