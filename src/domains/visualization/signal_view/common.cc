#include "common.hh"

#include <any>
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
constexpr F32 kLabelScale = 0.85f;

enum CursorInstance : U64 {
    kCursorLine = 0,
    kCursorHalo,
    kCursorMarker,
    kCursorPillEdge,
    kCursorPill,
    kCursorInstances,
};

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

void SignalViewCanvas::processInputEvents(std::vector<InputEvent>&& events,
                                          const SplitEdits& edits) {
    const auto& paddingScale = axis->paddingScale();
    const F32 splitRatio = edits.ratio;
    const F32 previousRatio = splitter.ratio;
    if (!splitter.dragging && !edits.pending()) {
        splitter.ratio = splitRatio;
    }

    std::vector<InputEvent> plotEvents;
    for (const auto& input : events) {
        if (const auto* key = std::get_if<KeyEvent>(&input)) {
            if (key->type == KeyEventType::Press && key->key == KeyCode::Space && !key->repeat) {
                displayHeld = !displayHeld;
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
        const auto layout = detail::CalculateSignalViewPanels(paddingScale,
                                                               interaction.viewSize,
                                                               splitter.ratio);
        bool commit = false;
        if (splitter.process(event, layout, interaction.viewSize,
                              interaction.scale, edits.enabled, commit)) {
            // Splitter capture must never start or continue a horizontal pan.
            interaction.dragging = false;
            if (commit && (splitter.ratio != splitRatio || edits.pending())) {
                if (edits.request(splitter.ratio) != Result::SUCCESS) {
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
    JST_CHECK(updateCursor(context));

    JST_CHECK(axis->present());
    if (text) {
        JST_CHECK(text->present());
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

Result SignalViewCanvas::updateCursor(const Context& context) {
    const auto& config = context.config;
    const bool lineplot = context.lineplot != nullptr;
    const auto& padding = axis->paddingScale();
    const F32 u = (cursor.position.x - 0.5f) / std::max(padding.x, 1e-6f) + 0.5f;
    const F32 v = (cursor.position.y - 0.5f) / std::max(padding.y, 1e-6f) + 0.5f;
    const bool visible = cursor.inside &&
                         !splitter.dragging &&
                         interaction.placement != SurfacePlacementType::Attached &&
                         u >= 0.0f && u <= 1.0f && v >= 0.0f && v <= 1.0f &&
                         context.numberOfElements >= 2;

    std::string xLabelText;
    std::string yLabelText;
    F32 xNdc = 0.0f;
    F32 yNdc = 0.0f;
    bool hasMarker = false;

    if (visible) {
        const F32 translation = PanTranslation(interaction);
        const F32 xPoint = std::clamp((u * 2.0f - 1.0f) / interaction.zoom - translation,
                                      -1.0f, 1.0f);
        xNdc = std::clamp((xPoint + translation) * interaction.zoom, -1.0f, 1.0f) * padding.x;

        const auto& frequency = context.frequency;
        if (frequency.valid) {
            xLabelText = jst::fmt::format("{:.4f} MHz",
                                          (frequency.center +
                                           xPoint * frequency.sampleRate / 2.0f) / 1e6f);
        } else {
            xLabelText = jst::fmt::format("{:.4f}", lineplot
                                                        ? xPoint
                                                        : (xPoint + 1.0f) * 0.5f);
        }

        const auto yPoint = lineplot ? context.lineplot->sample(xPoint) : std::nullopt;
        if (yPoint) {
            if (const auto value = detail::LineplotAmplitudeValue(*yPoint,
                                                                 config.rangeMin,
                                                                 config.rangeMax)) {
                const auto unit = detail::LabelUnit(config.amplitudeLabel);
                yLabelText = unit.empty()
                    ? jst::fmt::format("{:.1f}", *value)
                    : jst::fmt::format("{:.1f} {}", *value, unit);
                const bool combined = context.waterfall != nullptr;
                const F32 lineFraction = combined ? axis->getConfig().verticalScale : 1.0f;
                yNdc = padding.y * (1.0f - lineFraction) +
                       padding.y * lineFraction * std::clamp(*yPoint, -1.0f, 1.0f);
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

            positions[kCursorLine] = {xNdc, 0.0f};
            sizes[kCursorLine] = {2.0f * scale, padding.y * 2.0f * toPixelsY};

            if (hasMarker) {
                positions[kCursorHalo] = {xNdc, yNdc};
                sizes[kCursorHalo] = {16.0f * scale, 16.0f * scale};
                positions[kCursorMarker] = {xNdc, yNdc};
                sizes[kCursorMarker] = {11.0f * scale, 11.0f * scale};
            }

            if (cursorText) {
                const auto& font = cursorText->getConfig().font;
                const F32 lineHeight = font ? font->lineHeight() * kLabelScale : 0.0f;
                const F32 xWidth = cursorText->advance(xLabelText) * kLabelScale;
                const F32 yWidth = yLabelText.empty()
                    ? 0.0f
                    : cursorText->advance(yLabelText) * kLabelScale;
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
        auto xLabelElement = cursorText->get("cursor-x");
        xLabelElement.position = xLabelPosition;
        xLabelElement.fill = visible ? xLabelText : " ";
        JST_CHECK(cursorText->update("cursor-x", xLabelElement));

        auto yLabelElement = cursorText->get("cursor-y");
        yLabelElement.position = yLabelPosition;
        yLabelElement.fill = (visible && !yLabelText.empty()) ? yLabelText : " ";
        JST_CHECK(cursorText->update("cursor-y", yLabelElement));
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
