---
title: Sakura UI
description: CyberEther's retained, GPU-native component tree for building compositor views, layouts, themes, and embedded surfaces.
order: 355
category: Development
---

Sakura (`Jetstream::Sakura`) is CyberEther's retained, GPU-native user interface toolkit. It powers both the default flowgraph workbench and [Superluminal](/docs/superluminal), replacing earlier immediate-mode ImGui and ImNodes widgets with a unified component tree rendered directly through CyberEther's Metal, Vulkan, and WebGPU backends.

This guide is for contributors building or extending compositor views, settings panels, or custom frontends on top of Sakura.

## The Retained Component Model

Unlike immediate-mode UI libraries that rebuild every widget from scratch on every frame, Sakura maintains a persistent tree of `Sakura::Component` objects:

- **Owners keep components alive.** A view or presenter owns its child widgets as member fields (components are non-copyable and non-movable) and passes child pointers through widget configs. Custom `Component` subclasses manage their own children through protected helpers (`addChild()`, `removeChild()`, `clearChildren()`, `setChildren()`, and `mountChildren()`).
- **Concrete widgets update via `Config`.** Built-in widgets follow a value-type `Config` + `update(Config)` convention (not virtual methods on `Component`). Calling `update()` compares the incoming config against the current state and invalidates only what changed.
- **Invalidation is categorized.** Calling `invalidate(Dirty)` marks a component with one of five `Component::Dirty` categories: `Paint`, `Resource`, `Tree`, `Order`, or `Layout`. Unchanged subtrees avoid unnecessary layout and paint work during normal edits, and the interface only redraws when something changes (full canvas resource rebuilds still walk the tree as needed).

### Minimal Retained View Example

```cpp
#include <jetstream/render/sakura/base.hh>

using namespace Jetstream;

struct StatusPanel {
    Sakura::Div card;
    Sakura::Stack row;
    Sakura::Text label;
    Sakura::Badge badge;

    void update(const std::string& title, bool active) {
        label.update({
            .str = title,
            .font = Sakura::Text::Font::Display,
            .colorKey = "text_primary",
            .scale = Sakura::Typography::Scale::Lead,
        });
        badge.update({
            .label = active ? "Running" : "Paused",
        });
        row.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = Sakura::Spacing::Medium,
            .items = {
                {.child = &label},
                {.child = &badge},
            },
        });
        card.update({
            .content = &row,
            .size = {Sakura::Fill, 0.0f},
            .padding = Sakura::Spacing::Medium,
            .rounding = Sakura::Rounding::Card,
            .border = true,
        });
    }
};
```

### Component Lifecycle Hooks

Custom components derive from `Sakura::Component` (`include/jetstream/render/sakura/component.hh`) and override only the hooks they need:

| Hook | Purpose |
|---|---|
| `measure(ctx, available)` | Computes and returns the component's resolved preferred extent within the available bounds. |
| `layout(ctx)` | Positions child components via `layoutChild()` and sets clipping rectangles. |
| `build(ctx)` | Creates or updates GPU render primitives when `Dirty::Resource` is flagged. |
| `paint()` | Updates per-frame visual properties when `Dirty::Paint` is flagged. |
| `conceal()` | Hides render primitives when a component becomes invisible. |
| `wentOnscreen(frame)` / `wentOffscreen(frame)` | Runs during presentation according to tree visibility (distinct from destruction), letting live surfaces pause or resume work. |

Widget configs that accept an explicit size use `Sakura::Fill` (`-1.0f`) as a sentinel to request the full available width or height.

## Layout and Stacking Layers

Sakura separates layout arrangement from window-wide z-ordering:

- **Layout containers:** Compose interfaces with `Stack` (horizontal or vertical), `Grid`, `Div`, `Aside`, `ScrollArea`, `Dock`, `Panel`, `Spacer`, `Overlay`, `Reveal`, and `Popup`.
- **Stacking layers:** The top-level `Sakura::Root` (backed by `Sakura::Canvas`) renders an explicit list of `Layer*` children via `Root::Config::layers`. Stacking order comes strictly from the order passed to `Root::Config::layers`, not from layer class names. Built-in layer types include `BackgroundLayer`, `HudLayer`, `FloatingLayer`, `ModalLayer`, and `ToastLayer`.

## Palette, Style, and Typography

Visual consistency across the workbench comes from three lightweight headers:

- **Palette (`palette.hh`):** Defines `Sakura::Palette` as a `std::unordered_map<std::string, ColorRGBA<F32>>` mapping semantic color keys (such as `"text_primary"`) to RGBA colors. Internal compositor and render theme helpers (such as `MakeSurfaceTheme()` in `src/render/sakura/themes.hh`) derive matching visualization surface colors from the active palette.
- **Style tokens (`style.hh`):** Namespace-level structs define standard geometry constants:
  - **Rounding (`Sakura::Rounding`):** `Control` (`8`), `Card` (`12`), and `Popup` (`18`).
  - **Spacing (`Sakura::Spacing`):** `XSmall` (`4`), `Small` (`8`), `Medium` (`12`), `Large` (`16`), `XLarge` (`24`), and `XXLarge` (`32`).
- **Typography (`typography.hh`):** Base `Typography::FontSize` is `15.0f`, paired with `Typography::Scale` steps (`Small`, `Caption`, `Body`, `Lead`, `Large`, `Title`, `Heading`, `Headline`). Keep font family mapping in mind when configuring `Sakura::Text`: `Font::Display` and `Font::DisplaySemiBold` use the display fonts, `Font::Sans` and `Font::SansBold` use the proportional body fonts, and `Font::Body`, `Font::H1`, `Font::H2`, and `Font::Bold` resolve to the monospace font family.

## Input Routing, Focus, and Shortcuts

Pointer, scroll, keyboard, and text events travel through Sakura's input router to component hooks (`mouseEvent`, `scrollPhaseEnded`, `keyEvent`, `textEvent`, `shortcutEvent`, and `focusChanged`), while file drops enter through `Canvas::dropFiles()` and dispatch to `fileDrop`.

- **Focus management:** Components coordinate keyboard focus using `requestFocus()`, `offerFocus()`, `releaseFocus()`, `releaseFocusWithin()`, `hasFocus()`, and `textEditorFocused()`.
- **Clipboard:** Use `Sakura::SetClipboardText(window, value)` and `Sakura::GetClipboardText(window)` (`clipboard.hh`) for platform-independent clipboard access.
- **Keyboard shortcuts:** Wrap any subtree in `Sakura::Shortcuts` (`components/shortcuts.hh`) to bind key combinations:

```cpp
keys.update({
    .bindings = {
        {
            .key = KeyCode::K,
            .modifier = Sakura::Shortcuts::Modifier::Primary,
            .yieldsToText = true,
            .onPressed = [this]() { openSearch(); },
        },
    },
    .content = &page,
});
```

The `Modifier::Primary` option matches either Control or Super (Command) on every platform, while `Shortcuts::PrimaryName()` formats the displayed label for the current OS. Set `yieldsToText = true` on bindings that should defer when a text input or code editor holds focus.

## Embedding GPU Surfaces

Live visualization blocks (`SURFACE`-tainted modules) render into GPU textures that Sakura displays inside node cards and detached dock tiles via `Sakura::SurfaceImage` (`components/surface_image.hh`):

```cpp
surfaceImage.update({
    .texture = moduleSurfaceTexture,
    .rounding = Sakura::Rounding::Control,
    .onSize = [this](const Sakura::SurfaceResize& resize) {
        updateSurfaceExtent(resize.logicalSize, resize.framebufferSize, resize.scale);
    },
    .onInput = [this](InputEvent event) {
        surface->pushInputEvent(std::move(event));
    },
});
```

- The `onSize` callback delivers a `Sakura::SurfaceResize` containing `logicalSize`, `framebufferSize`, and display `scale`.
- The `onInput` callback delivers `InputEvent` values (`MouseEvent`, `KeyEvent`, and `FocusEvent`). Forwarding them to `Module::Surface::pushInputEvent()` is explicit.

For the module side of surface input (normalized coordinates, focus-loss cleanup, and `ProcessSurfaceInteraction()`), see [Interactive Surface Input](/docs/module-contract#interactive-surface-input).

## Built-In Component Families

Headers under `include/jetstream/render/sakura/components/` group into five main families:

| Family | Key Components |
|---|---|
| **Layout and navigation** | `Stack`, `Grid`, `Div`, `Aside`, `Dock`, `Panel`, `ScrollArea`, `SidebarItem`, `Outline`, `StatusBar` |
| **Controls and forms** | `Button`, `PillButton`, `ChromeButton`, `Toggle`, `Combo`, `Dropdown`, `TextInput`, `PillInput`, `PillStepper`, `PillTabs`, `SettingCard`, `SettingField` |
| **Drawing, content, and data** | `Box`, `Card`, `Text`, `Label`, `Badge`, `Markdown`, `Table`, `Image`, `SurfaceImage`, `Curves`, `Polyline`, `Sparkline`, `ProgressBar`, `QrCode` |
| **Editors and workbench tools** | `CodeEditor`, `MarkdownEditor`, `TextEditor`, `Console`, `LogView`, `Terminal`, `Chat`, `SearchCatalog`, `ModelPicker`, `node/` |
| **Overlays and feedback** | `ModalLayer`, `FloatingLayer`, `HudLayer`, `ToastLayer`, `Overlay`, `Popup`, `Menu`, `Spinner`, `TipBanner` |
