#include "terminal.hh"

#include "jetstream/logger.hh"
#include "jetstream/platform.hh"
#include "jetstream/render/components/text.hh"

#if defined(JETSTREAM_LOADER_GHOSTTY_VT_AVAILABLE)
#include <ghostty/vt.h>
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace Jetstream {

bool TerminalSession::AltSendsEscape() {
#if defined(JST_OS_MAC)
    return false;
#else
    return true;
#endif
}

#if defined(JETSTREAM_LOADER_GHOSTTY_VT_AVAILABLE)

namespace {

using Unicode = Render::Components::Text::Unicode;
using Screen = Sakura::Terminal::Screen;
using Row = Sakura::Terminal::Row;
using Span = Sakura::Terminal::Span;
using Link = Sakura::Terminal::Link;

constexpr U64 kReadBudget = 4 * 1024 * 1024;
constexpr size_t kScrollbackLines = 10000;
constexpr U16 kInitialColumns = 80;
constexpr U16 kInitialRows = 24;
constexpr U32 kMaxGraphemeCodepoints = 16;
constexpr F32 kFaintAlpha = 0.5f;
constexpr const char* kVersion = "CyberEther";
constexpr auto kKeystrokeSpacing = std::chrono::milliseconds(8);
constexpr auto kSyncTimeout = std::chrono::milliseconds(1000);

using Clock = std::chrono::steady_clock;

GhosttyColorRgb ToGhostty(const ColorRGBA<F32>& color) {
    const auto channel = [](F32 value) {
        return static_cast<uint8_t>(std::clamp(std::round(value * 255.0f), 0.0f, 255.0f));
    };
    return {channel(color.r), channel(color.g), channel(color.b)};
}

ColorRGBA<F32> FromGhostty(const GhosttyColorRgb& color) {
    return {static_cast<F32>(color.r) / 255.0f,
            static_cast<F32>(color.g) / 255.0f,
            static_cast<F32>(color.b) / 255.0f,
            1.0f};
}

struct KeyMapping {
    GhosttyKey key = GHOSTTY_KEY_UNIDENTIFIED;
    U32 codepoint = 0;
};

KeyMapping MapKey(KeyCode code) {
    const auto offset = [](KeyCode value, KeyCode base) {
        return static_cast<U16>(value) - static_cast<U16>(base);
    };
    if (code >= KeyCode::A && code <= KeyCode::Z) {
        const auto index = offset(code, KeyCode::A);
        return {static_cast<GhosttyKey>(GHOSTTY_KEY_A + index), static_cast<U32>('a' + index)};
    }
    if (code >= KeyCode::Digit0 && code <= KeyCode::Digit9) {
        const auto index = offset(code, KeyCode::Digit0);
        return {static_cast<GhosttyKey>(GHOSTTY_KEY_DIGIT_0 + index), static_cast<U32>('0' + index)};
    }
    if (code >= KeyCode::F1 && code <= KeyCode::F24) {
        return {static_cast<GhosttyKey>(GHOSTTY_KEY_F1 + offset(code, KeyCode::F1)), 0};
    }
    if (code >= KeyCode::Keypad0 && code <= KeyCode::Keypad9) {
        const auto index = offset(code, KeyCode::Keypad0);
        return {static_cast<GhosttyKey>(GHOSTTY_KEY_NUMPAD_0 + index), static_cast<U32>('0' + index)};
    }
    switch (code) {
        case KeyCode::Tab: return {GHOSTTY_KEY_TAB, 0};
        case KeyCode::LeftArrow: return {GHOSTTY_KEY_ARROW_LEFT, 0};
        case KeyCode::RightArrow: return {GHOSTTY_KEY_ARROW_RIGHT, 0};
        case KeyCode::UpArrow: return {GHOSTTY_KEY_ARROW_UP, 0};
        case KeyCode::DownArrow: return {GHOSTTY_KEY_ARROW_DOWN, 0};
        case KeyCode::PageUp: return {GHOSTTY_KEY_PAGE_UP, 0};
        case KeyCode::PageDown: return {GHOSTTY_KEY_PAGE_DOWN, 0};
        case KeyCode::Home: return {GHOSTTY_KEY_HOME, 0};
        case KeyCode::End: return {GHOSTTY_KEY_END, 0};
        case KeyCode::Insert: return {GHOSTTY_KEY_INSERT, 0};
        case KeyCode::Delete: return {GHOSTTY_KEY_DELETE, 0};
        case KeyCode::Backspace: return {GHOSTTY_KEY_BACKSPACE, 0};
        case KeyCode::Space: return {GHOSTTY_KEY_SPACE, ' '};
        case KeyCode::Enter: return {GHOSTTY_KEY_ENTER, 0};
        case KeyCode::Escape: return {GHOSTTY_KEY_ESCAPE, 0};
        case KeyCode::Apostrophe: return {GHOSTTY_KEY_QUOTE, '\''};
        case KeyCode::Comma: return {GHOSTTY_KEY_COMMA, ','};
        case KeyCode::Minus: return {GHOSTTY_KEY_MINUS, '-'};
        case KeyCode::Period: return {GHOSTTY_KEY_PERIOD, '.'};
        case KeyCode::Slash: return {GHOSTTY_KEY_SLASH, '/'};
        case KeyCode::Semicolon: return {GHOSTTY_KEY_SEMICOLON, ';'};
        case KeyCode::Equal: return {GHOSTTY_KEY_EQUAL, '='};
        case KeyCode::LeftBracket: return {GHOSTTY_KEY_BRACKET_LEFT, '['};
        case KeyCode::Backslash: return {GHOSTTY_KEY_BACKSLASH, '\\'};
        case KeyCode::RightBracket: return {GHOSTTY_KEY_BRACKET_RIGHT, ']'};
        case KeyCode::GraveAccent: return {GHOSTTY_KEY_BACKQUOTE, '`'};
        case KeyCode::KeypadDecimal: return {GHOSTTY_KEY_NUMPAD_DECIMAL, '.'};
        case KeyCode::KeypadDivide: return {GHOSTTY_KEY_NUMPAD_DIVIDE, '/'};
        case KeyCode::KeypadMultiply: return {GHOSTTY_KEY_NUMPAD_MULTIPLY, '*'};
        case KeyCode::KeypadSubtract: return {GHOSTTY_KEY_NUMPAD_SUBTRACT, '-'};
        case KeyCode::KeypadAdd: return {GHOSTTY_KEY_NUMPAD_ADD, '+'};
        case KeyCode::KeypadEnter: return {GHOSTTY_KEY_NUMPAD_ENTER, 0};
        case KeyCode::KeypadEqual: return {GHOSTTY_KEY_NUMPAD_EQUAL, '='};
        default: return {};
    }
}

GhosttyMods MapModifiers(const KeyModifiers& modifiers) {
    GhosttyMods mods = 0;
    if (modifiers.shift) {
        mods |= GHOSTTY_MODS_SHIFT;
    }
    if (modifiers.control) {
        mods |= GHOSTTY_MODS_CTRL;
    }
    if (modifiers.alt) {
        mods |= GHOSTTY_MODS_ALT;
    }
    if (modifiers.super) {
        mods |= GHOSTTY_MODS_SUPER;
    }
    return mods;
}

U32 Shifted(U32 codepoint, bool shift) {
    if (!shift) {
        return codepoint;
    }
    if (codepoint >= 'a' && codepoint <= 'z') {
        return codepoint - 'a' + 'A';
    }
    static constexpr std::string_view kPlain = "1234567890-=[]\\;',./`";
    static constexpr std::string_view kShifted = "!@#$%^&*()_+{}|:\"<>?~";
    if (codepoint < 0x80) {
        if (const auto index = kPlain.find(static_cast<char>(codepoint)); index != std::string_view::npos) {
            return static_cast<U8>(kShifted[index]);
        }
    }
    return codepoint;
}

Sakura::Terminal::CursorShape CursorShapeOf(GhosttyRenderStateCursorVisualStyle style) {
    switch (style) {
        case GHOSTTY_RENDER_STATE_CURSOR_VISUAL_STYLE_BAR:
            return Sakura::Terminal::CursorShape::Bar;
        case GHOSTTY_RENDER_STATE_CURSOR_VISUAL_STYLE_UNDERLINE:
            return Sakura::Terminal::CursorShape::Underline;
        case GHOSTTY_RENDER_STATE_CURSOR_VISUAL_STYLE_BLOCK_HOLLOW:
            return Sakura::Terminal::CursorShape::Hollow;
        default:
            return Sakura::Terminal::CursorShape::Block;
    }
}

bool PointBefore(const GhosttyPointCoordinate& a, const GhosttyPointCoordinate& b) {
    return a.y < b.y || (a.y == b.y && a.x < b.x);
}

struct LinkCell {
    U16 row = 0;
    U16 column = 0;
    std::string text;
};

struct FoundLink {
    U64 first = 0;
    U64 last = 0;
    bool open = false;
    std::string uri;
};

std::vector<FoundLink> DetectLinks(const std::vector<LinkCell>& cells) {
    static constexpr std::array<std::string_view, 4> kSchemes = {"https://", "http://", "file://", "mailto:"};
    static constexpr std::string_view kStops = " \t<>\"'`";
    static constexpr std::string_view kTrailing = ".,;:!?";
    std::string line;
    std::vector<U64> owners;
    for (U64 index = 0; index < cells.size(); ++index) {
        const std::string shown = cells[index].text.empty() ? std::string(" ") : cells[index].text;
        line += shown;
        owners.insert(owners.end(), shown.size(), index);
    }
    std::vector<FoundLink> found;
    U64 position = 0;
    while (position < line.size()) {
        U64 start = std::string_view::npos;
        U64 scheme = 0;
        for (const auto candidate : kSchemes) {
            const U64 match = line.find(candidate, position);
            if (match < start) {
                start = match;
                scheme = candidate.size();
            }
        }
        if (start == std::string_view::npos) {
            break;
        }
        U64 end = start + scheme;
        while (end < line.size() && kStops.find(line[end]) == std::string_view::npos) {
            ++end;
        }
        const bool open = end == line.size();
        while (end > start + scheme) {
            const char last = line[end - 1];
            const auto unbalanced = [&](char open, char close) {
                return last == close && std::count(line.begin() + start, line.begin() + end, open) <
                                            std::count(line.begin() + start, line.begin() + end, close);
            };
            if (kTrailing.find(last) == std::string_view::npos && !unbalanced('(', ')') && !unbalanced('[', ']')) {
                break;
            }
            --end;
        }
        const bool word = start > 0 && std::isalnum(static_cast<unsigned char>(line[start - 1]));
        if (end > start + scheme && !word) {
            found.push_back({.first = owners[start], .last = owners[end - 1], .open = open,
                             .uri = line.substr(start, end - start)});
        }
        position = std::max(end, start + 1);
    }
    return found;
}

}  // namespace

struct TerminalSession::Impl {
    struct Style {
        ColorRGBA<F32> color;
        bool bold = false;
        bool underline = false;
        bool strikethrough = false;

        bool operator==(const Style&) const = default;
    };

    struct Anchor {
        GhosttyPointCoordinate start{};
        GhosttyPointCoordinate end{};
        U8 clicks = 1;
    };

    struct Outgoing {
        std::string data;
        bool keystroke = false;
    };

    Platform::PseudoTerminal pty;
    GhosttyTerminal terminal = nullptr;
    GhosttyRenderState renderState = nullptr;
    GhosttyRenderStateRowIterator rowIterator = nullptr;
    GhosttyRenderStateRowCells rowCells = nullptr;
    GhosttyKeyEncoder keyEncoder = nullptr;
    GhosttyKeyEvent keyEvent = nullptr;
    GhosttyMouseEncoder mouseEncoder = nullptr;
    GhosttyMouseEvent mouseEvent = nullptr;

    Screen screen;
    std::vector<uint32_t> graphemes;
    std::vector<std::vector<std::pair<U16, std::string>>> texts;
    std::vector<std::vector<Link>> hyperlinks;
    std::vector<U8> wraps;
    U64 revision = 0;
    bool started = false;
    bool rebuildAll = true;
    std::string title;
    Theme palette;
    Theme colors;
    Sakura::Terminal::Grid grid = {kInitialColumns, kInitialRows, 8.0f, 16.0f};
    std::optional<Anchor> anchor;
    bool selected = false;
    std::optional<MouseButton> pressedButton;
    Extent2D<F32> pointerPosition = {0.0f, 0.0f};
    std::optional<Clock::time_point> holdSince;
    bool captured = false;
    std::thread io;
    std::mutex mutex;
    std::deque<Outgoing> outbox;
    std::string inbox;
    std::string received;
    bool stopping = false;

    ~Impl() {
        release();
    }

    void release() {
        if (mouseEvent) {
            ghostty_mouse_event_free(mouseEvent);
            mouseEvent = nullptr;
        }
        if (mouseEncoder) {
            ghostty_mouse_encoder_free(mouseEncoder);
            mouseEncoder = nullptr;
        }
        if (keyEvent) {
            ghostty_key_event_free(keyEvent);
            keyEvent = nullptr;
        }
        if (keyEncoder) {
            ghostty_key_encoder_free(keyEncoder);
            keyEncoder = nullptr;
        }
        if (rowCells) {
            ghostty_render_state_row_cells_free(rowCells);
            rowCells = nullptr;
        }
        if (rowIterator) {
            ghostty_render_state_row_iterator_free(rowIterator);
            rowIterator = nullptr;
        }
        if (renderState) {
            ghostty_render_state_free(renderState);
            renderState = nullptr;
        }
        if (terminal) {
            ghostty_terminal_free(terminal);
            terminal = nullptr;
        }
    }

    static void WritePty(GhosttyTerminal, void* userdata, const uint8_t* data, size_t length) {
        auto* self = static_cast<Impl*>(userdata);
        self->send({reinterpret_cast<const char*>(data), length}, false);
    }

    static void TitleChanged(GhosttyTerminal terminal, void* userdata) {
        auto* self = static_cast<Impl*>(userdata);
        GhosttyString value{};
        if (ghostty_terminal_get(terminal, GHOSTTY_TERMINAL_DATA_TITLE, &value) == GHOSTTY_SUCCESS) {
            self->title = std::string(reinterpret_cast<const char*>(value.ptr), value.len);
        }
    }

    static bool Size(GhosttyTerminal, void* userdata, GhosttySizeReportSize* size) {
        const auto* self = static_cast<Impl*>(userdata);
        size->rows = self->grid.rows;
        size->columns = self->grid.columns;
        size->cell_width = static_cast<uint32_t>(std::max(1.0f, self->grid.cellWidth));
        size->cell_height = static_cast<uint32_t>(std::max(1.0f, self->grid.cellHeight));
        return true;
    }

    static bool DeviceAttributes(GhosttyTerminal, void*, GhosttyDeviceAttributes* attributes) {
        attributes->primary.conformance_level = 62;
        attributes->primary.features[0] = 22;
        attributes->primary.num_features = 1;
        attributes->secondary.device_type = 1;
        attributes->secondary.firmware_version = 10;
        attributes->secondary.rom_cartridge = 0;
        attributes->tertiary.unit_id = 0;
        return true;
    }

    static bool ColorScheme(GhosttyTerminal, void* userdata, GhosttyColorScheme* scheme) {
        const auto& background = static_cast<Impl*>(userdata)->colors.background;
        const F32 luminance = 0.2126f * background.r + 0.7152f * background.g + 0.0722f * background.b;
        *scheme = luminance < 0.5f ? GHOSTTY_COLOR_SCHEME_DARK : GHOSTTY_COLOR_SCHEME_LIGHT;
        return true;
    }

    static void RenderHold(GhosttyTerminal terminal, void* userdata, bool held) {
        auto* self = static_cast<Impl*>(userdata);
        if (held) {
            self->holdSince = Clock::now();
            self->captured = ghostty_render_state_update(self->renderState, terminal) == GHOSTTY_SUCCESS;
            self->rebuild();
        } else {
            self->holdSince.reset();
        }
    }

    static GhosttyString Version(GhosttyTerminal, void*) {
        return {reinterpret_cast<const uint8_t*>(kVersion), std::char_traits<char>::length(kVersion)};
    }

    template<typename Function>
    void setCallback(GhosttyTerminalOption option, Function function) {
        (void)ghostty_terminal_set(terminal, option, reinterpret_cast<const void*>(function));
    }

    bool mode(GhosttyMode value) const {
        GhosttyTerminalModeConfig config{.mode = value, .value = false};
        if (ghostty_terminal_get(terminal, GHOSTTY_TERMINAL_DATA_MODE, &config) != GHOSTTY_SUCCESS) {
            return false;
        }
        return config.value;
    }

    bool alternateScreen() const {
        GhosttyTerminalScreen active = GHOSTTY_TERMINAL_SCREEN_PRIMARY;
        (void)ghostty_terminal_get(terminal, GHOSTTY_TERMINAL_DATA_ACTIVE_SCREEN, &active);
        return active == GHOSTTY_TERMINAL_SCREEN_ALTERNATE;
    }

    bool mouseTracking() const {
        bool tracking = false;
        (void)ghostty_terminal_get(terminal, GHOSTTY_TERMINAL_DATA_MOUSE_TRACKING, &tracking);
        return tracking;
    }

    void applyTheme() {
        const GhosttyColorRgb foreground = ToGhostty(palette.foreground);
        const GhosttyColorRgb background = ToGhostty(palette.background);
        (void)ghostty_terminal_set(terminal, GHOSTTY_TERMINAL_OPT_COLOR_FOREGROUND, &foreground);
        (void)ghostty_terminal_set(terminal, GHOSTTY_TERMINAL_OPT_COLOR_BACKGROUND, &background);
        (void)ghostty_terminal_set(terminal, GHOSTTY_TERMINAL_OPT_COLOR_CURSOR, &foreground);
        rebuildAll = true;
    }

    void send(std::string data, bool keystroke) {
        if (data.empty()) {
            return;
        }
        {
            std::lock_guard lock(mutex);
            outbox.push_back({.data = std::move(data), .keystroke = keystroke});
        }
        pty.interrupt();
    }

    void transfer() {
        Clock::time_point lastKeystroke{};
        std::vector<Outgoing> ready;
        std::string output;
        while (true) {
            std::optional<std::chrono::milliseconds> timeout;
            bool keyed = false;
            U64 room = 0;
            {
                std::lock_guard lock(mutex);
                if (stopping) {
                    return;
                }
                while (!outbox.empty()) {
                    if (outbox.front().keystroke) {
                        const auto elapsed = keyed ? Clock::duration::zero() : Clock::now() - lastKeystroke;
                        if (elapsed < kKeystrokeSpacing) {
                            timeout = std::chrono::ceil<std::chrono::milliseconds>(kKeystrokeSpacing - elapsed);
                            break;
                        }
                        keyed = true;
                    }
                    ready.push_back(std::move(outbox.front()));
                    outbox.pop_front();
                }
                room = kReadBudget - std::min<U64>(kReadBudget, inbox.size());
            }

            for (const auto& outgoing : ready) {
                (void)pty.write(outgoing.data);
            }
            ready.clear();
            if (keyed) {
                lastKeystroke = Clock::now();
            }
            (void)pty.flush();

            if (room > 0) {
                output.clear();
                (void)pty.read(output, room);
                if (!output.empty()) {
                    std::lock_guard lock(mutex);
                    inbox.append(output);
                }
            }

            if (pty.wait(timeout, room > 0) != Result::SUCCESS) {
                break;
            }
        }
    }

    bool held() {
        if (!holdSince.has_value()) {
            return false;
        }
        if (Clock::now() - *holdSince < kSyncTimeout) {
            return true;
        }
        const GhosttyTerminalModeConfig release{.mode = GHOSTTY_MODE_SYNC_OUTPUT, .value = false};
        (void)ghostty_terminal_set(terminal, GHOSTTY_TERMINAL_OPT_MODE, &release);
        holdSince.reset();
        return false;
    }

    void followOutput() {
        GhosttyTerminalScrollViewport behavior{};
        behavior.tag = GHOSTTY_SCROLL_VIEWPORT_BOTTOM;
        ghostty_terminal_scroll_viewport(terminal, behavior);
    }

    void clearSelection() {
        anchor.reset();
        if (selected) {
            selected = false;
            (void)ghostty_terminal_set(terminal, GHOSTTY_TERMINAL_OPT_SELECTION, nullptr);
            screen.selection.clear();
            rebuildAll = true;
        }
    }

    std::string selectionText() const {
        GhosttyTerminalSelectionFormatOptions options = GHOSTTY_INIT_SIZED(GhosttyTerminalSelectionFormatOptions);
        options.emit = GHOSTTY_FORMATTER_FORMAT_PLAIN;
        options.unwrap = true;
        options.trim = true;
        options.selection = nullptr;
        uint8_t* buffer = nullptr;
        size_t length = 0;
        if (ghostty_terminal_selection_format_alloc(terminal, nullptr, options, &buffer, &length) != GHOSTTY_SUCCESS) {
            return {};
        }
        std::string text(reinterpret_cast<const char*>(buffer), length);
        ghostty_free(nullptr, buffer, length);
        return text;
    }

    void encodeKey(GhosttyKey key, GhosttyMods mods, U32 codepoint, bool repeat, U32 text = 0) {
        ghostty_key_encoder_setopt_from_terminal(keyEncoder, terminal);
        ghostty_key_event_set_action(keyEvent, repeat ? GHOSTTY_KEY_ACTION_REPEAT : GHOSTTY_KEY_ACTION_PRESS);
        ghostty_key_event_set_key(keyEvent, key);
        ghostty_key_event_set_mods(keyEvent, mods);
        ghostty_key_event_set_consumed_mods(keyEvent, text != 0 ? (mods & GHOSTTY_MODS_SHIFT) : 0);
        ghostty_key_event_set_unshifted_codepoint(keyEvent, codepoint);
        const std::string utf8 = text != 0 ? Unicode::Encode(text) : std::string();
        ghostty_key_event_set_utf8(keyEvent, utf8.empty() ? nullptr : utf8.data(), utf8.size());
        char buffer[128];
        size_t length = 0;
        if (ghostty_key_encoder_encode(keyEncoder, keyEvent, buffer, sizeof(buffer), &length) == GHOSTTY_SUCCESS &&
            length > 0) {
            send({buffer, length}, true);
        }
    }

    std::optional<GhosttyPointCoordinate> screenPoint(U16 column, U16 row) const {
        GhosttyGridRef ref = GHOSTTY_INIT_SIZED(GhosttyGridRef);
        GhosttyPoint point{};
        point.tag = GHOSTTY_POINT_TAG_VIEWPORT;
        point.value.coordinate = {.x = column, .y = row};
        if (ghostty_terminal_grid_ref(terminal, point, &ref) != GHOSTTY_SUCCESS) {
            return std::nullopt;
        }
        GhosttyPointCoordinate coordinate{};
        if (ghostty_terminal_point_from_grid_ref(terminal, &ref, GHOSTTY_POINT_TAG_SCREEN, &coordinate) !=
            GHOSTTY_SUCCESS) {
            return std::nullopt;
        }
        return coordinate;
    }

    std::optional<GhosttyGridRef> screenRef(const GhosttyPointCoordinate& coordinate) const {
        GhosttyGridRef ref = GHOSTTY_INIT_SIZED(GhosttyGridRef);
        GhosttyPoint point{};
        point.tag = GHOSTTY_POINT_TAG_SCREEN;
        point.value.coordinate = coordinate;
        if (ghostty_terminal_grid_ref(terminal, point, &ref) != GHOSTTY_SUCCESS) {
            return std::nullopt;
        }
        return ref;
    }

    std::optional<std::pair<GhosttyPointCoordinate, GhosttyPointCoordinate>> expand(
        const GhosttyPointCoordinate& coordinate, U8 clicks) const {
        if (clicks < 2) {
            return std::make_pair(coordinate, coordinate);
        }
        const auto ref = screenRef(coordinate);
        if (!ref.has_value()) {
            return std::nullopt;
        }
        GhosttySelection selection = GHOSTTY_INIT_SIZED(GhosttySelection);
        GhosttyResult result = GHOSTTY_NO_VALUE;
        if (clicks == 2) {
            GhosttyTerminalSelectWordOptions options = GHOSTTY_INIT_SIZED(GhosttyTerminalSelectWordOptions);
            options.ref = *ref;
            result = ghostty_terminal_select_word(terminal, &options, &selection);
        } else {
            GhosttyTerminalSelectLineOptions options = GHOSTTY_INIT_SIZED(GhosttyTerminalSelectLineOptions);
            options.ref = *ref;
            result = ghostty_terminal_select_line(terminal, &options, &selection);
        }
        if (result != GHOSTTY_SUCCESS) {
            return std::make_pair(coordinate, coordinate);
        }
        GhosttyPointCoordinate start{};
        GhosttyPointCoordinate end{};
        if (ghostty_terminal_point_from_grid_ref(terminal, &selection.start, GHOSTTY_POINT_TAG_SCREEN, &start) !=
                GHOSTTY_SUCCESS ||
            ghostty_terminal_point_from_grid_ref(terminal, &selection.end, GHOSTTY_POINT_TAG_SCREEN, &end) !=
                GHOSTTY_SUCCESS) {
            return std::make_pair(coordinate, coordinate);
        }
        if (PointBefore(end, start)) {
            std::swap(start, end);
        }
        return std::make_pair(start, end);
    }

    void applySelection(const GhosttyPointCoordinate& start, const GhosttyPointCoordinate& end) {
        const auto first = screenRef(start);
        const auto last = screenRef(end);
        if (!first.has_value() || !last.has_value()) {
            return;
        }
        GhosttySelection selection = GHOSTTY_INIT_SIZED(GhosttySelection);
        selection.start = *first;
        selection.end = *last;
        selection.rectangle = false;
        if (ghostty_terminal_set(terminal, GHOSTTY_TERMINAL_OPT_SELECTION, &selection) == GHOSTTY_SUCCESS) {
            selected = true;
            screen.selection = selectionText();
            rebuildAll = true;
        }
    }

    Style styleOf(const GhosttyStyle& style, std::optional<ColorRGBA<F32>> foreground,
                  std::optional<ColorRGBA<F32>> background, std::optional<ColorRGBA<F32>>& fill) const {
        ColorRGBA<F32> color = foreground.value_or(colors.foreground);
        fill = background;
        if (style.inverse) {
            const ColorRGBA<F32> swapped = background.value_or(colors.background);
            fill = color;
            color = swapped;
        }
        if (style.faint) {
            color.a *= kFaintAlpha;
        }
        return {
            .color = color,
            .bold = style.bold,
            .underline = style.underline != 0,
            .strikethrough = style.strikethrough,
        };
    }

    std::string hyperlinkAt(U16 column, U16 row) const {
        GhosttyPoint point{};
        point.tag = GHOSTTY_POINT_TAG_VIEWPORT;
        point.value.coordinate = {.x = column, .y = row};
        GhosttyGridRef ref = GHOSTTY_INIT_SIZED(GhosttyGridRef);
        if (ghostty_terminal_grid_ref(terminal, point, &ref) != GHOSTTY_SUCCESS) {
            return {};
        }
        size_t length = 0;
        if (ghostty_grid_ref_hyperlink_uri(&ref, nullptr, 0, &length) != GHOSTTY_OUT_OF_SPACE || length == 0) {
            return {};
        }
        std::string uri(length, '\0');
        if (ghostty_grid_ref_hyperlink_uri(&ref, reinterpret_cast<uint8_t*>(uri.data()), uri.size(), &length) !=
            GHOSTTY_SUCCESS) {
            return {};
        }
        uri.resize(length);
        return uri;
    }

    void buildRow(Row& row, U16 rowIndex, const GhosttyRenderStateCursor& cursor) {
        row.spans.clear();
        row.backgrounds.clear();
        row.links.clear();
        row.selection.reset();
        auto& cells = texts[rowIndex];
        cells.clear();
        GhosttyRow rawRow = 0;
        bool wrapped = false;
        if (ghostty_render_state_row_get(rowIterator, GHOSTTY_RENDER_STATE_ROW_DATA_RAW, &rawRow) == GHOSTTY_SUCCESS) {
            (void)ghostty_row_get(rawRow, GHOSTTY_ROW_DATA_WRAP, &wrapped);
        }
        wraps[rowIndex] = wrapped;
        hyperlinks[rowIndex].clear();

        GhosttyRenderStateRowSelection selection = GHOSTTY_INIT_SIZED(GhosttyRenderStateRowSelection);
        if (ghostty_render_state_row_get(rowIterator, GHOSTTY_RENDER_STATE_ROW_DATA_SELECTION, &selection) ==
            GHOSTTY_SUCCESS) {
            row.selection = Sakura::Terminal::Range{selection.start_x, selection.end_x};
        }
        if (ghostty_render_state_row_get(rowIterator, GHOSTTY_RENDER_STATE_ROW_DATA_CELLS, &rowCells) !=
            GHOSTTY_SUCCESS) {
            return;
        }

        const bool cursorRow = cursor.visible && cursor.viewport_has_value && cursor.viewport_y == rowIndex;
        const U16 cursorColumn = cursorRow && cursor.wide_tail && cursor.viewport_x > 0
            ? static_cast<U16>(cursor.viewport_x - 1)
            : cursor.viewport_x;

        std::optional<Style> runStyle;
        const auto flush = [&]() {
            runStyle.reset();
        };

        U16 column = 0;
        while (ghostty_render_state_row_cells_next(rowCells)) {
            const U16 cellColumn = column++;
            GhosttyCell raw = 0;
            GhosttyCellWide wide = GHOSTTY_CELL_WIDE_NARROW;
            if (ghostty_render_state_row_cells_get(rowCells, GHOSTTY_RENDER_STATE_ROW_CELLS_DATA_RAW, &raw) ==
                GHOSTTY_SUCCESS) {
                (void)ghostty_cell_get(raw, GHOSTTY_CELL_DATA_WIDE, &wide);
            }
            if (wide == GHOSTTY_CELL_WIDE_SPACER_TAIL || wide == GHOSTTY_CELL_WIDE_SPACER_HEAD) {
                continue;
            }
            const U16 width = wide == GHOSTTY_CELL_WIDE_WIDE ? 2 : 1;

            GhosttyStyle style = GHOSTTY_INIT_SIZED(GhosttyStyle);
            bool styled = false;
            (void)ghostty_render_state_row_cells_get(rowCells, GHOSTTY_RENDER_STATE_ROW_CELLS_DATA_HAS_STYLING,
                                                     &styled);
            if (styled) {
                (void)ghostty_render_state_row_cells_get(rowCells, GHOSTTY_RENDER_STATE_ROW_CELLS_DATA_STYLE, &style);
            } else {
                ghostty_style_default(&style);
            }

            std::optional<ColorRGBA<F32>> foreground;
            std::optional<ColorRGBA<F32>> background;
            GhosttyColorRgb rgb{};
            if (ghostty_render_state_row_cells_get(rowCells, GHOSTTY_RENDER_STATE_ROW_CELLS_DATA_FG_COLOR, &rgb) ==
                GHOSTTY_SUCCESS) {
                foreground = FromGhostty(rgb);
            }
            if (ghostty_render_state_row_cells_get(rowCells, GHOSTTY_RENDER_STATE_ROW_CELLS_DATA_BG_COLOR, &rgb) ==
                GHOSTTY_SUCCESS) {
                background = FromGhostty(rgb);
            }
            std::optional<ColorRGBA<F32>> fill;
            const Style cellStyle = styleOf(style, foreground, background, fill);

            if (fill.has_value()) {
                if (!row.backgrounds.empty() && row.backgrounds.back().color == *fill &&
                    row.backgrounds.back().column + row.backgrounds.back().width == cellColumn) {
                    row.backgrounds.back().width += width;
                } else {
                    row.backgrounds.push_back({.column = cellColumn, .width = width, .color = *fill});
                }
            }

            uint32_t length = 0;
            (void)ghostty_render_state_row_cells_get(rowCells, GHOSTTY_RENDER_STATE_ROW_CELLS_DATA_GRAPHEMES_LEN,
                                                     &length);
            std::string text;
            if (length > 0 && !style.invisible) {
                graphemes.resize(length);
                (void)ghostty_render_state_row_cells_get(rowCells, GHOSTTY_RENDER_STATE_ROW_CELLS_DATA_GRAPHEMES_BUF,
                                                         graphemes.data());
                for (uint32_t index = 0; index < std::min(length, kMaxGraphemeCodepoints); ++index) {
                    text += Unicode::Encode(graphemes[index]);
                }
            }

            cells.emplace_back(cellColumn, text);
            bool linked = false;
            if (raw != 0 && ghostty_cell_get(raw, GHOSTTY_CELL_DATA_HAS_HYPERLINK, &linked) == GHOSTTY_SUCCESS && linked) {
                std::string uri = hyperlinkAt(cellColumn, rowIndex);
                if (!uri.empty()) {
                    auto& links = row.links;
                    if (!links.empty() && links.back().uri == uri && links.back().column + links.back().width == cellColumn) {
                        links.back().width += width;
                    } else {
                        links.push_back({.column = cellColumn, .width = width, .uri = std::move(uri)});
                    }
                }
            }

            if (cursorRow && cellColumn == cursorColumn) {
                screen.cursor.text = text == " " ? std::string() : text;
                screen.cursor.textColor = cellStyle.color;
                screen.cursor.width = width;
                flush();
                if (cellStyle.underline || cellStyle.strikethrough) {
                    row.spans.push_back({.column = cellColumn, .width = width, .text = " ",
                                         .color = cellStyle.color, .bold = cellStyle.bold,
                                         .underline = cellStyle.underline,
                                         .strikethrough = cellStyle.strikethrough});
                }
                continue;
            }

            const bool blank = text.empty() || text == " ";
            if (blank && !cellStyle.underline && !cellStyle.strikethrough) {
                flush();
                continue;
            }
            const bool ascii = text.size() <= 1 && Unicode::IsAscii(text);
            if (ascii && width == 1 && runStyle == cellStyle && !row.spans.empty() &&
                row.spans.back().column + row.spans.back().width == cellColumn) {
                auto& span = row.spans.back();
                span.text += blank ? std::string(" ") : text;
                span.width += 1;
                continue;
            }
            row.spans.push_back({
                .column = cellColumn,
                .width = width,
                .text = blank ? std::string(" ") : text,
                .color = cellStyle.color,
                .bold = cellStyle.bold,
                .underline = cellStyle.underline,
                .strikethrough = cellStyle.strikethrough,
            });
            if (ascii && width == 1) {
                runStyle = cellStyle;
            } else {
                flush();
            }
        }
        hyperlinks[rowIndex] = row.links;
    }

    void relink() {
        const U16 count = static_cast<U16>(screen.lines.size());
        for (U16 row = 0; row < count; ++row) {
            screen.lines[row].links = hyperlinks[row];
        }
        U16 start = 0;
        while (start < count) {
            U16 end = start;
            while (end + 1 < count && wraps[end]) {
                ++end;
            }
            std::vector<LinkCell> cells;
            for (U16 row = start; row <= end; ++row) {
                for (const auto& [column, text] : texts[row]) {
                    cells.push_back({.row = row, .column = column, .text = text});
                }
            }
            const bool clipped = end + 1 == count && wraps[end];
            for (const auto& found : DetectLinks(cells)) {
                if (clipped && found.open) {
                    continue;
                }
                U64 index = found.first;
                while (index <= found.last) {
                    U64 stop = index;
                    while (stop < found.last && cells[stop + 1].row == cells[index].row) {
                        ++stop;
                    }
                    const U16 first = cells[index].column;
                    const U16 last = cells[stop].column;
                    auto& links = screen.lines[cells[index].row].links;
                    const bool taken = std::any_of(links.begin(), links.end(), [&](const Link& link) {
                        return first < link.column + link.width && link.column <= last;
                    });
                    if (!taken) {
                        links.push_back({.column = first, .width = static_cast<U16>(last - first + 1), .uri = found.uri});
                    }
                    index = stop + 1;
                }
            }
            start = end + 1;
        }
    }

    void rebuild() {
        if (held()) {
            if (!captured) {
                return;
            }
        } else if (ghostty_render_state_update(renderState, terminal) != GHOSTTY_SUCCESS) {
            return;
        }
        captured = false;
        GhosttyRenderStateDirty dirty = GHOSTTY_RENDER_STATE_DIRTY_FALSE;
        (void)ghostty_render_state_get(renderState, GHOSTTY_RENDER_STATE_DATA_DIRTY, &dirty);

        Theme effective = palette;
        GhosttyColorRgb rgb{};
        if (ghostty_render_state_get(renderState, GHOSTTY_RENDER_STATE_DATA_COLOR_FOREGROUND, &rgb) == GHOSTTY_SUCCESS) {
            effective.foreground = FromGhostty(rgb);
        }
        if (ghostty_render_state_get(renderState, GHOSTTY_RENDER_STATE_DATA_COLOR_BACKGROUND, &rgb) == GHOSTTY_SUCCESS) {
            effective.background = FromGhostty(rgb);
        }
        if (effective != colors) {
            colors = effective;
            rebuildAll = true;
        }

        GhosttyTerminalScrollbar scrollbar{};
        (void)ghostty_terminal_get(terminal, GHOSTTY_TERMINAL_DATA_SCROLLBAR, &scrollbar);
        const bool tracking = mouseTracking();
        const bool scrolled = scrollbar.total != screen.scrollTotal || scrollbar.offset != screen.scrollOffset;

        GhosttyRenderStateCursor cursor = GHOSTTY_INIT_SIZED(GhosttyRenderStateCursor);
        (void)ghostty_render_state_get(renderState, GHOSTTY_RENDER_STATE_DATA_CURSOR, &cursor);
        const bool cursorVisible = cursor.visible && cursor.viewport_has_value;
        const U16 cursorColumn = cursor.wide_tail && cursor.viewport_x > 0 ? cursor.viewport_x - 1 : cursor.viewport_x;
        const auto cursorShape = CursorShapeOf(cursor.visual_style);
        const bool cursorMoved = cursorVisible != screen.cursor.visible || cursor.blinking != screen.cursor.blinking ||
                                 cursorShape != screen.cursor.shape ||
                                 (cursorVisible && (cursorColumn != screen.cursor.column ||
                                                    cursor.viewport_y != screen.cursor.row));
        if (dirty == GHOSTTY_RENDER_STATE_DIRTY_FALSE && !rebuildAll && !scrolled && !cursorMoved &&
            tracking == screen.mouseTracking) {
            return;
        }

        uint16_t columns = 0;
        uint16_t rows = 0;
        (void)ghostty_render_state_get(renderState, GHOSTTY_RENDER_STATE_DATA_COLS, &columns);
        (void)ghostty_render_state_get(renderState, GHOSTTY_RENDER_STATE_DATA_ROWS, &rows);
        const bool resized = columns != screen.columns || rows != screen.rows;
        const bool full = rebuildAll || resized || dirty == GHOSTTY_RENDER_STATE_DIRTY_FULL;

        const U16 previousCursorRow = screen.cursor.row;
        const bool previousCursorVisible = screen.cursor.visible;
        screen.columns = columns;
        screen.rows = rows;
        screen.lines.resize(rows);
        texts.resize(rows);
        hyperlinks.resize(rows);
        wraps.resize(rows);
        screen.scrollTotal = scrollbar.total;
        screen.scrollOffset = scrollbar.offset;
        screen.mouseTracking = tracking;
        screen.cursor = {};
        screen.cursor.visible = cursorVisible;
        screen.cursor.blinking = cursor.blinking;
        screen.cursor.column = cursorColumn;
        screen.cursor.row = cursor.viewport_y;
        screen.cursor.textColor = colors.foreground;
        screen.cursor.shape = cursorShape;

        if (ghostty_render_state_get(renderState, GHOSTTY_RENDER_STATE_DATA_ROW_ITERATOR, &rowIterator) ==
            GHOSTTY_SUCCESS) {
            U16 index = 0;
            bool built = false;
            while (ghostty_render_state_row_iterator_next(rowIterator) && index < rows) {
                bool rowDirty = full;
                if (!rowDirty) {
                    (void)ghostty_render_state_row_get(rowIterator, GHOSTTY_RENDER_STATE_ROW_DATA_DIRTY, &rowDirty);
                }
                const bool cursorTouched = index == screen.cursor.row ||
                                           (previousCursorVisible && index == previousCursorRow);
                if (rowDirty || cursorTouched) {
                    buildRow(screen.lines[index], index, cursor);
                    built = true;
                }
                ++index;
            }
            if (built) {
                relink();
            }
        }

        (void)ghostty_render_state_clean(renderState);
        rebuildAll = false;
        predominate();
        ++revision;
    }

    void predominate() {
        std::vector<std::pair<ColorRGBA<F32>, U64>> counts = {{colors.background, 0}};
        const auto tally = [&](const ColorRGBA<F32>& color, U64 width) {
            const auto entry = std::find_if(counts.begin(), counts.end(), [&](const auto& item) {
                return item.first == color;
            });
            if (entry == counts.end()) {
                counts.emplace_back(color, width);
            } else {
                entry->second += width;
            }
        };
        const auto at = [&](const Sakura::Terminal::Row& row, U16 column) {
            for (const auto& run : row.backgrounds) {
                if (column >= run.column && column < run.column + run.width) {
                    return run.color;
                }
            }
            return colors.background;
        };
        const U64 last = screen.lines.empty() ? 0 : screen.lines.size() - 1;
        for (U64 index = 0; index < screen.lines.size() && screen.columns > 0; ++index) {
            const auto& row = screen.lines[index];
            if (index == 0 || index == last) {
                U64 covered = 0;
                for (const auto& run : row.backgrounds) {
                    tally(run.color, run.width);
                    covered += run.width;
                }
                tally(colors.background, screen.columns - std::min<U64>(covered, screen.columns));
                continue;
            }
            tally(at(row, 0), 1);
            tally(at(row, screen.columns - 1), 1);
        }
        screen.base = palette.background;
        screen.background = colors.background;
        screen.fill = colors.background;
        U64 best = counts.front().second;
        for (const auto& [color, count] : counts) {
            if (count > best) {
                best = count;
                screen.fill = color;
            }
        }
    }
};

TerminalSession::TerminalSession() {
    impl = std::make_unique<Impl>();
}

TerminalSession::~TerminalSession() {
    stop();
}

bool TerminalSession::Supported() {
    return Platform::PseudoTerminal::Supported();
}

Result TerminalSession::start(const Config& config) {
    stop();
    impl->palette = config.theme;

    if (ghostty_terminal_new(nullptr, &impl->terminal, impl->grid.columns, impl->grid.rows) != GHOSTTY_SUCCESS ||
        ghostty_render_state_new(nullptr, &impl->renderState) != GHOSTTY_SUCCESS ||
        ghostty_render_state_row_iterator_new(nullptr, &impl->rowIterator) != GHOSTTY_SUCCESS ||
        ghostty_render_state_row_cells_new(nullptr, &impl->rowCells) != GHOSTTY_SUCCESS ||
        ghostty_key_encoder_new(nullptr, &impl->keyEncoder) != GHOSTTY_SUCCESS ||
        ghostty_key_event_new(nullptr, &impl->keyEvent) != GHOSTTY_SUCCESS ||
        ghostty_mouse_encoder_new(nullptr, &impl->mouseEncoder) != GHOSTTY_SUCCESS ||
        ghostty_mouse_event_new(nullptr, &impl->mouseEvent) != GHOSTTY_SUCCESS) {
        JST_ERROR("[TERMINAL] Failed to create the terminal emulator.");
        impl->release();
        return Result::ERROR;
    }

    (void)ghostty_terminal_set(impl->terminal, GHOSTTY_TERMINAL_OPT_USERDATA, impl.get());
    impl->setCallback(GHOSTTY_TERMINAL_OPT_WRITE_PTY, &Impl::WritePty);
    impl->setCallback(GHOSTTY_TERMINAL_OPT_TITLE_CHANGED, &Impl::TitleChanged);
    impl->setCallback(GHOSTTY_TERMINAL_OPT_SIZE, &Impl::Size);
    impl->setCallback(GHOSTTY_TERMINAL_OPT_DEVICE_ATTRIBUTES, &Impl::DeviceAttributes);
    impl->setCallback(GHOSTTY_TERMINAL_OPT_COLOR_SCHEME, &Impl::ColorScheme);
    impl->setCallback(GHOSTTY_TERMINAL_OPT_XTVERSION, &Impl::Version);
    impl->setCallback(GHOSTTY_TERMINAL_OPT_RENDER_HOLD, &Impl::RenderHold);
    const size_t scrollback = kScrollbackLines;
    (void)ghostty_terminal_set(impl->terminal, GHOSTTY_TERMINAL_OPT_SCROLLBACK_MAX_LINES, &scrollback);
    impl->applyTheme();

    std::string shell = config.program;
    std::vector<std::string> arguments = config.arguments;
    if (shell.empty()) {
        if (Platform::EnvironmentVariable("SHELL", shell) != Result::SUCCESS || shell.empty()) {
            shell = "/bin/sh";
        }
#if defined(JST_OS_MAC)
        arguments = {"-l"};
#endif
    }
    std::string directory = config.directory;
    if (directory.empty()) {
        (void)Platform::EnvironmentVariable("HOME", directory);
    }

    Platform::PseudoTerminal::Config pty{
        .program = shell,
        .arguments = arguments,
        .directory = directory,
        .environment = {
            {"TERM", "xterm-256color"},
            {"COLORTERM", "truecolor"},
            {"TERM_PROGRAM", kVersion},
        },
        .columns = impl->grid.columns,
        .rows = impl->grid.rows,
    };
    std::string locale;
    if (Platform::EnvironmentVariable("LANG", locale) != Result::SUCCESS || locale.empty()) {
        pty.environment.push_back({"LANG", "en_US.UTF-8"});
    }

    if (impl->pty.start(pty) != Result::SUCCESS) {
        impl->release();
        return Result::ERROR;
    }

    impl->io = std::thread([self = impl.get()]() {
        self->transfer();
    });
    impl->started = true;
    impl->rebuildAll = true;
    impl->rebuild();
    return Result::SUCCESS;
}

void TerminalSession::stop() {
    {
        std::lock_guard lock(impl->mutex);
        impl->stopping = true;
    }
    impl->pty.interrupt();
    if (impl->io.joinable()) {
        impl->io.join();
    }
    impl->pty.stop();
    impl->release();
    impl->started = false;
    impl->anchor.reset();
    impl->selected = false;
    impl->pressedButton.reset();
    impl->holdSince.reset();
    impl->captured = false;
    impl->screen.selection.clear();
    impl->outbox.clear();
    impl->inbox.clear();
    impl->received.clear();
    impl->stopping = false;
}

void TerminalSession::poll() {
    if (!impl->started) {
        return;
    }
    {
        std::lock_guard lock(impl->mutex);
        impl->received.swap(impl->inbox);
    }
    if (impl->received.size() >= kReadBudget) {
        impl->pty.interrupt();
    }
    if (!impl->received.empty()) {
        ghostty_terminal_vt_write(impl->terminal, reinterpret_cast<const uint8_t*>(impl->received.data()),
                                  impl->received.size());
        impl->received.clear();
    }
    impl->rebuild();
}

bool TerminalSession::running() const {
    return impl->started && impl->pty.running();
}

const std::string& TerminalSession::title() const {
    return impl->title;
}

const Sakura::Terminal::Screen& TerminalSession::screen() const {
    return impl->screen;
}

U64 TerminalSession::revision() const {
    return impl->revision;
}

void TerminalSession::theme(const Theme& theme) {
    if (theme == impl->palette) {
        return;
    }
    impl->palette = theme;
    if (impl->started) {
        impl->applyTheme();
        impl->rebuild();
    }
}

void TerminalSession::resize(const Sakura::Terminal::Grid& grid) {
    impl->grid = grid;
    if (!impl->started || grid.columns == 0 || grid.rows == 0) {
        return;
    }
    const auto cellWidth = static_cast<uint32_t>(std::max(1.0f, grid.cellWidth));
    const auto cellHeight = static_cast<uint32_t>(std::max(1.0f, grid.cellHeight));
    (void)ghostty_terminal_resize(impl->terminal, grid.columns, grid.rows, cellWidth, cellHeight);
    (void)impl->pty.resize(grid.columns, grid.rows,
                           static_cast<U16>(std::min<U32>(grid.columns * cellWidth, 0xFFFF)),
                           static_cast<U16>(std::min<U32>(grid.rows * cellHeight, 0xFFFF)));
    impl->rebuildAll = true;
    impl->rebuild();
}

void TerminalSession::key(const KeyEvent& event) {
    if (!impl->started || event.type != KeyEventType::Press) {
        return;
    }
    const auto mapping = MapKey(event.key);
    if (mapping.key == GHOSTTY_KEY_UNIDENTIFIED) {
        return;
    }
    const auto& modifiers = event.modifiers;
    if (modifiers.shift && !modifiers.control && !modifiers.alt && !impl->alternateScreen()) {
        GhosttyTerminalScrollViewport behavior{};
        if (event.key == KeyCode::PageUp || event.key == KeyCode::PageDown) {
            const auto page = static_cast<intptr_t>(std::max<U16>(1, impl->grid.rows));
            behavior.tag = GHOSTTY_SCROLL_VIEWPORT_DELTA;
            behavior.value.delta = event.key == KeyCode::PageUp ? -page : page;
            ghostty_terminal_scroll_viewport(impl->terminal, behavior);
            impl->rebuild();
            return;
        }
        if (event.key == KeyCode::Home || event.key == KeyCode::End) {
            behavior.tag = event.key == KeyCode::Home ? GHOSTTY_SCROLL_VIEWPORT_TOP : GHOSTTY_SCROLL_VIEWPORT_BOTTOM;
            ghostty_terminal_scroll_viewport(impl->terminal, behavior);
            impl->rebuild();
            return;
        }
    }
    const bool printable = mapping.codepoint != 0 && event.key != KeyCode::KeypadEnter;
    const bool meta = AltSendsEscape() && modifiers.alt && !modifiers.control;
    if (printable && !modifiers.control && !meta) {
        return;
    }
    impl->clearSelection();
    impl->followOutput();
    impl->encodeKey(mapping.key, MapModifiers(modifiers), mapping.codepoint, event.repeat,
                    (meta || modifiers.control) && printable ? Shifted(mapping.codepoint, modifiers.shift) : 0);
    impl->rebuild();
}

void TerminalSession::text(const std::string& text) {
    if (!impl->started || text.empty()) {
        return;
    }
    impl->clearSelection();
    impl->followOutput();
    impl->send(text, true);
    impl->rebuild();
}

void TerminalSession::paste(const std::string& text) {
    if (!impl->started || text.empty()) {
        return;
    }
    std::string data = text;
    const bool bracketed = impl->mode(GHOSTTY_MODE_BRACKETED_PASTE);
    size_t length = 0;
    (void)ghostty_paste_encode(data.data(), data.size(), bracketed, nullptr, 0, &length);
    std::string encoded(length, '\0');
    if (ghostty_paste_encode(data.data(), data.size(), bracketed, encoded.data(), encoded.size(), &length) !=
        GHOSTTY_SUCCESS) {
        return;
    }
    encoded.resize(length);
    impl->clearSelection();
    impl->followOutput();
    impl->send(std::move(encoded), true);
    impl->rebuild();
}

void TerminalSession::scroll(I32 rows, Extent2D<F32> position, KeyModifiers modifiers) {
    if (!impl->started || rows == 0) {
        return;
    }
    const U32 steps = static_cast<U32>(std::abs(rows));
    if (impl->mouseTracking()) {
        ghostty_mouse_encoder_setopt_from_terminal(impl->mouseEncoder, impl->terminal);
        GhosttyMouseEncoderSize size = GHOSTTY_INIT_SIZED(GhosttyMouseEncoderSize);
        size.screen_width = static_cast<uint32_t>(impl->grid.columns * impl->grid.cellWidth);
        size.screen_height = static_cast<uint32_t>(impl->grid.rows * impl->grid.cellHeight);
        size.cell_width = static_cast<uint32_t>(std::max(1.0f, impl->grid.cellWidth));
        size.cell_height = static_cast<uint32_t>(std::max(1.0f, impl->grid.cellHeight));
        ghostty_mouse_encoder_setopt(impl->mouseEncoder, GHOSTTY_MOUSE_ENCODER_OPT_SIZE, &size);
        ghostty_mouse_event_set_action(impl->mouseEvent, GHOSTTY_MOUSE_ACTION_PRESS);
        ghostty_mouse_event_set_button(impl->mouseEvent, rows < 0 ? GHOSTTY_MOUSE_BUTTON_FOUR : GHOSTTY_MOUSE_BUTTON_FIVE);
        ghostty_mouse_event_set_mods(impl->mouseEvent, MapModifiers(modifiers));
        ghostty_mouse_event_set_position(impl->mouseEvent, {position.x, position.y});
        for (U32 step = 0; step < steps; ++step) {
            char buffer[64];
            size_t length = 0;
            if (ghostty_mouse_encoder_encode(impl->mouseEncoder, impl->mouseEvent, buffer, sizeof(buffer), &length) ==
                    GHOSTTY_SUCCESS &&
                length > 0) {
                impl->send({buffer, length}, false);
            }
        }
        return;
    }
    if (impl->alternateScreen()) {
        if (impl->mode(GHOSTTY_MODE_ALT_SCROLL)) {
            const GhosttyKey key = rows < 0 ? GHOSTTY_KEY_ARROW_UP : GHOSTTY_KEY_ARROW_DOWN;
            for (U32 step = 0; step < steps; ++step) {
                impl->encodeKey(key, 0, 0, false);
            }
        }
        return;
    }
    GhosttyTerminalScrollViewport behavior{};
    behavior.tag = GHOSTTY_SCROLL_VIEWPORT_DELTA;
    behavior.value.delta = rows;
    ghostty_terminal_scroll_viewport(impl->terminal, behavior);
    impl->rebuild();
}

void TerminalSession::scrollTo(U64 row) {
    if (!impl->started) {
        return;
    }
    GhosttyTerminalScrollViewport behavior{};
    behavior.tag = GHOSTTY_SCROLL_VIEWPORT_ROW;
    behavior.value.row = static_cast<size_t>(row);
    ghostty_terminal_scroll_viewport(impl->terminal, behavior);
    impl->rebuild();
}

void TerminalSession::select(const Sakura::Terminal::Selection& selection) {
    if (!impl->started) {
        return;
    }
    const auto point = impl->screenPoint(selection.column, selection.row);
    if (!point.has_value()) {
        return;
    }
    using Gesture = Sakura::Terminal::Gesture;
    if (selection.gesture == Gesture::Press) {
        impl->clearSelection();
        const U8 clicks = std::min<U8>(3, std::max<U8>(1, selection.clicks));
        const auto range = impl->expand(*point, clicks);
        if (!range.has_value()) {
            return;
        }
        impl->anchor = Impl::Anchor{.start = range->first, .end = range->second, .clicks = clicks};
        if (clicks > 1) {
            impl->applySelection(range->first, range->second);
        }
        impl->rebuild();
        return;
    }
    if (selection.gesture != Gesture::Drag || !impl->anchor.has_value()) {
        return;
    }
    const auto& anchor = *impl->anchor;
    const auto head = impl->expand(*point, anchor.clicks);
    if (!head.has_value()) {
        return;
    }
    if (PointBefore(head->first, anchor.start)) {
        impl->applySelection(head->first, anchor.end);
    } else {
        impl->applySelection(anchor.start, head->second);
    }
    impl->rebuild();
}

void TerminalSession::pointer(const MouseEvent& event) {
    if (!impl->started || !impl->mouseTracking()) {
        return;
    }
    ghostty_mouse_encoder_setopt_from_terminal(impl->mouseEncoder, impl->terminal);
    GhosttyMouseEncoderSize size = GHOSTTY_INIT_SIZED(GhosttyMouseEncoderSize);
    size.screen_width = static_cast<uint32_t>(impl->grid.columns * impl->grid.cellWidth);
    size.screen_height = static_cast<uint32_t>(impl->grid.rows * impl->grid.cellHeight);
    size.cell_width = static_cast<uint32_t>(std::max(1.0f, impl->grid.cellWidth));
    size.cell_height = static_cast<uint32_t>(std::max(1.0f, impl->grid.cellHeight));
    ghostty_mouse_encoder_setopt(impl->mouseEncoder, GHOSTTY_MOUSE_ENCODER_OPT_SIZE, &size);

    std::optional<MouseButton> button = event.button;
    switch (event.type) {
        case MouseEventType::Click:
            ghostty_mouse_event_set_action(impl->mouseEvent, GHOSTTY_MOUSE_ACTION_PRESS);
            impl->pressedButton = event.button;
            break;
        case MouseEventType::Release:
            ghostty_mouse_event_set_action(impl->mouseEvent, GHOSTTY_MOUSE_ACTION_RELEASE);
            impl->pressedButton.reset();
            break;
        case MouseEventType::Move:
            ghostty_mouse_event_set_action(impl->mouseEvent, GHOSTTY_MOUSE_ACTION_MOTION);
            button = impl->pressedButton;
            break;
        case MouseEventType::Cancel:
            if (!impl->pressedButton.has_value()) {
                return;
            }
            ghostty_mouse_event_set_action(impl->mouseEvent, GHOSTTY_MOUSE_ACTION_RELEASE);
            button = impl->pressedButton;
            impl->pressedButton.reset();
            break;
        default:
            return;
    }
    if (!button.has_value()) {
        ghostty_mouse_event_clear_button(impl->mouseEvent);
    } else {
        switch (*button) {
            case MouseButton::Left:
                ghostty_mouse_event_set_button(impl->mouseEvent, GHOSTTY_MOUSE_BUTTON_LEFT);
                break;
            case MouseButton::Right:
                ghostty_mouse_event_set_button(impl->mouseEvent, GHOSTTY_MOUSE_BUTTON_RIGHT);
                break;
            case MouseButton::Middle:
                ghostty_mouse_event_set_button(impl->mouseEvent, GHOSTTY_MOUSE_BUTTON_MIDDLE);
                break;
        }
    }
    const bool pressed = impl->pressedButton.has_value();
    (void)ghostty_mouse_encoder_setopt(impl->mouseEncoder, GHOSTTY_MOUSE_ENCODER_OPT_ANY_BUTTON_PRESSED, &pressed);
    if (event.type != MouseEventType::Cancel) {
        impl->pointerPosition = event.position;
    }
    ghostty_mouse_event_set_mods(impl->mouseEvent, MapModifiers(event.modifiers));
    ghostty_mouse_event_set_position(impl->mouseEvent, {impl->pointerPosition.x, impl->pointerPosition.y});
    char buffer[64];
    size_t length = 0;
    if (ghostty_mouse_encoder_encode(impl->mouseEncoder, impl->mouseEvent, buffer, sizeof(buffer), &length) ==
            GHOSTTY_SUCCESS &&
        length > 0) {
        impl->send({buffer, length}, false);
    }
}

void TerminalSession::focus(bool focused) {
    if (!impl->started || !impl->mode(GHOSTTY_MODE_FOCUS_EVENT)) {
        return;
    }
    char buffer[16];
    size_t length = 0;
    if (ghostty_focus_encode(focused ? GHOSTTY_FOCUS_GAINED : GHOSTTY_FOCUS_LOST, buffer, sizeof(buffer), &length) ==
            GHOSTTY_SUCCESS &&
        length > 0) {
        impl->send({buffer, length}, false);
    }
}

#else

struct TerminalSession::Impl {
    Sakura::Terminal::Screen screen;
    std::string title;
};

TerminalSession::TerminalSession() {
    impl = std::make_unique<Impl>();
}

TerminalSession::~TerminalSession() = default;

bool TerminalSession::Supported() {
    return false;
}

Result TerminalSession::start(const Config&) {
    JST_ERROR("[TERMINAL] The integrated terminal is not available in this build.");
    return Result::ERROR;
}

void TerminalSession::stop() {}

void TerminalSession::poll() {}

bool TerminalSession::running() const {
    return false;
}

const std::string& TerminalSession::title() const {
    return impl->title;
}

const Sakura::Terminal::Screen& TerminalSession::screen() const {
    return impl->screen;
}

U64 TerminalSession::revision() const {
    return 0;
}

void TerminalSession::theme(const Theme&) {}

void TerminalSession::resize(const Sakura::Terminal::Grid&) {}

void TerminalSession::key(const KeyEvent&) {}

void TerminalSession::text(const std::string&) {}

void TerminalSession::paste(const std::string&) {}

void TerminalSession::scroll(I32, Extent2D<F32>, KeyModifiers) {}

void TerminalSession::scrollTo(U64) {}

void TerminalSession::select(const Sakura::Terminal::Selection&) {}

void TerminalSession::pointer(const MouseEvent&) {}

void TerminalSession::focus(bool) {}

#endif

}  // namespace Jetstream
