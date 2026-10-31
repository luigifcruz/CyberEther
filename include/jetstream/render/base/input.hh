#ifndef JETSTREAM_RENDER_BASE_INPUT_HH
#define JETSTREAM_RENDER_BASE_INPUT_HH

#include <array>
#include <bitset>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "jetstream/surface.hh"
#include "jetstream/types.hh"

namespace Jetstream::Render {

class JETSTREAM_API InputQueue {
 public:
    struct Pointer {
        std::optional<Extent2D<F32>> position;
        bool remote = false;
    };

    struct Button {
        MouseButton button = MouseButton::Left;
        bool down = false;
        bool remote = false;
    };

    struct Wheel {
        Extent2D<F32> delta = {0.0f, 0.0f};
        bool remote = false;
        ScrollPhase phase = ScrollPhase::None;
        bool precise = false;
    };

    struct Key {
        KeyCode key = KeyCode::Unknown;
        bool down = false;
        bool remote = false;
    };

    struct Modifiers {
        KeyModifiers modifiers;
        bool remote = false;
    };

    struct Text {
        char32_t codepoint = 0;
        bool remote = false;
    };

    struct Focus {
        bool focused = true;
    };

    struct Hover {
        bool entered = true;
    };

    using Event = std::variant<Pointer, Button, Wheel, Key, Modifiers, Text, Focus, Hover>;

    void push(Event event);
    void pushText(const std::string& utf8);
    std::vector<Event> drain();
    void wait(F64 seconds);
    void wake();

    static std::vector<char32_t> DecodeUtf8(const std::string& utf8);
    static std::optional<MouseButton> ButtonFromIndex(I64 index);

 private:
    std::mutex mutex;
    std::condition_variable signal;
    std::vector<Event> events;
    bool woken = false;
};

class JETSTREAM_API InputState {
 public:
    struct Config {
        bool macOS = false;
    };

    using Event = std::variant<MouseEvent, KeyEvent, TextEvent, FocusEvent>;

    InputState();
    explicit InputState(const Config& config);

    void configure(const Config& config);
    void advance(F64 now, const std::vector<InputQueue::Event>& queued);

    U64 frame() const { return frameCount; }
    U64 generation() const { return sourceGeneration; }
    F64 time() const { return elapsed; }
    F32 deltaTime() const { return delta; }
    bool windowFocused() const { return focused && !focusLost; }
    const KeyModifiers& modifiers() const { return currentModifiers; }
    const std::optional<Extent2D<F32>>& pointer() const { return pointerPosition; }
    const std::array<bool, 3>& buttons() const { return buttonDown; }
    bool keyDown(KeyCode key) const;
    bool hasPending() const { return !pendingEvents.empty(); }
    const std::vector<Event>& events() const { return frameEvents; }

 private:
    static constexpr U64 kKeyCount = static_cast<U64>(KeyCode::Oem102) + 1;

    Config config;

    U64 sourceGeneration = 0;
    U64 frameCount = 0;
    std::optional<F64> lastClock;
    F64 elapsed = 0.0;
    F32 delta = 0.0f;

    bool hostFocused = true;
    bool focused = true;
    bool focusLost = false;
    KeyModifiers currentModifiers;
    std::optional<Extent2D<F32>> pointerPosition;
    std::optional<Extent2D<F32>> lastValidPointer = Extent2D<F32>{0.0f, 0.0f};
    bool pointerInside = false;
    std::array<bool, 3> buttonDown{};
    std::array<F64, 3> clickTime{};
    std::array<std::optional<Extent2D<F32>>, 3> clickPosition{};
    std::array<U8, 3> clickCount{};
    bool ctrlLeftAsRight = false;
    std::array<bool, kKeyCount> keys{};
    std::array<F32, kKeyCount> keyDuration{};
    std::vector<Event> frameEvents;
    std::deque<InputQueue::Event> pendingEvents;

    struct Trickle {
        bool moved = false;
        bool wheeled = false;
        bool keyChanged = false;
        bool nonCharKeyChanged = false;
        bool textInputted = false;
        std::bitset<3> buttons;
        std::bitset<4> modifiers;
        std::bitset<kKeyCount> keys;
    };

    bool process(const InputQueue::Event& event, Trickle& trickle);
    bool movePointer(const std::optional<Extent2D<F32>>& next, Trickle& trickle);
    bool button(MouseButton button, bool down, Trickle& trickle);
    bool potentiallyChar(KeyCode key) const;
    void clear();
    void repeatKeys();
};

}  // namespace Jetstream::Render

#endif  // JETSTREAM_RENDER_BASE_INPUT_HH
