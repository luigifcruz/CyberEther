#include "jetstream/render/base/input.hh"

#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <utility>

namespace Jetstream::Render {

namespace {

constexpr char32_t kReplacementCharacter = 0xFFFD;
constexpr F32 kDoubleClickTime = 0.30f;
constexpr F32 kDoubleClickDistance = 6.0f;
constexpr F32 kKeyRepeatDelay = 0.275f;
constexpr F32 kKeyRepeatRate = 0.050f;

bool ValidScalar(char32_t codepoint) {
    return codepoint <= 0x10FFFF && !(codepoint >= 0xD800 && codepoint <= 0xDFFF);
}

std::optional<Extent2D<F32>> PixelPosition(const std::optional<Extent2D<F32>>& position) {
    if (!position.has_value() || !std::isfinite(position->x) || !std::isfinite(position->y)) {
        return std::nullopt;
    }
    return Extent2D<F32>{std::floor(position->x), std::floor(position->y)};
}

I32 TypematicRepeatAmount(F32 t0, F32 t1, F32 delay, F32 rate) {
    const I32 before = (t0 < delay) ? -1 : static_cast<I32>((t0 - delay) / rate);
    const I32 after = (t1 < delay) ? -1 : static_cast<I32>((t1 - delay) / rate);
    return after - before;
}

U64 ButtonIndex(MouseButton button) {
    return static_cast<U64>(button);
}

Platform::DeadlineWaiter::Clock::time_point WaitDeadline(F64 seconds) {
    using Clock = Platform::DeadlineWaiter::Clock;

    const auto now = Clock::now();
    if (!(seconds > 0.0)) {
        return now;
    }

    const auto available = std::chrono::duration<F64>(Clock::time_point::max() - now).count();
    if (seconds >= available) {
        return Clock::time_point::max();
    }

    return now + std::chrono::ceil<Clock::duration>(std::chrono::duration<F64>(seconds));
}

}  // namespace

void InputQueue::push(Event event) {
    if (auto* wheel = std::get_if<Wheel>(&event); wheel && wheel->synthesizeMomentum && !wheel->receivedAt) {
        wheel->receivedAt = std::chrono::steady_clock::now();
    }
    {
        std::lock_guard lock(mutex);
        events.push_back(std::move(event));
    }
    signal.notify();
}

void InputQueue::pushText(const std::string& utf8) {
    const auto codepoints = DecodeUtf8(utf8);
    {
        std::lock_guard lock(mutex);
        for (const char32_t codepoint : codepoints) {
            events.push_back(Text{codepoint});
        }
    }
    signal.notify();
}

std::vector<InputQueue::Event> InputQueue::drain() {
    std::lock_guard lock(mutex);
    return std::exchange(events, {});
}

bool InputQueue::wait(F64 seconds, bool surfaces) {
    const auto deadline = WaitDeadline(seconds);
    const auto ready = [&] {
        return woken || !events.empty() || (surfaces && surfaceWoken);
    };

    std::unique_lock lock(mutex);
    while (!ready() && Platform::DeadlineWaiter::Clock::now() < deadline) {
        const auto generation = signal.generation();
        lock.unlock();
        signal.waitUntil(deadline, generation);
        lock.lock();
    }

    const bool surfaceWake = surfaces && surfaceWoken;
    woken = false;
    surfaceWoken = false;

    return surfaceWake;
}

void InputQueue::wake() {
    {
        std::lock_guard lock(mutex);
        woken = true;
    }
    signal.notify();
}

void InputQueue::wakeSurface() {
    {
        std::lock_guard lock(mutex);
        if (std::exchange(surfaceWoken, true)) {
            return;
        }
    }
    signal.notify();
}

std::vector<char32_t> InputQueue::DecodeUtf8(const std::string& utf8) {
    std::vector<char32_t> codepoints;
    const auto* bytes = reinterpret_cast<const unsigned char*>(utf8.data());
    const U64 size = utf8.size();

    U64 index = 0;
    while (index < size) {
        const unsigned char lead = bytes[index];
        if (lead == 0) {
            ++index;
            continue;
        }

        U64 length = 0;
        char32_t codepoint = 0;
        char32_t minimum = 0;
        if (lead < 0x80) {
            codepoints.push_back(lead);
            ++index;
            continue;
        } else if ((lead & 0xE0) == 0xC0) {
            length = 2;
            codepoint = lead & 0x1F;
            minimum = 0x80;
        } else if ((lead & 0xF0) == 0xE0) {
            length = 3;
            codepoint = lead & 0x0F;
            minimum = 0x800;
        } else if ((lead & 0xF8) == 0xF0) {
            length = 4;
            codepoint = lead & 0x07;
            minimum = 0x10000;
        } else {
            codepoints.push_back(kReplacementCharacter);
            ++index;
            continue;
        }

        U64 consumed = 1;
        bool valid = true;
        while (consumed < length) {
            if (index + consumed >= size || (bytes[index + consumed] & 0xC0) != 0x80) {
                valid = false;
                break;
            }
            codepoint = (codepoint << 6) | (bytes[index + consumed] & 0x3F);
            ++consumed;
        }

        if (valid && (codepoint < minimum || !ValidScalar(codepoint))) {
            valid = false;
        }

        codepoints.push_back(valid ? codepoint : kReplacementCharacter);
        index += consumed;
    }

    return codepoints;
}

std::optional<MouseButton> InputQueue::ButtonFromIndex(I64 index) {
    if (index < 0 || index > static_cast<I64>(MouseButton::Middle)) {
        return std::nullopt;
    }
    return static_cast<MouseButton>(index);
}

InputState::InputState() : InputState(Config{}) {}

InputState::InputState(const Config& config) : config(config) {
    static std::atomic<U64> generations{0};
    sourceGeneration = ++generations;
    keyDuration.fill(-1.0f);
    clickTime.fill(-std::numeric_limits<F64>::infinity());
}

void InputState::configure(const Config& next) {
    config = next;
}

bool InputState::keyDown(KeyCode key) const {
    const U64 index = static_cast<U64>(key);
    return index > 0 && index < kKeyCount && keys[index];
}

F64 InputState::waitTimeout(F64 maximum) const {
    if (!pendingEvents.empty()) {
        return 0.0;
    }
    return momentum.active() ? std::min(maximum, 1.0 / 240.0) : maximum;
}

void InputState::advance(F64 now, const std::vector<InputQueue::Event>& queued) {
    if (lastClock.has_value()) {
        if (now <= *lastClock) {
            now = *lastClock + 1e-5;
        }
        delta = static_cast<F32>(now - *lastClock);
    } else {
        delta = 1.0f / 60.0f;
    }
    lastClock = now;

    ++frameCount;
    elapsed += delta;

    frameEvents.clear();
    focusLost = false;

    const auto receivedNow = std::chrono::steady_clock::now();
    for (const auto& event : queued) {
        F64 time = elapsed;
        if (const auto* wheel = std::get_if<InputQueue::Wheel>(&event);
            wheel && wheel->synthesizeMomentum && !wheel->remote) {
            if (wheel->receivedAt) {
                time -= std::max(0.0, std::chrono::duration<F64>(receivedNow - *wheel->receivedAt).count());
            }
            const U32 span = wheel->timestamp - scrollTimestamp.value_or(wheel->timestamp);
            if (scrollTimestamp && span < 0x80000000u) {
                time = std::min(time, scrollTime + span / 1000.0);
            }
            scrollTimestamp = wheel->timestamp;
            scrollTime = time;
        }
        pendingEvents.push_back({event, time});
    }
    Trickle trickle;
    while (!pendingEvents.empty() && process(pendingEvents.front().event, pendingEvents.front().time, trickle)) {
        pendingEvents.pop_front();
    }
    advanceMomentum();

    for (U64 index = 1; index < kKeyCount; ++index) {
        F32& duration = keyDuration[index];
        duration = keys[index] ? (duration < 0.0f ? 0.0f : duration + delta) : -1.0f;
    }

    repeatKeys();
}

bool InputState::process(const InputQueue::Event& event, F64 time, Trickle& trickle) {
    if (const auto* focus = std::get_if<InputQueue::Focus>(&event)) {
        hostFocused = focus->focused;
        focusLost = !focus->focused;
        if (focusLost) {
            clear();
        }
        if (focused != focus->focused) {
            focused = focus->focused;
            if (focused && pointerInside) {
                pointerPosition = lastValidPointer;
            }
            frameEvents.push_back(FocusEvent{focused});
        }
        return true;
    }

    const bool remote = std::visit([](const auto& typed) {
        if constexpr (requires { typed.remote; }) {
            return typed.remote;
        } else {
            return false;
        }
    }, event);
    const bool passive = std::holds_alternative<InputQueue::Pointer>(event) ||
                         std::holds_alternative<InputQueue::Hover>(event) ||
                         std::holds_alternative<InputQueue::Wheel>(event) ||
                         std::holds_alternative<InputQueue::Modifiers>(event);
    if (!hostFocused && !remote && !passive) {
        return true;
    }
    if (!focused && (hostFocused || remote)) {
        focused = true;
        focusLost = false;
        if (pointerInside) {
            pointerPosition = lastValidPointer;
        }
        frameEvents.push_back(FocusEvent{true});
    }

    const Extent2D<F32> point = pointerPosition.value_or(Extent2D<F32>{0.0f, 0.0f});

    if (const auto* pointer = std::get_if<InputQueue::Pointer>(&event)) {
        const auto position = PixelPosition(pointer->position);
        if (!movePointer(position, trickle)) {
            return false;
        }
        pointerInside = position.has_value();
        if (position.has_value()) {
            lastValidPointer = position;
        }
        return true;
    }

    if (const auto* hover = std::get_if<InputQueue::Hover>(&event)) {
        if (hover->entered) {
            if (!movePointer(lastValidPointer, trickle)) {
                return false;
            }
            pointerInside = true;
        } else {
            if (!movePointer(std::nullopt, trickle)) {
                return false;
            }
            pointerInside = false;
        }
        return true;
    }

    if (const auto* pressed = std::get_if<InputQueue::Button>(&event)) {
        return button(pressed->button, pressed->down, trickle);
    }

    if (const auto* wheel = std::get_if<InputQueue::Wheel>(&event)) {
        if (wheel->delta.x == 0.0f && wheel->delta.y == 0.0f && wheel->phase == ScrollPhase::None &&
            !wheel->precise) {
            return true;
        }
        if (trickle.buttons.any()) {
            return false;
        }
        const bool ends = wheel->phase == ScrollPhase::Ended || wheel->phase == ScrollPhase::MomentumEnded;
        if (ends && wheel->explicitRelease && !wheel->synthesizeMomentum && !wheel->remote) {
            if (momentum.active() || samplingMomentum || gestureHeld) {
                trickle.wheeled = true;
                cancelMomentum();
            }
            return true;
        }
        if (!wheel->synthesizeMomentum || wheel->remote || wheel->phase == ScrollPhase::Began) {
            cancelMomentum();
        }
        trickle.wheeled = true;
        MouseEvent scroll{};
        scroll.type = MouseEventType::Scroll;
        scroll.position = wheel->position.value_or(point);
        scroll.scroll = wheel->delta;
        scroll.modifiers = currentModifiers;
        scroll.positioned = wheel->position.has_value() || pointerPosition.has_value();
        scroll.phase = wheel->phase;
        scroll.precise = wheel->precise;
        scroll.smooth = wheel->smooth;
        frameEvents.push_back(scroll);
        if (wheel->synthesizeMomentum && !wheel->remote) {
            if (wheel->phase == ScrollPhase::Began) {
                samplingMomentum = true;
            }
            if (wheel->phase == ScrollPhase::Began || wheel->phase == ScrollPhase::Changed) {
                gestureHeld = true;
                momentumEvent = scroll;
                if (samplingMomentum) {
                    momentum.sample(wheel->timestamp, wheel->delta);
                }
            } else if (wheel->phase == ScrollPhase::Ended) {
                gestureHeld = false;
            }
            if (samplingMomentum && wheel->phase == ScrollPhase::Ended) {
                samplingMomentum = false;
                if (momentum.release(wheel->timestamp)) {
                    momentum.advance(std::max(0.0, elapsed - time));
                }
                if (momentum.active()) {
                    momentumEvent = scroll;
                    momentumEvent.phase = ScrollPhase::Momentum;
                    momentumEvent.scroll = {0.0f, 0.0f};
                    momentumSteppedAt = elapsed;
                    frameEvents.push_back(momentumEvent);
                }
            }
        }
        return true;
    }

    if (const auto* key = std::get_if<InputQueue::Key>(&event)) {
        const U64 index = static_cast<U64>(key->key);
        if (index == 0 || index >= kKeyCount || keys[index] == key->down) {
            return true;
        }
        const bool character = potentiallyChar(key->key);
        if (trickle.keys.test(index) || trickle.buttons.any() || (trickle.textInputted && !character)) {
            return false;
        }
        trickle.keys.set(index);
        trickle.keyChanged = true;
        trickle.nonCharKeyChanged = trickle.nonCharKeyChanged || !character;
        keys[index] = key->down;
        if (key->down) {
            cancelMomentum();
        }
        keyDuration[index] = -1.0f;
        frameEvents.push_back(KeyEvent{key->down ? KeyEventType::Press : KeyEventType::Release,
                                       key->key, currentModifiers});
        return true;
    }

    if (const auto* modifiers = std::get_if<InputQueue::Modifiers>(&event)) {
        const auto& next = modifiers->modifiers;
        std::bitset<4> changed;
        changed[0] = next.control != currentModifiers.control;
        changed[1] = next.shift != currentModifiers.shift;
        changed[2] = next.alt != currentModifiers.alt;
        changed[3] = next.super != currentModifiers.super;
        if (changed.none()) {
            return true;
        }
        if ((changed & trickle.modifiers).any() || trickle.buttons.any() || trickle.textInputted) {
            return false;
        }
        trickle.modifiers |= changed;
        trickle.keyChanged = true;
        trickle.nonCharKeyChanged = true;
        currentModifiers = next;
        return true;
    }

    if (const auto* text = std::get_if<InputQueue::Text>(&event)) {
        if (text->codepoint == 0) {
            return true;
        }
        if (trickle.buttons.any() || trickle.moved || trickle.wheeled || trickle.nonCharKeyChanged) {
            return false;
        }
        trickle.textInputted = true;
        frameEvents.push_back(TextEvent{ValidScalar(text->codepoint) ? text->codepoint : kReplacementCharacter,
                                        currentModifiers});
        return true;
    }

    return true;
}

bool InputState::movePointer(const std::optional<Extent2D<F32>>& next, Trickle& trickle) {
    if (pointerPosition == next) {
        return true;
    }
    if (trickle.buttons.any() || trickle.keyChanged) {
        return false;
    }
    trickle.moved = true;
    if (!next.has_value()) {
        cancelMomentum();
    }
    pointerPosition = next;
    MouseEvent move{};
    move.type = next.has_value() ? MouseEventType::Move : MouseEventType::Leave;
    move.position = next.value_or(Extent2D<F32>{0.0f, 0.0f});
    move.modifiers = currentModifiers;
    move.positioned = next.has_value();
    frameEvents.push_back(move);
    return true;
}

bool InputState::potentiallyChar(KeyCode key) const {
    const bool shortcut = config.macOS ? currentModifiers.super
                                       : (currentModifiers.control && !currentModifiers.alt);
    if (shortcut) {
        return false;
    }
    if ((key >= KeyCode::Digit0 && key <= KeyCode::Z) ||
        (key >= KeyCode::Apostrophe && key <= KeyCode::GraveAccent && key != KeyCode::Backslash) ||
        (key >= KeyCode::Keypad0 && key <= KeyCode::KeypadAdd)) {
        return true;
    }
    return key == KeyCode::Tab || key == KeyCode::Space || key == KeyCode::KeypadEqual;
}

bool InputState::button(MouseButton pressed, bool down, Trickle& trickle) {
    if (ButtonIndex(pressed) >= buttonDown.size()) {
        return true;
    }

    bool clearAlias = false;
    if (config.macOS && pressed == MouseButton::Left && ctrlLeftAsRight) {
        pressed = MouseButton::Right;
        clearAlias = !down;
    }

    if (buttonDown[ButtonIndex(pressed)] == down) {
        ctrlLeftAsRight = ctrlLeftAsRight && !clearAlias;
        return true;
    }

    bool setAlias = false;
    if (config.macOS && pressed == MouseButton::Left && down && currentModifiers.control) {
        setAlias = true;
        pressed = MouseButton::Right;
        if (buttonDown[ButtonIndex(pressed)]) {
            ctrlLeftAsRight = true;
            return true;
        }
    }

    const U64 index = ButtonIndex(pressed);
    if (trickle.buttons.test(index) || trickle.wheeled) {
        return false;
    }
    trickle.buttons.set(index);
    ctrlLeftAsRight = setAlias || (ctrlLeftAsRight && !clearAlias);
    buttonDown[index] = down;
    if (down) {
        cancelMomentum();
    }

    MouseEvent event{};
    event.type = down ? MouseEventType::Click : MouseEventType::Release;
    event.button = pressed;
    event.position = pointerPosition.value_or(Extent2D<F32>{0.0f, 0.0f});
    event.modifiers = currentModifiers;
    event.positioned = pointerPosition.has_value();

    if (down) {
        bool repeated = false;
        if (static_cast<F32>(elapsed - clickTime[index]) < kDoubleClickTime) {
            const auto& last = clickPosition[index];
            repeated = !pointerPosition.has_value();
            if (pointerPosition.has_value() && last.has_value()) {
                const Extent2D<F32> offset = {pointerPosition->x - last->x, pointerPosition->y - last->y};
                repeated = offset.x * offset.x + offset.y * offset.y < kDoubleClickDistance * kDoubleClickDistance;
            }
        }
        if (!repeated) {
            clickCount[index] = 1;
        } else if (clickCount[index] < std::numeric_limits<U8>::max()) {
            ++clickCount[index];
        }
        clickTime[index] = elapsed;
        clickPosition[index] = pointerPosition;
        event.clicks = clickCount[index];
    }

    frameEvents.push_back(event);
    return true;
}

void InputState::clear() {
    cancelMomentum();
    keys.fill(false);
    keyDuration.fill(-1.0f);
    currentModifiers = {};
    buttonDown.fill(false);
    clickCount.fill(0);
    clickTime.fill(-std::numeric_limits<F64>::infinity());
    ctrlLeftAsRight = false;
}

void InputState::cancelMomentum() {
    if (momentum.active() || samplingMomentum || gestureHeld) {
        auto end = momentumEvent;
        end.phase = samplingMomentum || gestureHeld ? ScrollPhase::Ended : ScrollPhase::MomentumEnded;
        end.scroll = {0.0f, 0.0f};
        frameEvents.push_back(end);
    }
    momentum.reset();
    samplingMomentum = false;
    gestureHeld = false;
}

void InputState::advanceMomentum() {
    if (!momentum.active()) {
        return;
    }
    auto event = momentumEvent;
    event.scroll = momentum.advance(elapsed - momentumSteppedAt);
    momentumSteppedAt = elapsed;
    if (event.scroll.x != 0.0f || event.scroll.y != 0.0f) {
        frameEvents.push_back(event);
    }
    if (!momentum.active()) {
        event.phase = ScrollPhase::MomentumEnded;
        event.scroll = {0.0f, 0.0f};
        frameEvents.push_back(event);
    }
}

void InputState::repeatKeys() {
    for (U64 index = 1; index < kKeyCount; ++index) {
        const F32 duration = keyDuration[index];
        if (!keys[index] || duration <= kKeyRepeatDelay) {
            continue;
        }
        if (TypematicRepeatAmount(duration - delta, duration, kKeyRepeatDelay, kKeyRepeatRate) <= 0) {
            continue;
        }
        frameEvents.push_back(KeyEvent{KeyEventType::Press, static_cast<KeyCode>(index), currentModifiers, true});
    }
}

}  // namespace Jetstream::Render
