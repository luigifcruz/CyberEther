#include "jetstream/platform.hh"

#include <condition_variable>
#include <mutex>

#if defined(JST_OS_WINDOWS)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef ERROR
#undef FATAL
#endif

namespace Jetstream::Platform {

struct DeadlineWaiter::Impl {
    std::mutex mutex;
    std::condition_variable condition;
    U64 generation = 0;

#if defined(JST_OS_WINDOWS)
    struct Slot {
        HANDLE timer = CreateWaitableTimerExW(nullptr,
                                              nullptr,
                                              CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                              TIMER_MODIFY_STATE | SYNCHRONIZE);
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        bool active = false;

        ~Slot() {
            if (timer) {
                CloseHandle(timer);
            }
            if (event) {
                CloseHandle(event);
            }
        }
    };

    std::vector<std::unique_ptr<Slot>> slots;

    Slot* acquire() {
        for (auto& slot : slots) {
            if (!slot->active) {
                return slot.get();
            }
        }

        slots.push_back(std::make_unique<Slot>());
        return slots.back().get();
    }
#endif
};

DeadlineWaiter::DeadlineWaiter() : impl(std::make_unique<Impl>()) {}

DeadlineWaiter::~DeadlineWaiter() = default;

U64 DeadlineWaiter::generation() const {
    std::lock_guard lock(impl->mutex);
    return impl->generation;
}

void DeadlineWaiter::notify() {
    {
        std::lock_guard lock(impl->mutex);
        ++impl->generation;

#if defined(JST_OS_WINDOWS)
        for (auto& slot : impl->slots) {
            if (slot->active && slot->event) {
                SetEvent(slot->event);
            }
        }
#endif
    }
    impl->condition.notify_all();
}

bool DeadlineWaiter::waitUntil(const Clock::time_point deadline, const U64 observedGeneration) {
    std::unique_lock lock(impl->mutex);
    const auto notified = [&] {
        return impl->generation != observedGeneration;
    };

    if (notified()) {
        return true;
    }
    if (Clock::now() >= deadline) {
        return false;
    }

#if defined(JST_OS_WINDOWS)
    auto* slot = impl->acquire();
    if (slot->timer && slot->event && ResetEvent(slot->event)) {
        using Tick = std::chrono::duration<LONGLONG, std::ratio<1, 10000000>>;
        const HANDLE handles[] = {slot->event, slot->timer};
        slot->active = true;

        while (!notified()) {
            const auto now = Clock::now();
            if (now >= deadline) {
                break;
            }

            LARGE_INTEGER due;
            due.QuadPart = -std::chrono::ceil<Tick>(deadline - now).count();
            if (!SetWaitableTimer(slot->timer, &due, 0, nullptr, nullptr, FALSE)) {
                impl->condition.wait_until(lock, deadline, notified);
                break;
            }

            lock.unlock();
            const auto result = WaitForMultipleObjects(2, handles, FALSE, INFINITE);
            lock.lock();

            if (result == WAIT_FAILED) {
                impl->condition.wait_until(lock, deadline, notified);
                break;
            }
        }

        CancelWaitableTimer(slot->timer);
        slot->active = false;

        return notified();
    }
#endif

    return impl->condition.wait_until(lock, deadline, notified);
}

}  // namespace Jetstream::Platform
