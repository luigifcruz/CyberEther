#include "jetstream/base.hpp"

namespace Jetstream {

Result Engine::compute() {
    DEBUG_PUSH("compute_wait");

    std::unique_lock<std::mutex> sync(m);
    access.wait(sync, [=]{ return !waiting; });

    DEBUG_POP();
    DEBUG_PUSH("compute");

    for (const auto& transform : *this) {
        auto result = transform->compute();
        if (result != Result::SUCCESS) {
            return result;
        }
    }

    for (const auto& transform : *this) {
        auto result = transform->barrier();
        if (result != Result::SUCCESS) {
            return result;
        }
    }

    DEBUG_POP();

    return Result::SUCCESS;
}

Result Engine::present() {
    DEBUG_PUSH("present_wait");

    waiting = true;
    {
        const std::scoped_lock<std::mutex> lock(m);

        DEBUG_POP();
        DEBUG_PUSH("present");

        for (const auto& transform : *this) {
            auto result = transform->present();
            if (result != Result::SUCCESS) {
                return result;
            }
        }
    }
    waiting = false;
    access.notify_one();

    DEBUG_POP();

    return Result::SUCCESS;
}

} // namespace Jetstream
