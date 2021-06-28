#include "jetstream/base.hpp"

namespace Jetstream {

Result Engine::compute() {
    DEBUG_PUSH("compute_wait");

    std::unique_lock<std::mutex> sync(m);
    access.wait(sync, [=]{ return !waiting; });

    DEBUG_POP();
    DEBUG_PUSH("compute");

    for (const auto& [key, worker] : stream) {
        CHECK(worker.run->compute());
    }

    for (const auto& [key, worker] : stream) {
        CHECK(worker.run->barrier());
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

        for (const auto& [key, worker] : stream) {
            CHECK(worker.mod->present());
        }
    }
    waiting = false;
    access.notify_one();

    DEBUG_POP();

    return Result::SUCCESS;
}

} // namespace Jetstream
