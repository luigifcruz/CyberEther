#include "jetstream/base.hpp"

namespace Jetstream {

Result Loop::compute() {
    DEBUG_PUSH("compute_wait");

    std::unique_lock<std::mutex> sync(m);
    access.wait(sync, [=]{ return !waiting; });

    DEBUG_POP();
    DEBUG_PUSH("compute");

    Result err = Result::SUCCESS;
    for (const auto & [name, mod] : blocks) {
        DEBUG_PUSH(name + "_compute");
        if ((err = mod->compute()) != Result::SUCCESS) {
            return err;
        }
        DEBUG_POP();
    }

    for (const auto & [name, mod] : blocks) {
        DEBUG_PUSH(name + "_barrier");
        if ((err = mod->barrier()) != Result::SUCCESS) {
            return err;
        }
        DEBUG_POP();
    }

    DEBUG_POP();

    return err;
}

Result Loop::present() {
    DEBUG_PUSH("present_wait");

    waiting = true;
    Result err = Result::SUCCESS;
    {
        const std::unique_lock<std::mutex> lock(m);

        DEBUG_POP();
        DEBUG_PUSH("present");

        for (const auto & [name, mod] : blocks) {
            DEBUG_PUSH(name);
            if ((err = mod->present()) != Result::SUCCESS) {
                return err;
            }
            DEBUG_POP();
        }
    }
    waiting = false;
    access.notify_one();

    DEBUG_POP();

    return err;
}

} // namespace Jetstream::Loop
