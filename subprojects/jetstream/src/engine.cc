#include "jetstream/base.hpp"

namespace Jetstream {

Result Loop::compute() {
    GT_DEBUG_PUSH("compute_wait");

    std::unique_lock<std::mutex> sync(m);
    access.wait(sync, [=]{ return !waiting; });

    GT_DEBUG_POP();
    GT_DEBUG_PUSH("compute");

    Result err = Result::SUCCESS;
    for (const auto & [name, mod] : blocks) {
        GT_DEBUG_PUSH(name + "_compute");
        if ((err = mod->compute()) != Result::SUCCESS) {
            return err;
        }
        GT_DEBUG_POP();
    }

    for (const auto & [name, mod] : blocks) {
        GT_DEBUG_PUSH(name + "_barrier");
        if ((err = mod->barrier()) != Result::SUCCESS) {
            return err;
        }
        GT_DEBUG_POP();
    }

    GT_DEBUG_POP();

    return err;
}

Result Loop::present() {
    GT_DEBUG_PUSH("present_wait");

    waiting = true;
    Result err = Result::SUCCESS;
    {
        const std::unique_lock<std::mutex> lock(m);

        GT_DEBUG_POP();
        GT_DEBUG_PUSH("present");

        for (const auto & [name, mod] : blocks) {
            GT_DEBUG_PUSH(name);
            if ((err = mod->present()) != Result::SUCCESS) {
                return err;
            }
            GT_DEBUG_POP();
        }
    }
    waiting = false;
    access.notify_one();

    GT_DEBUG_POP();

    return err;
}

} // namespace Jetstream::Loop
