#include "jetstream/base.hpp"

namespace Jetstream {

const Result Instance::stream(const std::vector<std::shared_ptr<Module>>& modules) {
    this->modules = modules; 
    return Result::SUCCESS;
}

const Result Instance::present() {
    Result err = Result::SUCCESS;

    DEBUG_PUSH("present_wait");

    presentSync.test_and_set();
    computeSync.wait(true);

    DEBUG_POP();
    DEBUG_PUSH("present");

    for (const auto& module : modules) {
        if ((err = module->present()) != Result::SUCCESS) {
            return err;
        }
    }

    presentSync.clear();
    presentSync.notify_one();

    DEBUG_POP();

    return err;
}

const Result Instance::compute() {
    Result err = Result::SUCCESS;

    DEBUG_PUSH("compute_wait");

    presentSync.wait(true);
    computeSync.test_and_set();

    DEBUG_POP();
    DEBUG_PUSH("compute");

    for (const auto& module : modules) {
        if ((err = module->compute()) != Result::SUCCESS) {
            return err;
        }
    }

    DEBUG_POP();

    computeSync.clear();
    computeSync.notify_one();

    return err;
}

static Instance& GetDefaultInstance() {
    static std::unique_ptr<Instance> instance;

    if (!instance) {
        instance = std::make_unique<Instance>();
    }

    return *instance;
}

Result Stream(const std::vector<std::shared_ptr<Module>>& stream) {
    return GetDefaultInstance().stream(stream);
}

Result Compute() {
    return GetDefaultInstance().compute();
}

Result Present() {
    return GetDefaultInstance().present();
}

}  // namespace Jetstream 
