#include "jetstream/instance.hh"

namespace Jetstream {

const Result Instance::schedule(const std::shared_ptr<Module>& block) {
    this->blocks.push_back(block);
    return Result::SUCCESS;
}

const Result Instance::present() {
    Result err = Result::SUCCESS;

    presentSync.test_and_set();
    computeSync.wait(true);

    for (const auto& block : blocks) {
        if ((err = block->present()) != Result::SUCCESS) {
            return err;
        }
    }

    presentSync.clear();
    presentSync.notify_one();

    return err;
}

const Result Instance::compute() {
    Result err = Result::SUCCESS;

    presentSync.wait(true);
    computeSync.test_and_set();

    for (const auto& block : blocks) {
        if ((err = block->compute()) != Result::SUCCESS) {
            return err;
        }
    }

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

Result Schedule(const std::shared_ptr<Module>& block) {
    return GetDefaultInstance().schedule(block);
}

Result Compute() {
    return GetDefaultInstance().compute();
}

Result Present() {
    return GetDefaultInstance().present();
}

}  // namespace Jetstream
