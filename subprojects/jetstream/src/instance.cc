#include "jetstream/instance.hh"

namespace Jetstream {

const Result Instance::conduit(const std::vector<std::shared_ptr<Module>>& modules) {
    this->modules = modules; 
    return Result::SUCCESS;
}

const Result Instance::present() {
    Result err = Result::SUCCESS;

    presentSync.test_and_set();
    computeSync.wait(true);

    for (const auto& module : modules) {
        if ((err = module->present()) != Result::SUCCESS) {
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

    for (const auto& module : modules) {
        if ((err = module->compute()) != Result::SUCCESS) {
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

Result Conduit(const std::vector<std::shared_ptr<Module>>& stream) {
    return GetDefaultInstance().conduit(stream);
}

Result Compute() {
    return GetDefaultInstance().compute();
}

Result Present() {
    return GetDefaultInstance().present();
}

}  // namespace Jetstream
