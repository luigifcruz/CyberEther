#include "jetstream/instance.hh"

namespace Jetstream {

Instance::~Instance() {
    if (_window) {
        _window->destroy();
    }
}

const Result Instance::commit() {
    JST_INFO("Building compute graph.");

    // Check minimum requirements.
    if (commited) {
        JST_FATAL("The instance was already commited.");
        return Result::ERROR;
    }

    if (blocks.size() < 1) {
        JST_ERROR("No blocks were added to this instance.");
        return Result::ERROR;
    }

    // Register Blocks with Compute;
    std::vector<std::tuple<std::shared_ptr<Module>, 
                           std::shared_ptr<Compute>>> computeBlocks;

    for (const auto& moduleBlock : blocks) {
        if (auto computeBlock = std::dynamic_pointer_cast<Compute>(moduleBlock)) {
            computeBlocks.push_back({moduleBlock, computeBlock});
        }
    }

    Device lastDevice = Device::None; 
    for (auto& [moduleBlock, computeBlock] : computeBlocks) {
        if ((moduleBlock->device() & lastDevice) != lastDevice) {
            graphs.push_back(NewGraph(moduleBlock->device()));
        }
        lastDevice = moduleBlock->device();
        graphs.back()->schedule(computeBlock);
    }

    // Register Blocks with Present.
    for (const auto& moduleBlock : blocks) {
        if (auto presentBlock = std::dynamic_pointer_cast<Present>(moduleBlock)) {
            if (!_window) {
                JST_ERROR("A window is required since a present block was added.");
                return Result::ERROR;
            }
            presentBlocks.push_back(presentBlock);
        }
    }

    // Creating module internals.
    for (const auto& graph : graphs) {
        JST_CHECK(graph->createCompute());
    }

    for (const auto& block : presentBlocks) {
        JST_CHECK(block->createPresent(*_window));
    }

    if (_window) {
        JST_CHECK(_window->create());
    }

    commited = true;

    return Result::SUCCESS;
}

const Result Instance::compute() {
    Result err = Result::SUCCESS;

    if (!commited) {
        JST_FATAL("The instance wasn't commited.");
        return Result::ERROR;
    }

    presentSync.wait(true);
    computeSync.test_and_set();

    for (const auto& graph : graphs) {
        JST_CHECK(graph->compute());
    }

    computeSync.clear();
    computeSync.notify_one();

    return err;
}

const Result Instance::begin() {
    if (_window) {
        return _window->begin();
    }

    return Result::SUCCESS;
}

const Result Instance::present() {
    Result err = Result::SUCCESS;

    if (!commited) {
        JST_FATAL("The instance wasn't commited.");
        return Result::ERROR;
    }

    presentSync.test_and_set();
    computeSync.wait(true);

    for (const auto& block : presentBlocks) {
        JST_CHECK(block->present(*_window));
    }

    presentSync.clear();
    presentSync.notify_one();

    return err;
}

const Result Instance::end() {
    if (_window) {
        return _window->end();
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream
