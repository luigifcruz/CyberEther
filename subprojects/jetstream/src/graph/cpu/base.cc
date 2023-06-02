#include "jetstream/graph/cpu.hh"

namespace Jetstream {

CPU::CPU() {
    JST_DEBUG("Creating new CPU compute graph.");
    metadata = std::make_shared<RuntimeMetadata>();
}

Result CPU::createCompute() {
    for (const auto& block : blocks) {
        JST_CHECK(block->createCompute(*metadata));
    }

    return Result::SUCCESS;
}

Result CPU::compute() {
    Result err = Result::SUCCESS;

    for (const auto& block : blocks) {
        if ((err = block->compute(*metadata)) != Result::SUCCESS) {
            return err;
        }
    }

    return Result::SUCCESS;
}

Result CPU::destroyCompute() {
    for (const auto& block : blocks) {
        JST_CHECK(block->destroyCompute(*metadata));
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream
