#include "jetstream/compute/graph/cpu.hh"

namespace Jetstream {

CPU::CPU() {
    JST_DEBUG("Creating new CPU compute graph.");
    context = std::make_shared<Compute::Context>();
    context->cpu = this;
}

CPU::~CPU() {
    context.reset();
}

Result CPU::create() {
    for (const auto& block : blocks) {
        JST_CHECK(block->createCompute(*context));
    }
    return Result::SUCCESS;
}

Result CPU::computeReady() {
    for (const auto& block : blocks) {
        JST_CHECK(block->computeReady());
    }
    return Result::SUCCESS;
}

Result CPU::compute() {
    for (const auto& block : blocks) { 
        JST_CHECK(block->compute(*context));
    }
    return Result::SUCCESS;
}

Result CPU::destroy() {
    for (const auto& block : blocks) {
        JST_CHECK(block->destroyCompute(*context));
    }
    blocks.clear();
    return Result::SUCCESS;
}

}  // namespace Jetstream
