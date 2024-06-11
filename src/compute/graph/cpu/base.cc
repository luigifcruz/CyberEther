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
    for (const auto& computeUnit : computeUnits) {
        JST_CHECK(computeUnit.block->createCompute(*context));
    }
    return Result::SUCCESS;
}

Result CPU::computeReady() {
    for (const auto& computeUnit : computeUnits) {
        JST_CHECK(computeUnit.block->computeReady());
    }
    return Result::SUCCESS;
}

Result CPU::compute(std::unordered_set<U64>& yielded) {
    for (const auto& computeUnit : computeUnits) {
        if (Graph::ShouldYield(yielded, computeUnit.inputSet)) {
            Graph::Yield(yielded, computeUnit.outputSet);
            continue;
        }

        const auto& res = computeUnit.block->compute(*context);

        if (res == Result::SUCCESS) {
            continue;
        }

        if (res == Result::YIELD) {
            Graph::Yield(yielded, computeUnit.outputSet);
            continue;
        }

        JST_CHECK(res);
    }
    return Result::SUCCESS;
}

Result CPU::destroy() {
    for (const auto& computeUnit : computeUnits) {
        JST_CHECK(computeUnit.block->destroyCompute(*context));
    }
    computeUnits.clear();
    return Result::SUCCESS;
}

}  // namespace Jetstream
