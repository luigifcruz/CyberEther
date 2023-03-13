#include "jetstream/graph/metal.hh"

namespace Jetstream {

Metal::Metal() {
    JST_DEBUG("Creating new Metal compute graph.");
    metadata = std::make_shared<RuntimeMetadata>();

    metadata->metal.commandBuffer = nullptr;

    auto device = Backend::State<Device::Metal>()->getDevice();
    commandQueue = device->newCommandQueue();
}

const Result Metal::createCompute() {
    for (const auto& block : blocks) {
        JST_CHECK(block->createCompute(*metadata));
    }

    return Result::SUCCESS;
}

const Result Metal::compute() {
    Result err = Result::SUCCESS;

    metadata->metal.commandBuffer = commandQueue->commandBuffer();

    for (const auto& block : blocks) {
        if ((err = block->compute(*metadata)) != Result::SUCCESS) {
            return err;
        }
    }

    metadata->metal.commandBuffer->commit();
    metadata->metal.commandBuffer->waitUntilCompleted();
    // metadata->metal.commandBuffer->release();

    return err;
}

}  // namespace Jetstream
