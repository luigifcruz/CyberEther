#include "jetstream/graph/metal.hh"

namespace Jetstream {

Metal::Metal() {
    JST_DEBUG("Creating new Metal compute graph.");
    metadata = std::make_shared<RuntimeMetadata>();
}

const Result Metal::createCompute() {
    return Result::SUCCESS;
}

const Result Metal::compute() {
    return Result::SUCCESS;
}

}  // namespace Jetstream
