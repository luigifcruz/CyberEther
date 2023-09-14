#include "jetstream/compute/graph/generic.hh"

namespace Jetstream {

Result Graph::setWiredInput(const U64& input) {
    wiredInputSet.emplace(input);
    return Result::SUCCESS;
}

Result Graph::setWiredOutput(const U64& output) {
    wiredOutputSet.emplace(output);
    return Result::SUCCESS;
}

Result Graph::setExternallyWiredInput(const U64& input) {
    externallyWiredInputSet.emplace(input);
    return Result::SUCCESS;
}

Result Graph::setExternallyWiredOutput(const U64& output) {
    externallyWiredOutputSet.emplace(output);
    return Result::SUCCESS;
}

Result Graph::setModule(const std::shared_ptr<Compute>& block) {
    blocks.push_back(block);
    return Result::SUCCESS;
}

}  // namespace Jetstream