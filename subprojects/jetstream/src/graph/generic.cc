#include "jetstream/graph/generic.hh"

namespace Jetstream {

Result Graph::setWiredInputs(const std::vector<U64>& inputs) {
    for (const auto& input : inputs) {
        wiredInputSet.emplace(input);
    }
    return Result::SUCCESS;
}

Result Graph::setWiredOutputs(const std::vector<U64>& outputs) {
    for (const auto& output : outputs) {
        wiredOutputSet.emplace(output);
    }
    return Result::SUCCESS;
}

Result Graph::setExternallyWiredInputs(const std::vector<U64>& inputs) {
    for (const auto& input : inputs) {
        externallyWiredInputSet.emplace(input);
    }
    return Result::SUCCESS;
}

Result Graph::setExternallyWiredOutputs(const std::vector<U64>& outputs) {
    for (const auto& output : outputs) {
        externallyWiredOutputSet.emplace(output);
    }
    return Result::SUCCESS;
}

Result Graph::setModule(const std::shared_ptr<Compute>& block) {
    blocks.push_back(block);
    return Result::SUCCESS;
}

}  // namespace Jetstream