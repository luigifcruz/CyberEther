#include "jetstream/compute/graph/generic.hh"

namespace Jetstream {

Result Graph::setExternallyWiredInput(const U64& input) {
    externallyWiredInputSet.emplace(input);
    return Result::SUCCESS;
}

Result Graph::setExternallyWiredOutput(const U64& output) {
    externallyWiredOutputSet.emplace(output);
    return Result::SUCCESS;
}

Result Graph::setModule(const std::shared_ptr<Compute>& block, 
                        const std::unordered_set<U64>& inputSet,
                        const std::unordered_set<U64>& outputSet) {
    computeUnits.push_back({
        .block = block,
        .inputSet = inputSet,
        .outputSet = outputSet,
    });

    for (const U64& hash : inputSet) {
        wiredInputSet.emplace(hash);
    }

    for (const U64& hash : outputSet) {
        wiredOutputSet.emplace(hash);
    }

    return Result::SUCCESS;
}

void Graph::Yield(std::unordered_set<U64>& yielded, const std::unordered_set<U64>& outputSet) {
    for (const auto& output : outputSet) {
        yielded.emplace(output);
    }
}

bool Graph::ShouldYield(std::unordered_set<U64>& yielded, const std::unordered_set<U64>& inputSet) {
    if (yielded.empty()) {
        return false;
    }

    return std::ranges::any_of(inputSet, [&](const U64& input) {
        return yielded.contains(input);
    });
}

}  // namespace Jetstream