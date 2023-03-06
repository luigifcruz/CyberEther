#include "jetstream/graph/generic.hh"

namespace Jetstream {

const Result Graph::schedule(const std::shared_ptr<Compute>& block) {
    blocks.push_back(block);
    return Result::SUCCESS;
}

}  // namespace Jetstream