#include "quick_tune_budget.hh"

#include <algorithm>

namespace Jetstream::Radio {

QuickTuneBudget::QuickTuneBudget(const std::optional<U64> profileCapacity)
    : capacity(profileCapacity) {}

QuickTuneBudget::Decision QuickTuneBudget::plan(const std::vector<F64>& frequencies) const {
    if (!prepared.empty() && prepared == frequencies) {
        return Decision::Reuse;
    }
    if (capacity && allocated + frequencies.size() > *capacity) {
        return Decision::Fallback;
    }
    return Decision::Build;
}

void QuickTuneBudget::built(const std::vector<F64>& frequencies) {
    allocated += frequencies.size();
    prepared = frequencies;
}

void QuickTuneBudget::failed(const U64 partial) {
    allocated += partial;
    prepared.clear();
}

void QuickTuneBudget::invalidate() {
    prepared.clear();
}

U64 QuickTuneBudget::used() const {
    return allocated;
}

std::optional<U64> QuickTuneBudget::remaining() const {
    if (!capacity) {
        return std::nullopt;
    }
    return *capacity - std::min(allocated, *capacity);
}

}  // namespace Jetstream::Radio
