#include "quick_tune_budget.hh"

#include <algorithm>

namespace Jetstream::Radio {

BladeRfQuickTuneBudget::BladeRfQuickTuneBudget(const std::optional<U64> profileCapacity)
    : capacity(profileCapacity) {}

BladeRfQuickTuneBudget::Decision BladeRfQuickTuneBudget::plan(
    const std::vector<F64>& frequencies) const {
    if (!prepared.empty() && prepared == frequencies) {
        return Decision::Reuse;
    }
    if (capacity && allocated + frequencies.size() > *capacity) {
        return Decision::Fallback;
    }
    return Decision::Build;
}

void BladeRfQuickTuneBudget::built(const std::vector<F64>& frequencies) {
    allocated += frequencies.size();
    prepared = frequencies;
}

void BladeRfQuickTuneBudget::failed(const U64 partial) {
    allocated += partial;
    prepared.clear();
}

U64 BladeRfQuickTuneBudget::used() const {
    return allocated;
}

std::optional<U64> BladeRfQuickTuneBudget::remaining() const {
    if (!capacity) {
        return std::nullopt;
    }
    return *capacity - std::min(allocated, *capacity);
}

}  // namespace Jetstream::Radio
