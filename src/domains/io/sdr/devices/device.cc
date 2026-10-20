#include "device.hh"

#include <algorithm>
#include <cmath>

namespace Jetstream::Radio {

bool Device::Range::Contains(const std::vector<Range>& ranges, const F64 value) {
    return std::isfinite(value) &&
           std::any_of(ranges.begin(), ranges.end(), [&](const auto& range) {
               return value >= range.minimum && value <= range.maximum;
           });
}

std::vector<F64> Device::Capabilities::fixedSampleRates() const {
    std::vector<F64> rates;
    for (const auto& range : sampleRateRanges) {
        if (range.minimum != range.maximum) {
            return {};
        }
        rates.push_back(range.minimum);
    }
    std::sort(rates.begin(), rates.end());
    rates.erase(std::unique(rates.begin(), rates.end()), rates.end());
    return rates;
}

}  // namespace Jetstream::Radio
