#ifndef JETSTREAM_RENDER_COMPONENTS_AXIS_TICKS_HH
#define JETSTREAM_RENDER_COMPONENTS_AXIS_TICKS_HH

#include <algorithm>
#include <cmath>
#include <vector>

#include "jetstream/types.hh"

namespace Jetstream::Render::Components {

struct AxisTicks {
    static constexpr U64 MaxValues = 1024;

    F64 step = 0.0;
    U64 minorDivisions = 5;
    U64 decimals = 0;
    std::vector<F64> values;

    static AxisTicks Compute(const F64 left, const F64 right, const F64 minimumStep) {
        AxisTicks ticks;
        const F64 span = right - left;
        if (!std::isfinite(left) || !std::isfinite(right) || !(span > 0.0) ||
            !std::isfinite(span)) {
            return ticks;
        }

        const F64 minimum = (std::isfinite(minimumStep) && minimumStep > 0.0)
            ? minimumStep
            : span / 10.0;
        const F64 base = std::pow(10.0, std::floor(std::log10(minimum)));
        U64 mantissa = 10;
        for (const U64 candidate : {1, 2, 5}) {
            if (static_cast<F64>(candidate) * base >= minimum * (1.0 - 1e-6)) {
                mantissa = candidate;
                break;
            }
        }
        ticks.step = static_cast<F64>(mantissa) * base;
        ticks.minorDivisions = (mantissa == 2) ? 4 : 5;
        ticks.decimals = static_cast<U64>(std::max(
            0.0, -std::floor(std::log10(ticks.step) + 1e-9)));
        ticks.values = Multiples(left, right, ticks.step, 0, MaxValues);

        return ticks;
    }

    std::vector<F64> minors(const F64 left, const F64 right) const {
        if (!(step > 0.0)) {
            return {};
        }

        return Multiples(left,
                         right,
                         step / static_cast<F64>(minorDivisions),
                         minorDivisions,
                         MaxValues * minorDivisions);
    }

 private:
    static std::vector<F64> Multiples(const F64 left,
                                      const F64 right,
                                      const F64 step,
                                      const U64 skipEvery,
                                      const U64 limit) {
        std::vector<F64> values;
        const F64 margin = (right - left) * 1e-6;
        const F64 first = std::ceil((left + margin) / step);
        const F64 last = std::floor((right - margin) / step);
        for (U64 i = 0; i < limit && first + static_cast<F64>(i) <= last; i++) {
            const F64 k = first + static_cast<F64>(i);
            if (skipEvery > 0 && std::fmod(k, static_cast<F64>(skipEvery)) == 0.0) {
                continue;
            }
            const F64 value = k * step;
            if (!values.empty() && value <= values.back()) {
                break;
            }
            values.push_back(value);
        }

        return values;
    }
};

}  // namespace Jetstream::Render::Components

#endif  // JETSTREAM_RENDER_COMPONENTS_AXIS_TICKS_HH
