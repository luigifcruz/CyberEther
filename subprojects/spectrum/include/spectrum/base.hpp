#ifndef SPECTRUM_H
#define SPECTRUM_H

#include "types.hpp"

#include "spectrum/base/instance.hpp"
#include "spectrum/base/lineplot.hpp"

#ifdef SPECTRUM_FFTW_AVAILABLE
#include "spectrum/fftw/instance.hpp"
#include "spectrum/fftw/lineplot.hpp"
#endif

namespace Spectrum {

inline std::vector<API> AvailableAPIs = {
#ifdef SPECTRUM_FFTW_AVAILABLE
    API::FFTW,
#endif
};

inline std::shared_ptr<Instance> Instantiate(API api_hint, Instance::Config& cfg, bool force = false) {
    auto api = api_hint;

    if (std::find(AvailableAPIs.begin(), AvailableAPIs.end(),
                api_hint) == AvailableAPIs.end()) {
        if (force) {
            ASSERT_SUCCESS(Result::FAIL);
        }

        for (const auto& a : AvailableAPIs) {
#ifdef RENDER_DEBUG
            std::cout << "[SPECTRUM] Selected "
                      << magic_enum::enum_name(api_hint)
                      << " API not available, switching to "
                      << magic_enum::enum_name(a)
                      << "." << std::endl;
#endif
            api = a;
        }
    }

    switch (api) {
#ifdef SPECTRUM_FFTW_AVAILABLE
        case API::FFTW:
            return std::make_shared<FFTW>(cfg);
#endif
        default:
#ifdef RENDER_DEBUG
            std::cerr << "[SPECTRUM] No API available." << std::endl;
#endif
            ASSERT_SUCCESS(Result::FAIL);
    }
}

} // namespace Spectrum

#endif
