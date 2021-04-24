#ifndef SPECTRUM_TYPES_H
#define SPECTRUM_TYPES_H

#include <vector>
#include <memory>
#include <iostream>
#include <algorithm>
#include <complex>
#include <cstddef>
#include <unistd.h>

#include "render/base.hpp"
#include "spectrum_config.hpp"
#include "magic_enum.hpp"

namespace Spectrum {

#ifndef ASSERT_SUCCESS
#define ASSERT_SUCCESS(result) { \
    if (result != Result::SUCCESS) { \
        std::cerr << "Spectrum encountered an exception (" <<  magic_enum::enum_name(result) << ") in line " \
            << __LINE__ << " of file " << __FILE__ << "." << std::endl; \
        throw result; \
    } \
}
#endif

enum struct Result {
    SUCCESS = 0,
    FAIL,
};

enum struct DataFormat {
    F32,
    F64,
    CF32,
    CF64,
};

enum struct API {
    FFTW,
    CUDA,
    SYCL,
    SIGX,
};

} // namespace Spectrum

#endif
