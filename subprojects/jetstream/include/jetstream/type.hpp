#ifndef JETSTREAM_TYPE_H
#define JETSTREAM_TYPE_H

#include <complex>
#include <future>
#include <iostream>
#include <vector>

#include "jetstream_config.hpp"

#ifndef JETSTREAM_ASSERT_SUCCESS
#define JETSTREAM_ASSERT_SUCCESS(result) { \
    if (result != Jetstream::Result::SUCCESS) { \
        std::cerr << "Jetstream encountered an exception (" <<  magic_enum::enum_name(result) << ") in " \
            << __PRETTY_FUNCTION__ << " in line " << __LINE__ << " of file " << __FILE__ << "." << std::endl; \
        throw result; \
    } \
}
#endif

namespace Jetstream {

enum Result {
    SUCCESS = 0,
    ERROR = 1,
    UNKNOWN = 2,
    ERROR_FUTURE_INVALID,
};

namespace cpu {
    namespace arr {
        struct c32 {
            std::vector<std::complex<float>> data;
        };

        struct c64 {
            std::vector<std::complex<double>> data;
        };
    }
}

} // namespace Jetstream

#endif
