#ifndef SPECTRUM_TYPES_H
#define SPECTRUM_TYPES_H

#include <iostream>

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
};

} // namespace Spectrum

#endif