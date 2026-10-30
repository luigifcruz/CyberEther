#ifndef JETSTREAM_TOOLS_RANDOM_HH
#define JETSTREAM_TOOLS_RANDOM_HH

#include "jetstream/types.hh"

#include <chrono>
#include <cstdio>
#include <random>
#include <string>

namespace Jetstream::Random {

inline std::string HexToken() {
    static thread_local std::mt19937_64 rng([] {
        std::random_device device;
        const U64 seed = (static_cast<U64>(device()) << 32) ^ device();
        return seed ^ static_cast<U64>(
            std::chrono::steady_clock::now().time_since_epoch().count());
    }());
    std::uniform_int_distribution<U64> distribution;
    char buffer[33];
    std::snprintf(buffer, sizeof(buffer), "%016llx%016llx",
                  static_cast<unsigned long long>(distribution(rng)),
                  static_cast<unsigned long long>(distribution(rng)));
    return buffer;
}

}  // namespace Jetstream::Random

#endif  // JETSTREAM_TOOLS_RANDOM_HH
