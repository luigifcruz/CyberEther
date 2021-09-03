#ifndef JETSTREAM_TYPE_H
#define JETSTREAM_TYPE_H

#include <iostream>
#include <complex>
#include <vector>
#include <memory>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <span>
#include <chrono>
#include <atomic>

#include <gadget/base.hpp>
#include <gadget/tools/span.hpp>
using namespace Gadget;

namespace Jetstream {

enum class Locale : uint8_t {
    CPU     = 1 << 0,
    CUDA    = 1 << 1,
};

template<typename T>
struct Data {
    Locale location;
    T buf;
};

enum Result {
    SUCCESS = 0,
    ERROR = 1,
};

} // namespace Jetstream

#endif
