#ifndef JETSTREAM_MEMORY_BUFFER_H
#define JETSTREAM_MEMORY_BUFFER_H

#include <mutex>
#include <cstring>
#include <memory>
#include <algorithm>
#include <condition_variable>
#include <chrono>

#include "jetstream/types.hh"

namespace Jetstream::Memory {

template <class T>
class CircularBuffer {
public:
    // TODO: Convert to typedef standard.
    CircularBuffer(size_t);
    ~CircularBuffer();

    const bool IsEmpty() const;
    const bool IsFull() const;

    const Result Get(T*, size_t);
    const Result Put(T*, size_t);
    const Result Reset();

    const Result WaitBufferOccupancy(size_t);

    const U64 GetCapacity() const;
    const U64 GetOccupancy() const;

    constexpr const F64 GetThroughput() const {
        return throughput;
    }

    constexpr const U64 GetOverflows() const {
        return overflows;
    }

private:
    // TODO: Replace with non-blocking atomics.
    std::mutex io_mtx;
    std::mutex sync_mtx;
    std::condition_variable semaphore;

    std::unique_ptr<T[]> buffer{};

    U64 transfers;
    F64 throughput;
    std::chrono::system_clock::time_point lastGet;

    U64 head;
    U64 tail;
    U64 capacity;
    U64 occupancy;
    U64 overflows;
};

}  // namespace Jetstream::Memory

#endif

