#ifndef JETSTREAM_MEMORY_BUFFER_H
#define JETSTREAM_MEMORY_BUFFER_H

#include <mutex>
#include <memory>
#include <condition_variable>
#include <chrono>
#include <complex>

#include "jetstream/types.hh"

namespace Jetstream::Memory {

template <class T>
class CircularBuffer {
public:
    CircularBuffer();
    CircularBuffer(const U64& capacity);
    ~CircularBuffer();

    bool isEmpty() const;
    bool isFull() const;

    Result get(T*, const U64& size);
    Result put(const T*, const U64& size);
    Result reset();
    Result resize(const U64& capacity);

    Result waitBufferOccupancy(const U64& occupancy);

    constexpr U64 getCapacity() const {
        return capacity;
    }

    constexpr U64 getOccupancy() const {
        return occupancy;
    }

    constexpr F64 getThroughput() const {
        return throughput;
    }

    constexpr U64 getOverflows() const {
        return overflows;
    }

private:
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

