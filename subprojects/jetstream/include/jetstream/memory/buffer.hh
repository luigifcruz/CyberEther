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
    CircularBuffer(U64);
    ~CircularBuffer();

    bool isEmpty() const;
    bool isFull() const;

    Result get(T*, U64);
    Result put(T*, U64);
    Result reset();

    Result waitBufferOccupancy(U64);

    U64 getCapacity() const;
    U64 getOccupancy() const;

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

