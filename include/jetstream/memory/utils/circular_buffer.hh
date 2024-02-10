#ifndef JETSTREAM_MEMORY_UTILS_CIRCULAR_BUFFER_H
#define JETSTREAM_MEMORY_UTILS_CIRCULAR_BUFFER_H

#include <mutex>
#include <memory>
#include <condition_variable>
#include <chrono>
#include <complex>

#include "jetstream/types.hh"

namespace Jetstream::Memory {

/**
 * @class CircularBuffer
 * @brief A circular buffer implementation for storing elements of type T.
 * 
 * The CircularBuffer class provides a thread-safe circular buffer that can be used to store elements of type T.
 * It supports operations like getting elements, putting elements, resetting the buffer, and resizing the buffer capacity.
 * It also provides information about the buffer's capacity, occupancy, throughput, and overflows.
 */
template <class T>
class CircularBuffer {
public:
    /**
     * @brief Default constructor.
     */
    CircularBuffer();

    /**
     * @brief Constructor that initializes the buffer with a given capacity.
     * @param capacity The initial capacity of the buffer. Element number. Not byte size.
     */
    CircularBuffer(const U64& capacity);

    /**
     * @brief Destructor.
     */
    ~CircularBuffer();

    /**
     * @brief Check if the buffer is empty.
     * 
     * @return True if the buffer is empty, false otherwise.
     */
    bool isEmpty() const;

    /**
     * @brief Check if the buffer is full.
     * 
     * @return True if the buffer is full, false otherwise.
     */
    bool isFull() const;

    /**
     * @brief Get elements from the buffer.
     * @param[out] data Pointer to the memory where the elements will be stored.
     * @param size The number of elements to get.
     * 
     * @return Result indicating the success or failure of the operation.
     */
    Result get(T* data, const U64& size);

    /**
     * @brief Put elements into the buffer.
     * @param[in] data Pointer to the memory containing the elements to put.
     * @param size The number of elements to put.
     * 
     * @return Result indicating the success or failure of the operation.
     */
    Result put(const T* data, const U64& size);

    /**
     * @brief Reset the buffer.
     * @return Result indicating the success or failure of the operation.
     */
    Result reset();

    /**
     * @brief Resize the buffer capacity.
     * @param capacity The new capacity of the buffer.
     * 
     * @return Result indicating the success or failure of the operation.
     */
    Result resize(const U64& capacity);

    /**
     * @brief Wait until the buffer occupancy reaches a specified value.
     * @note This function is blocking with a timeout of 5 seconds.
     * @param occupancy The desired occupancy value.
     * 
     * @return Result indicating the success or failure of the operation.
     */
    Result waitBufferOccupancy(const U64& occupancy);

    /**
     * @brief Get the capacity of the buffer.
     * 
     * @return The capacity of the buffer.
     */
    constexpr U64 getCapacity() const {
        return capacity;
    }

    /**
     * @brief Get the current occupancy of the buffer.
     * 
     * @return The current occupancy of the buffer.
     */
    constexpr U64 getOccupancy() const {
        return occupancy;
    }

    /**
     * @brief Get the current throughput of the buffer.
     * 
     * @return The current throughput of the buffer.
     */
    constexpr F64 getThroughput() const {
        return throughput;
    }

    /**
     * @brief Get the number of overflows that have occurred in the buffer.
     * 
     * @return The number of overflows.
     */
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
