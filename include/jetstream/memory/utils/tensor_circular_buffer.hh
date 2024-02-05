#ifndef JETSTREAM_MEMORY_UTILS_TENSOR_CIRCULAR_BUFFER_H
#define JETSTREAM_MEMORY_UTILS_TENSOR_CIRCULAR_BUFFER_H

#include <memory>
#include <vector>

#include "jetstream/types.hh"

namespace Jetstream::Memory {

/**
 * @class TensorCircularBuffer
 * @brief A circular buffer implementation for storing tensors.
 * 
 * The TensorCircularBuffer class provides a circular buffer data structure for storing tensors of type T.
 * It works as a standard first-in-first-out (FIFO) queue but with a fixed tensor size.
 * It supports operations such as resizing, clearing, checking capacity, occupancy, emptiness, and fullness.
 * It also provides methods for waiting, getting, and putting tensors into the buffer.
 * 
 * @tparam T The type of the tensors to be stored in the buffer.
 */
template<typename T>
class TensorCircularBuffer {
 public:
    /**
     * @brief Default constructor.
     */
    TensorCircularBuffer() = default;

    /**
     * @brief Constructor that initializes the buffer with a given size.
     * @param size The size of the buffer.
     * @param args Additional arguments to be forwarded to the constructor of type T.
     */
    template<typename... Args>
    TensorCircularBuffer(const U64& size, Args&&... args) {
        resize(size, std::forward<Args>(args)...);
    }

    /**
     * @brief Resizes the buffer to a new size.
     * @param size The new size of the buffer.
     * @param args Additional arguments to be forwarded to the constructor of type T.
     */
    template<typename... Args>
    void resize(const U64& size, Args&&... args) {
        clear();
        pool.reserve(size);
        for (U64 i = 0; i < size; ++i) {
            pool.push_back(std::forward<Args>(args)...);
        }
    }

    /**
     * @brief Clears the buffer.
     */
    void clear() {
        pool.clear();
        head = 0;
        tail = 0;
    }

    /**
     * @brief Returns the capacity of the buffer.
     * @return The capacity of the buffer.
     */
    U64 capacity() const {
        return pool.size();
    }

    /**
     * @brief Returns the current occupancy of the buffer.
     * @return The current occupancy of the buffer.
     */
    U64 occupancy() const {
        return tail - head;
    }

    /**
     * @brief Checks if the buffer is empty.
     * @return True if the buffer is empty, false otherwise.
     */
    bool empty() const {
        return head == tail;
    }

    /**
     * @brief Checks if the buffer is full.
     * @return True if the buffer is full, false otherwise.
     */
    bool full() const {
        return occupancy() == capacity();
    }

    /**
     * @brief Returns the number of times the buffer has overflowed.
     * @return The number of times the buffer has overflowed.
     */
    const U64& overflows() const {
        return overflowCount;
    }

    /**
     * @brief Waits until the buffer is not empty or a timeout occurs.
     * @param timeout The timeout value in milliseconds (default: 1000 ms).
     * @return True if the buffer is not empty, false if a timeout occurs.
     */
    bool wait(const U64& timeout = 1000) {
        std::unique_lock<std::mutex> sync(mtx);
        while (empty()) {
            if (cv.wait_for(sync, std::chrono::milliseconds(timeout)) == std::cv_status::timeout) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Gets a tensor from the buffer.
     * @param buffer The variable to store the retrieved tensor.
     * @param timeout The timeout value in milliseconds (default: 1000 ms).
     * @return True if a tensor is successfully retrieved, false otherwise.
     */
    bool get(T& buffer, const U64& timeout = 1000) {
        if (empty()) {
            return false;
        }

        if (!wait(timeout)) {
            return false;
        }

        {
            std::lock_guard<std::mutex> guard(mtx);
            buffer = pool[head % capacity()];
            ++head;
        }

        return true;
    }

    /**
     * @brief Puts a tensor into the buffer.
     * @param callback The callback function to modify the tensor before putting it into the buffer.
     */
    void put(const std::function<void(T&)>& callback) {
        {
            std::lock_guard<std::mutex> guard(mtx);

            if (full()) {
                ++overflowCount;
                ++head;
            }

            callback(pool[tail % capacity()]);
            ++tail;
        }

        cv.notify_one();
    }

 private:
    std::vector<T> pool;
    U64 head = 0;
    U64 tail = 0;
    U64 overflowCount = 0;

    std::mutex mtx;
    std::condition_variable cv;
};

}  // namespace Jetstream::Memory

#endif

