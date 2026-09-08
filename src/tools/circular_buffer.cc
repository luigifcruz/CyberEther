#include <algorithm>
#include <complex>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "jetstream/tools/circular_buffer.hh"

namespace Jetstream::Tools {

template<class T>
struct CircularBuffer<T>::Impl {
    [[nodiscard]] U64 physicalIndex(U64 offset) const {
        const U64 bufferCapacity = static_cast<U64>(storage.size());
        if (bufferCapacity == 0) {
            return 0;
        }
        const U64 tailSize = bufferCapacity - head;
        return offset < tailSize ? head + offset : offset - tailSize;
    }

    void copyOut(U64 offset, T* data, U64 size) const {
        const U64 bufferCapacity = static_cast<U64>(storage.size());
        if (size == 0) {
            return;
        }
        const U64 source = physicalIndex(offset);
        const U64 firstSize = std::min(size, bufferCapacity - source);
        std::copy_n(storage.data() + source,
                    static_cast<std::size_t>(firstSize),
                    data);
        if (firstSize < size) {
            std::copy_n(storage.data(),
                        static_cast<std::size_t>(size - firstSize),
                        data + firstSize);
        }
    }

    void copyOutStrided(U64 offset, T* data, U64 size, U64 stride) const {
        for (U64 i = 0; i < size; ++i) {
            data[i * stride] = storage[physicalIndex(offset + i)];
        }
    }

    void discard(U64 size) {
        if (size == 0) {
            return;
        }
        head = physicalIndex(size);
        occupancy -= size;
        if (occupancy == 0) {
            head = 0;
        }
    }

    void reset() {
        head = 0;
        occupancy = 0;
        overflowCount = 0;
        transfers = 0;
        measuredThroughput = 0.0;
        lastTransfer = std::chrono::steady_clock::now();
    }

    void updateThroughput(U64 size) {
        transfers += size;
        const auto now = std::chrono::steady_clock::now();
        const std::chrono::duration<F64> elapsed = now - lastTransfer;
        if (elapsed.count() > 0.5) {
            measuredThroughput = static_cast<F64>(transfers) / elapsed.count();
            transfers = 0;
            lastTransfer = now;
        }
    }

    mutable std::mutex mutex;
    std::condition_variable dataAvailable;
    std::vector<T> storage;
    CircularBufferOverflowPolicy policy =
        CircularBufferOverflowPolicy::OverwriteOldest;
    U64 head = 0;
    U64 occupancy = 0;
    U64 overflowCount = 0;
    U64 transfers = 0;
    F64 measuredThroughput = 0.0;
    std::chrono::steady_clock::time_point lastTransfer =
        std::chrono::steady_clock::now();
};

template<class T>
CircularBuffer<T>::CircularBuffer()
    : pimpl(std::make_unique<Impl>()) {}

template<class T>
CircularBuffer<T>::CircularBuffer(const U64 capacity,
                                  const CircularBufferOverflowPolicy overflowPolicy)
    : pimpl(std::make_unique<Impl>()) {
    if (capacity > static_cast<U64>(std::numeric_limits<std::size_t>::max())) {
        throw std::length_error("Circular buffer capacity exceeds size_t.");
    }
    pimpl->storage.resize(static_cast<std::size_t>(capacity));
    pimpl->policy = overflowPolicy;
}

template<class T>
CircularBuffer<T>::~CircularBuffer() = default;

template<class T>
CircularBuffer<T>::CircularBuffer(CircularBuffer&& other)
    : pimpl(std::make_unique<Impl>()) {
    pimpl.swap(other.pimpl);
}

template<class T>
CircularBuffer<T>& CircularBuffer<T>::operator=(CircularBuffer&& other) noexcept {
    if (this != &other) {
        pimpl.swap(other.pimpl);
    }
    return *this;
}

template<class T>
Result CircularBuffer<T>::push(const T* data, const U64 size) {
    return pushStrided(data, size, 1);
}

template<class T>
Result CircularBuffer<T>::pushStrided(const T* data,
                                      const U64 size,
                                      const U64 stride) {
    if (size > 0 && data == nullptr) {
        return Result::ERROR;
    }

    {
        std::lock_guard lock(pimpl->mutex);
        const U64 bufferCapacity = static_cast<U64>(pimpl->storage.size());
        if (size == 0) {
            return Result::SUCCESS;
        }
        if (bufferCapacity == 0) {
            ++pimpl->overflowCount;
            return Result::ERROR;
        }

        U64 sourceOffset = 0;
        U64 writeSize = size;
        if (size > bufferCapacity - pimpl->occupancy) {
            ++pimpl->overflowCount;
            if (pimpl->policy == CircularBufferOverflowPolicy::Reject) {
                return size > bufferCapacity ? Result::ERROR : Result::INCOMPLETE;
            }

            if (size >= bufferCapacity) {
                sourceOffset = size - bufferCapacity;
                writeSize = bufferCapacity;
                pimpl->head = 0;
                pimpl->occupancy = 0;
            } else {
                pimpl->discard(pimpl->occupancy + size - bufferCapacity);
            }
        }

        const U64 tail = pimpl->physicalIndex(pimpl->occupancy);
        for (U64 i = 0; i < writeSize; ++i) {
            U64 destination = tail + i;
            if (destination >= bufferCapacity) {
                destination -= bufferCapacity;
            }
            pimpl->storage[static_cast<std::size_t>(destination)] =
                data[(sourceOffset + i) * stride];
        }
        pimpl->occupancy += writeSize;
    }
    pimpl->dataAvailable.notify_all();
    return Result::SUCCESS;
}

template<class T>
Result CircularBuffer<T>::pop(T* data, const U64 size) {
    return popStrided(data, size, 1);
}

template<class T>
Result CircularBuffer<T>::popStrided(T* data,
                                    const U64 size,
                                    const U64 stride) {
    if (size > 0 && data == nullptr) {
        return Result::ERROR;
    }

    std::lock_guard lock(pimpl->mutex);
    if (size > pimpl->occupancy) {
        return Result::INCOMPLETE;
    }
    pimpl->copyOutStrided(0, data, size, stride);
    pimpl->discard(size);
    pimpl->updateThroughput(size);
    return Result::SUCCESS;
}

template<class T>
Result CircularBuffer<T>::peek(const U64 offset,
                               T* data,
                               const U64 size) const {
    if (size > 0 && data == nullptr) {
        return Result::ERROR;
    }

    std::lock_guard lock(pimpl->mutex);
    if (offset > pimpl->occupancy || size > pimpl->occupancy - offset) {
        return Result::INCOMPLETE;
    }
    pimpl->copyOut(offset, data, size);
    return Result::SUCCESS;
}

template<class T>
Result CircularBuffer<T>::discard(const U64 size) {
    std::lock_guard lock(pimpl->mutex);
    if (size > pimpl->occupancy) {
        return Result::INCOMPLETE;
    }
    pimpl->discard(size);
    return Result::SUCCESS;
}

template<class T>
Result CircularBuffer<T>::clear() {
    {
        std::lock_guard lock(pimpl->mutex);
        pimpl->reset();
    }
    pimpl->dataAvailable.notify_all();
    return Result::SUCCESS;
}

template<class T>
Result CircularBuffer<T>::resize(const U64 capacity) {
    if (capacity > static_cast<U64>(std::numeric_limits<std::size_t>::max())) {
        return Result::ERROR;
    }

    std::vector<T> replacement;
    try {
        replacement.resize(static_cast<std::size_t>(capacity));
    } catch (const std::exception&) {
        return Result::ERROR;
    }

    {
        std::lock_guard lock(pimpl->mutex);
        pimpl->storage = std::move(replacement);
        pimpl->reset();
    }
    pimpl->dataAvailable.notify_all();
    return Result::SUCCESS;
}

template<class T>
Result CircularBuffer<T>::waitForSize(
    const U64 size,
    const std::chrono::milliseconds timeout) {
    std::unique_lock lock(pimpl->mutex);
    if (size > pimpl->storage.size()) {
        return Result::ERROR;
    }
    if (!pimpl->dataAvailable.wait_for(lock, timeout, [&] {
            return pimpl->occupancy >= size || size > pimpl->storage.size();
        })) {
        return Result::TIMEOUT;
    }
    return size > pimpl->storage.size() ? Result::ERROR : Result::SUCCESS;
}

template<class T>
bool CircularBuffer<T>::empty() const {
    return size() == 0;
}

template<class T>
bool CircularBuffer<T>::full() const {
    std::lock_guard lock(pimpl->mutex);
    return !pimpl->storage.empty() &&
           pimpl->occupancy == pimpl->storage.size();
}

template<class T>
U64 CircularBuffer<T>::capacity() const {
    std::lock_guard lock(pimpl->mutex);
    return static_cast<U64>(pimpl->storage.size());
}

template<class T>
U64 CircularBuffer<T>::size() const {
    std::lock_guard lock(pimpl->mutex);
    return pimpl->occupancy;
}

template<class T>
F64 CircularBuffer<T>::throughput() const {
    std::lock_guard lock(pimpl->mutex);
    return pimpl->measuredThroughput;
}

template<class T>
U64 CircularBuffer<T>::overflows() const {
    std::lock_guard lock(pimpl->mutex);
    return pimpl->overflowCount;
}

template class CircularBuffer<I8>;
template class CircularBuffer<CI8>;
template class CircularBuffer<F32>;
template class CircularBuffer<CF32>;
template class CircularBuffer<F64>;
template class CircularBuffer<CF64>;

}  // namespace Jetstream::Tools
