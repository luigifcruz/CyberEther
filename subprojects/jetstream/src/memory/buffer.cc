#include "jetstream/memory/buffer.hh"

#define MIN(a,b) (((a)<(b))?(a):(b))

using namespace std::chrono_literals;

namespace Jetstream::Memory {

template<class T>
CircularBuffer<T>::CircularBuffer(size_t capacity) : capacity(capacity) {
    this->Reset();
    this->buffer = std::unique_ptr<T[]>(new T[Capacity()]);
}

template<class T>
CircularBuffer<T>::~CircularBuffer() {
    semaphore.notify_all();
    io_mtx.lock();
    buffer.reset();
}

template<class T>
const Result CircularBuffer<T>::WaitBufferOccupancy(size_t size) {
    std::unique_lock<std::mutex> sync(sync_mtx);
    while (Occupancy() < size) {
        if (semaphore.wait_for(sync, 5s) == std::cv_status::timeout)
            return Result::ERROR_TIMEOUT;
    }
    return Result::SUCCESS;
}

template<class T>
const Result CircularBuffer<T>::Get(T* buf, size_t size) {
    if (Capacity() < size) {
        return Result::ERROR_BEYOND_CAPACITY;
    }

    Result res = WaitBufferOccupancy(size);
    if (res != Result::SUCCESS)
        goto exception;

    {
        const std::lock_guard<std::mutex> lock(io_mtx);

        U64 stage_a = MIN(size, Capacity() - head);

        std::copy_n(buffer.get() + head, stage_a, buf);

        if (stage_a < size) {
            std::copy_n(buffer.get(), size - stage_a, buf + stage_a);
        }

        head = (head + size) % Capacity();
        occupancy -= size;
    }

exception:
    return res;
}

template<class T>
const Result CircularBuffer<T>::Put(T* buf, size_t size) {
    if (Capacity() < size) {
        return Result::ERROR_BEYOND_CAPACITY;
    }

    {
        const std::lock_guard<std::mutex> lock(io_mtx);

        if (Capacity() < (Occupancy() + size)) {
            printf("o");
            occupancy = 0;
            head = tail;
        }

        U64 stage_a = MIN(size, Capacity() - tail);
        std::copy_n(buf, stage_a, buffer.get() + tail);

        if (stage_a < size) {
            std::copy_n(buf + stage_a, size - stage_a, buffer.get());
        }

        tail = (tail + size) % Capacity();
        occupancy += size;
    }

    semaphore.notify_all();
    return Result::SUCCESS;
}

template<class T>
const Result CircularBuffer<T>::Reset() {
    {
        const std::lock_guard<std::mutex> lock(io_mtx);
        this->head = 0;
        this->tail = 0;
        this->occupancy = 0;
    }

    semaphore.notify_all();
    return Result::SUCCESS;
}

template<class T>
const U64 CircularBuffer<T>::Capacity() {
    return this->capacity;
}

template<class T>
const U64 CircularBuffer<T>::Occupancy() {
    return this->occupancy;
}

template<class T>
const bool CircularBuffer<T>::IsEmpty() {
    return Occupancy() == 0;
}

template<class T>
const bool CircularBuffer<T>::IsFull() {
    return Occupancy() == Capacity();
}

template class CircularBuffer<I8>;
template class CircularBuffer<CI8>;
template class CircularBuffer<F32>;
template class CircularBuffer<CF32>;
template class CircularBuffer<F64>;
template class CircularBuffer<CF64>;

}  // namespace Jetstream::Memory
