#include "jetstream/memory/buffer.hh"

using namespace std::chrono_literals;

namespace Jetstream::Memory {

template<class T>
CircularBuffer<T>::CircularBuffer(U64 capacity)
     : transfers(0),
       throughput(0.0),
       capacity(capacity),
       overflows(0) {
    this->reset();
    this->buffer = std::unique_ptr<T[]>(new T[getCapacity()]);
}

template<class T>
CircularBuffer<T>::~CircularBuffer() {
    semaphore.notify_all();
    io_mtx.lock();
    buffer.reset();
}

template<class T>
const Result CircularBuffer<T>::waitBufferOccupancy(U64 size) {
    std::unique_lock<std::mutex> sync(sync_mtx);
    while (getOccupancy() < size) {
        if (semaphore.wait_for(sync, 5s) == std::cv_status::timeout)
            return Result::ERROR_TIMEOUT;
    }
    return Result::SUCCESS;
}

template<class T>
const Result CircularBuffer<T>::get(T* buf, U64 size) {
    if (getCapacity() < size) {
        return Result::ERROR_BEYOND_CAPACITY;
    }

    Result res = waitBufferOccupancy(size);
    if (res != Result::SUCCESS)
        goto exception;

    {
        const std::lock_guard<std::mutex> lock(io_mtx);

        U64 stage_a = JST_MIN(size, getCapacity() - head);

        std::copy_n(buffer.get() + head, stage_a, buf);

        if (stage_a < size) {
            std::copy_n(buffer.get(), size - stage_a, buf + stage_a);
        }

        head = (head + size) % getCapacity();
        occupancy -= size;

        // Throughput Calculator
        auto now = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = now - lastGet;

        if (elapsed.count() > 0.5) {
            throughput = (transfers * sizeof(T)) / elapsed.count();
            transfers = 0.0;
            lastGet = std::chrono::system_clock::now();
        }

        transfers += size;
    }

exception:
    return res;
}

template<class T>
const Result CircularBuffer<T>::put(T* buf, U64 size) {
    if (getCapacity() < size) {
        return Result::ERROR_BEYOND_CAPACITY;
    }

    {
        const std::lock_guard<std::mutex> lock(io_mtx);

        if (getCapacity() < (getOccupancy() + size)) {
            overflows += 1;
            occupancy = 0;
            head = tail;
        }

        U64 stage_a = JST_MIN(size, getCapacity() - tail);
        std::copy_n(buf, stage_a, buffer.get() + tail);

        if (stage_a < size) {
            std::copy_n(buf + stage_a, size - stage_a, buffer.get());
        }

        tail = (tail + size) % getCapacity();
        occupancy += size;
    }

    semaphore.notify_all();
    return Result::SUCCESS;
}

template<class T>
const Result CircularBuffer<T>::reset() {
    {
        const std::lock_guard<std::mutex> lock(io_mtx);
        this->head = 0;
        this->tail = 0;
        this->occupancy = 0;
        this->transfers = 0;
        this->throughput = 0;
        this->overflows = 0;
    }

    semaphore.notify_all();
    return Result::SUCCESS;
}

template<class T>
const U64 CircularBuffer<T>::getCapacity() const {
    return this->capacity;
}

template<class T>
const U64 CircularBuffer<T>::getOccupancy() const {
    return this->occupancy;
}

template<class T>
const bool CircularBuffer<T>::isEmpty() const {
    return getOccupancy() == 0;
}

template<class T>
const bool CircularBuffer<T>::isFull() const {
    return getOccupancy() == getCapacity();
}

template class CircularBuffer<I8>;
template class CircularBuffer<CI8>;
template class CircularBuffer<F32>;
template class CircularBuffer<CF32>;
template class CircularBuffer<F64>;
template class CircularBuffer<CF64>;

}  // namespace Jetstream::Memory
