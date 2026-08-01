#ifndef JETSTREAM_TOOLS_CIRCULAR_BUFFER_HH
#define JETSTREAM_TOOLS_CIRCULAR_BUFFER_HH

#include <chrono>
#include <memory>

#include "jetstream/types.hh"

namespace Jetstream::Tools {

enum class CircularBufferOverflowPolicy {
    Reject,
    OverwriteOldest,
};

template<class T>
class CircularBuffer {
 public:
    CircularBuffer();
    explicit CircularBuffer(U64 capacity,
                            CircularBufferOverflowPolicy overflowPolicy =
                                CircularBufferOverflowPolicy::OverwriteOldest);
    ~CircularBuffer();

    CircularBuffer(const CircularBuffer&) = delete;
    CircularBuffer& operator=(const CircularBuffer&) = delete;
    CircularBuffer(CircularBuffer&& other);
    CircularBuffer& operator=(CircularBuffer&&) noexcept;

    Result push(const T* data, U64 size);
    Result pushStrided(const T* data, U64 size, U64 stride);
    Result pop(T* data, U64 size);
    Result popStrided(T* data, U64 size, U64 stride);
    Result peek(U64 offset, T* data, U64 size) const;
    Result discard(U64 size);
    Result clear();
    Result resize(U64 capacity);
    Result waitForSize(U64 size,
                       std::chrono::milliseconds timeout =
                           std::chrono::seconds(5));

    [[nodiscard]] bool empty() const;
    [[nodiscard]] bool full() const;
    [[nodiscard]] U64 capacity() const;
    [[nodiscard]] U64 size() const;
    [[nodiscard]] F64 throughput() const;
    [[nodiscard]] U64 overflows() const;

 private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

}  // namespace Jetstream::Tools

#endif  // JETSTREAM_TOOLS_CIRCULAR_BUFFER_HH
