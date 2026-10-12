#ifndef JETSTREAM_DOMAINS_IO_SDR_DEVICES_HACKRF_TRANSFER_QUEUE_HH
#define JETSTREAM_DOMAINS_IO_SDR_DEVICES_HACKRF_TRANSFER_QUEUE_HH

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <vector>

#include "jetstream/macros.hh"
#include "jetstream/types.hh"

namespace Jetstream::Radio {

class JETSTREAM_API HackRfTransferQueue {
 public:
    struct Transfer {
        std::vector<U8> bytes;
        bool gapBefore = false;
    };

    explicit HackRfTransferQueue(U64 depth);

    void push(const U8* data, U64 size);
    bool pop(Transfer& transfer, std::chrono::milliseconds timeout);
    void clear();
    U64 dropped() const;

 private:
    mutable std::mutex mutex;
    std::condition_variable available;
    std::deque<Transfer> transfers;
    U64 depth = 0;
    U64 droppedTransfers = 0;
    bool pendingGap = false;
};

}  // namespace Jetstream::Radio

#endif  // JETSTREAM_DOMAINS_IO_SDR_DEVICES_HACKRF_TRANSFER_QUEUE_HH
