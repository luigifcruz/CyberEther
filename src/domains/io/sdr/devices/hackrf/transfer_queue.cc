#include "transfer_queue.hh"

namespace Jetstream::Radio {

HackRfTransferQueue::HackRfTransferQueue(const U64 queueDepth) : depth(queueDepth) {}

void HackRfTransferQueue::push(const U8* data, const U64 size) {
    {
        std::lock_guard lock(mutex);
        if (transfers.size() >= depth) {
            ++droppedTransfers;
            pendingGap = true;
            return;
        }
        transfers.push_back({.bytes = {data, data + size}, .gapBefore = pendingGap});
        pendingGap = false;
    }
    available.notify_all();
}

bool HackRfTransferQueue::pop(Transfer& transfer, const std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex);
    if (!available.wait_for(lock, timeout, [&] { return !transfers.empty(); })) {
        return false;
    }
    transfer = std::move(transfers.front());
    transfers.pop_front();
    return true;
}

void HackRfTransferQueue::clear() {
    std::lock_guard lock(mutex);
    transfers.clear();
    pendingGap = false;
}

U64 HackRfTransferQueue::dropped() const {
    std::lock_guard lock(mutex);
    return droppedTransfers;
}

}  // namespace Jetstream::Radio
