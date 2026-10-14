#ifndef JETSTREAM_DOMAINS_IO_SDR_TOOLS_STREAM_ACCESS_HH
#define JETSTREAM_DOMAINS_IO_SDR_TOOLS_STREAM_ACCESS_HH

#include <atomic>
#include <condition_variable>
#include <mutex>

namespace Jetstream::Radio {

class StreamAccess {
 public:
    std::unique_lock<std::mutex> lock() {
        std::unique_lock lock(mutex);
        ready.wait(lock, [this] { return !interrupted(); });
        return lock;
    }

    template<typename Change>
    auto change(Change&& change) {
        ++changes;
        std::unique_lock lock(mutex, std::defer_lock);
        try {
            lock.lock();
        } catch (...) {
            --changes;
            ready.notify_all();
            throw;
        }
        struct Completion {
            StreamAccess& access;
            ~Completion() {
                --access.changes;
                access.ready.notify_all();
            }
        } completion{*this};
        return change();
    }

    bool interrupted() const {
        return changes.load() != 0;
    }

 private:
    std::mutex mutex;
    std::condition_variable ready;
    std::atomic<unsigned> changes{0};
};

}

#endif
