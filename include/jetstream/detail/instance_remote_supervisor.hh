#pragma once

#include "jetstream/instance_remote.hh"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <set>
#include <string>
#include <thread>

namespace Jetstream {

struct JETSTREAM_API Instance::Remote::Supervisor {
    Supervisor(Instance::Remote* remote, bool autoJoin);
    ~Supervisor();

    void print() const;
    void start();
    void stop();

  private:
    bool prompt(const std::string& code) const;
    void run();

    Instance::Remote* remote_ = nullptr;
    bool autoJoin = false;
    std::set<std::string> seenSessions;
    std::atomic<bool> running_{false};
    std::mutex waitMutex;
    std::condition_variable waitCondition;
    std::thread worker_;
};

}  // namespace Jetstream
