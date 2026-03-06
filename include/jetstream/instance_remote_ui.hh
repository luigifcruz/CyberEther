#pragma once

#include <atomic>
#include <functional>
#include <set>
#include <string>
#include <thread>

#include "jetstream/instance_remote.hh"

void PrintRemoteInfo(Jetstream::Instance::Remote* remote);
bool PromptRemoteClientApproval(const std::string& code);

class RemoteSessionMonitor {
 public:
    using ApprovalPrompt = std::function<bool(const std::string& code)>;

    RemoteSessionMonitor(Jetstream::Instance::Remote* remote,
                         bool autoJoin,
                         ApprovalPrompt approvalPrompt);
    ~RemoteSessionMonitor();

    void start();
    void stop();

 private:
    void run();

    Jetstream::Instance::Remote* remote_ = nullptr;
    bool autoJoin_ = false;
    ApprovalPrompt approvalPrompt_;

    std::set<std::string> seenSessions_;
    std::atomic<bool> running_{false};
    std::thread worker_;
};
