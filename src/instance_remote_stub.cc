#include "jetstream/instance_remote.hh"
#include "jetstream/logger.hh"

#include <memory>

namespace Jetstream {

struct Instance::Remote::Impl {
    Viewport::Generic* viewport = nullptr;
    DeviceType viewportDevice = DeviceType::None;
    bool started_ = false;
    std::string roomId_;
    std::string consumerToken;
    std::string inviteUrl_;
    std::vector<std::string> waitlist_;
    std::vector<ClientInfo> clients_;

    bool supported() const {
        return false;
    }

    Result create(const Instance::Remote::Config&) {
        JST_ERROR("[REMOTE] Remote streaming is not available in this build.");
        return Result::ERROR;
    }

    Result destroy() {
        return Result::SUCCESS;
    }

    Result captureFrame() {
        return Result::SUCCESS;
    }

    Result updateWaitlist() {
        return Result::SUCCESS;
    }

    Result updateSessions() {
        return Result::SUCCESS;
    }

    Result approveClient(const std::string&) {
        return Result::ERROR;
    }
};

Instance::Remote::Remote(Viewport::Generic* viewport) {
    impl = std::make_shared<Impl>();
    impl->viewport = viewport;
    impl->viewportDevice = viewport ? viewport->device() : DeviceType::None;
}

Instance::Remote::~Remote() = default;

bool Instance::Remote::supported() const {
    return impl->supported();
}

Result Instance::Remote::create(const Config& config) {
    return impl->create(config);
}

Result Instance::Remote::destroy() {
    return impl->destroy();
}

bool Instance::Remote::started() const {
    return impl->started_;
}

Result Instance::Remote::captureFrame() {
    return impl->captureFrame();
}

const std::string& Instance::Remote::roomId() const {
    return impl->roomId_;
}

const std::string& Instance::Remote::accessToken() const {
    return impl->consumerToken;
}

const std::string& Instance::Remote::inviteUrl() const {
    return impl->inviteUrl_;
}

const std::vector<Instance::Remote::ClientInfo>& Instance::Remote::clients() const {
    return impl->clients_;
}

const std::vector<std::string>& Instance::Remote::waitlist() const {
    return impl->waitlist_;
}

Result Instance::Remote::updateWaitlist() {
    return impl->updateWaitlist();
}

Result Instance::Remote::updateSessions() {
    return impl->updateSessions();
}

Result Instance::Remote::approveClient(const std::string& code) {
    return impl->approveClient(code);
}

}  // namespace Jetstream
