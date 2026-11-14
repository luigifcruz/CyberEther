#include "jetstream/instance_remote.hh"
#include "jetstream/logger.hh"

#include <memory>

namespace Jetstream {

struct Instance::Remote::Impl {
    bool started_ = false;
    std::string roomId_;
    std::string consumerToken;
    std::string inviteUrl_;
    std::vector<std::string> waitlist_;
    std::vector<ClientInfo> clients_;

    bool supported() const {
        return false;
    }

    std::vector<EncoderType> available(CodecType) {
        return {};
    }

    Result create(const Instance::Remote::Config&) {
        JST_ERROR("[REMOTE] Remote streaming is not available in this build.");
        return Result::ERROR;
    }

    Result destroy() {
        return Result::SUCCESS;
    }

    Result captureFrame(bool) {
        return Result::SUCCESS;
    }

    Result approveClient(const std::string&) {
        return Result::ERROR;
    }

    Result removeClient(const std::string&) {
        return Result::ERROR;
    }
};

Instance::Remote::Remote(Viewport::Generic*) {
    impl = std::make_shared<Impl>();
}

Instance::Remote::~Remote() = default;

bool Instance::Remote::supported() const {
    return impl->supported();
}

std::vector<Instance::Remote::EncoderType> Instance::Remote::available(CodecType codec) {
    return impl->available(codec);
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

Result Instance::Remote::captureFrame(bool presented) {
    return impl->captureFrame(presented);
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

std::vector<Instance::Remote::ClientInfo> Instance::Remote::clients() const {
    return impl->clients_;
}

std::vector<std::string> Instance::Remote::waitlist() const {
    return impl->waitlist_;
}

Result Instance::Remote::approveClient(const std::string& sessionId) {
    return impl->approveClient(sessionId);
}

Result Instance::Remote::removeClient(const std::string& sessionId) {
    return impl->removeClient(sessionId);
}

}  // namespace Jetstream
