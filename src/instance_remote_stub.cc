#include "jetstream/instance_remote.hh"
#include "jetstream/logger.hh"
#include "jetstream/types.hh"

namespace Jetstream {

struct Instance::Remote::Impl {
    Viewport::Generic* viewport = nullptr;
};

Instance::Remote::Remote(Viewport::Generic* viewport) {
    impl = std::make_shared<Impl>();
    impl->viewport = viewport;
}

Instance::Remote::~Remote() {
    impl.reset();
}

bool Instance::Remote::supported() const {
    return false;
}

Result Instance::Remote::create(const Config&) {
    JST_ERROR("[REMOTE] Remote streaming is not available in this build.");
    return Result::ERROR;
}

Result Instance::Remote::destroy() {
    return Result::SUCCESS;
}

bool Instance::Remote::started() const {
    return false;
}

Result Instance::Remote::captureFrame() {
    return Result::SUCCESS;
}

const std::string& Instance::Remote::roomId() const {
    static std::string empty;
    return empty;
}

const std::string& Instance::Remote::accessToken() const {
    static std::string empty;
    return empty;
}

const std::string& Instance::Remote::inviteUrl() const {
    static std::string empty;
    return empty;
}

const std::vector<Instance::Remote::ClientInfo>& Instance::Remote::clients() const {
    static std::vector<ClientInfo> empty;
    return empty;
}

const std::vector<std::string>& Instance::Remote::waitlist() const {
    static std::vector<std::string> empty;
    return empty;
}

Result Instance::Remote::updateWaitlist() {
    return Result::SUCCESS;
}

Result Instance::Remote::updateSessions() {
    return Result::SUCCESS;
}

Result Instance::Remote::approveClient(const std::string&) {
    return Result::ERROR;
}

}  // namespace Jetstream
