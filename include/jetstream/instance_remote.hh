#ifndef JETSTREAM_INSTANCE_REMOTE_HH
#define JETSTREAM_INSTANCE_REMOTE_HH

#include "jetstream/instance.hh"
#include "jetstream/viewport/types.hh"

#include <string>
#include <vector>

namespace Jetstream {

struct Instance::Remote {
 public:
    struct Config {
        std::string broker = "127.0.0.1:8080";
        Viewport::VideoCodec codec = Viewport::VideoCodec::H264;
        bool autoJoinSessions = false;
        bool hardwareAcceleration = true;
        U32 framerate = 30;
    };

    struct ClientInfo {
        std::string sessionId;
    };

    Remote(Viewport::Generic* viewport);
    ~Remote();

    bool supported() const;

    Result create(const Config& config);
    Result destroy();

    bool started() const;
    Result captureFrame();

    const std::string& roomId() const;
    const std::string& accessToken() const;
    const std::string& inviteUrl() const;
    const std::vector<ClientInfo>& clients() const;
    const std::vector<std::string>& waitlist() const;

    Result updateWaitlist();
    Result updateSessions();
    Result approveClient(const std::string& code);

 private:
    struct Impl;
    std::shared_ptr<Impl> impl;
};

}  // namespace Jetstream

#endif  // JETSTREAM_INSTANCE_REMOTE_HH
