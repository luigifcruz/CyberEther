#ifndef JETSTREAM_INSTANCE_REMOTE_HH
#define JETSTREAM_INSTANCE_REMOTE_HH

#include "jetstream/instance.hh"

#include <array>
#include <ostream>
#include <string>
#include <vector>

namespace Jetstream {

struct Instance::Remote {
 public:
    enum class CodecType : uint32_t {
        H264,
        AV1,
        VP8,
        VP9,
    };

    enum class EncoderType {
        Auto,
        Software,
        NVENC,
        V4L2,
    };

    struct Config {
        std::string broker = "https://cyberether.org";
        bool autoJoinSessions = false;
        U32 framerate = 30;
        EncoderType encoder = EncoderType::Auto;
        CodecType codec = CodecType::H264;
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

std::string GetRemoteCodecName(const Instance::Remote::CodecType& codec);
Instance::Remote::CodecType StringToRemoteCodec(const std::string& codec);
const char* GetRemoteCodecPrettyName(const Instance::Remote::CodecType& codec);

std::string GetRemoteEncoderName(const Instance::Remote::EncoderType& encoder);
Instance::Remote::EncoderType StringToRemoteEncoder(const std::string& encoder);
const char* GetRemoteEncoderPrettyName(const Instance::Remote::EncoderType& encoder);

static inline constexpr std::array<Instance::Remote::CodecType, 4> RemoteCodecTypes = {
    Instance::Remote::CodecType::H264,
    Instance::Remote::CodecType::AV1,
    Instance::Remote::CodecType::VP8,
    Instance::Remote::CodecType::VP9,
};

static inline constexpr std::array<Instance::Remote::EncoderType, 4> RemoteEncoderTypes = {
    Instance::Remote::EncoderType::Auto,
    Instance::Remote::EncoderType::Software,
    Instance::Remote::EncoderType::NVENC,
    Instance::Remote::EncoderType::V4L2,
};

inline std::ostream& operator<<(std::ostream& os, const Instance::Remote::EncoderType& encoder) {
    return os << GetRemoteEncoderPrettyName(encoder);
}

inline std::ostream& operator<<(std::ostream& os, const Instance::Remote::CodecType& codec) {
    return os << GetRemoteCodecPrettyName(codec);
}

}  // namespace Jetstream

template <> struct jst::fmt::formatter<Jetstream::Instance::Remote::CodecType> : ostream_formatter {};
template <> struct jst::fmt::formatter<Jetstream::Instance::Remote::EncoderType> : ostream_formatter {};

#endif  // JETSTREAM_INSTANCE_REMOTE_HH
