#ifndef JETSTREAM_INSTANCE_REMOTE_HH
#define JETSTREAM_INSTANCE_REMOTE_HH

#include "jetstream/instance.hh"

#include <array>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace Jetstream {

struct JETSTREAM_API Instance::Remote {
 public:
    struct Impl;
    struct Supervisor;

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
        VideoToolbox,
        MediaFoundation,
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
    std::vector<EncoderType> available(CodecType codec);

    Result create(const Config& config);
    Result destroy();

    bool started() const;
    Result captureFrame();

    const std::string& roomId() const;
    const std::string& accessToken() const;
    const std::string& inviteUrl() const;
    std::vector<ClientInfo> clients() const;
    std::vector<std::string> waitlist() const;

    Result approveClient(const std::string& code);

 private:
    std::shared_ptr<Impl> impl;
};

JETSTREAM_API std::string GetRemoteCodecName(const Instance::Remote::CodecType& codec);
JETSTREAM_API Instance::Remote::CodecType StringToRemoteCodec(const std::string& codec);
JETSTREAM_API const char* GetRemoteCodecPrettyName(const Instance::Remote::CodecType& codec);

JETSTREAM_API std::string GetRemoteEncoderName(const Instance::Remote::EncoderType& encoder);
JETSTREAM_API Instance::Remote::EncoderType StringToRemoteEncoder(const std::string& encoder);
JETSTREAM_API const char* GetRemoteEncoderPrettyName(const Instance::Remote::EncoderType& encoder);

static inline constexpr std::array<Instance::Remote::CodecType, 4> RemoteCodecTypes = {
    Instance::Remote::CodecType::H264,
    Instance::Remote::CodecType::AV1,
    Instance::Remote::CodecType::VP8,
    Instance::Remote::CodecType::VP9,
};

static inline constexpr std::array<Instance::Remote::EncoderType, 6> RemoteEncoderTypes = {
    Instance::Remote::EncoderType::Auto,
    Instance::Remote::EncoderType::Software,
    Instance::Remote::EncoderType::NVENC,
    Instance::Remote::EncoderType::V4L2,
    Instance::Remote::EncoderType::VideoToolbox,
    Instance::Remote::EncoderType::MediaFoundation,
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
