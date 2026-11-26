#ifndef JETSTREAM_INSTANCE_REMOTE_HH
#define JETSTREAM_INSTANCE_REMOTE_HH

#include "jetstream/instance.hh"

#include <array>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

namespace Jetstream {

struct JETSTREAM_API Instance::Remote {
 public:
    struct Impl;
    struct Supervisor;
    struct Broker;
    struct Encoder;

    enum class CodecType : uint32_t {
        H264,
        AV1,
        H265,
    };

    enum class EncoderType {
        Auto,
        Software,
        NVENC,
        V4L2,
        VideoToolbox,
        AMF,
        QSV,
        VAAPI,
    };

    enum class TransportType {
        Auto,
        UDP,
    };

    struct Config {
        struct Access {
            bool autoApprove = false;
        };

        struct Video {
            U32 framerate = 30;
            CodecType codec = CodecType::H264;
            EncoderType encoder = EncoderType::Auto;
        };

        struct Broker {
            std::string address = "0.0.0.0";
            U16 port = 8686;
            std::string publicUrl;
            std::string certificate;
            std::string key;
        };

        struct Media {
            TransportType transport = TransportType::Auto;
            U16 minPort = 0;
            U16 maxPort = 0;
            std::string address;
        };

        Access access;
        Video video;
        Broker broker;
        Media media;
    };

    struct ClientInfo {
        std::string sessionId;
    };

    struct Stats {
        EncoderType encoder = EncoderType::Auto;
        CodecType codec = CodecType::H264;
        DeviceType memory = DeviceType::None;
        std::string format;
        Extent2D<U64> size = {};
        F32 targetRate = 0.0f;
        F32 frameRate = 0.0f;
        F32 encodeAverage = 0.0f;
        F32 bitrate = 0.0f;
        U64 keyframeBytes = 0;
        std::optional<F32> roundTrip;
    };

    Remote(Viewport::Generic* viewport);
    ~Remote();

    bool supported() const;
    std::vector<EncoderType> available(CodecType codec);

    Result create(const Config& config);
    Result destroy();

    bool started() const;
    Result captureFrame(bool presented);

    const std::string& roomId() const;
    const std::string& accessToken() const;
    const std::string& inviteUrl() const;
    std::vector<ClientInfo> clients() const;
    std::vector<std::string> waitlist() const;
    Stats stats() const;

    Result approveClient(const std::string& sessionId);
    Result removeClient(const std::string& sessionId);

 private:
    std::shared_ptr<Impl> impl;
};

JETSTREAM_API std::string GetRemoteCodecName(const Instance::Remote::CodecType& codec);
JETSTREAM_API Instance::Remote::CodecType StringToRemoteCodec(const std::string& codec);
JETSTREAM_API const char* GetRemoteCodecPrettyName(const Instance::Remote::CodecType& codec);

JETSTREAM_API bool IsRemotePublicUrlValid(const std::string& url);

JETSTREAM_API std::string GetRemoteEncoderName(const Instance::Remote::EncoderType& encoder);
JETSTREAM_API Instance::Remote::EncoderType StringToRemoteEncoder(const std::string& encoder);
JETSTREAM_API const char* GetRemoteEncoderPrettyName(const Instance::Remote::EncoderType& encoder);

JETSTREAM_API std::string GetRemoteTransportName(const Instance::Remote::TransportType& transport);
JETSTREAM_API Instance::Remote::TransportType StringToRemoteTransport(const std::string& transport);
JETSTREAM_API const char* GetRemoteTransportPrettyName(const Instance::Remote::TransportType& transport);

JETSTREAM_API bool IsRemotePortRangeValid(U64 minPort, U64 maxPort);
JETSTREAM_API bool ParseRemotePortRange(const std::string& value, U16& minPort, U16& maxPort);
JETSTREAM_API std::string GetRemotePortRangeName(U64 minPort, U64 maxPort);

static inline constexpr std::array<Instance::Remote::CodecType, 3> RemoteCodecTypes = {
    Instance::Remote::CodecType::H264,
    Instance::Remote::CodecType::H265,
    Instance::Remote::CodecType::AV1,
};

static inline constexpr std::array<Instance::Remote::EncoderType, 8> RemoteEncoderTypes = {
    Instance::Remote::EncoderType::Auto,
    Instance::Remote::EncoderType::Software,
    Instance::Remote::EncoderType::NVENC,
    Instance::Remote::EncoderType::AMF,
    Instance::Remote::EncoderType::QSV,
    Instance::Remote::EncoderType::VAAPI,
    Instance::Remote::EncoderType::V4L2,
    Instance::Remote::EncoderType::VideoToolbox,
};

static inline constexpr std::array<Instance::Remote::TransportType, 2> RemoteTransportTypes = {
    Instance::Remote::TransportType::Auto,
    Instance::Remote::TransportType::UDP,
};

inline std::ostream& operator<<(std::ostream& os, const Instance::Remote::EncoderType& encoder) {
    return os << GetRemoteEncoderPrettyName(encoder);
}

inline std::ostream& operator<<(std::ostream& os, const Instance::Remote::CodecType& codec) {
    return os << GetRemoteCodecPrettyName(codec);
}

inline std::ostream& operator<<(std::ostream& os, const Instance::Remote::TransportType& transport) {
    return os << GetRemoteTransportPrettyName(transport);
}

}  // namespace Jetstream

template <> struct jst::fmt::formatter<Jetstream::Instance::Remote::CodecType> : ostream_formatter {};
template <> struct jst::fmt::formatter<Jetstream::Instance::Remote::EncoderType> : ostream_formatter {};
template <> struct jst::fmt::formatter<Jetstream::Instance::Remote::TransportType> : ostream_formatter {};

#endif  // JETSTREAM_INSTANCE_REMOTE_HH
