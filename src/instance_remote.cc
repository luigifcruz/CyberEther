#include "jetstream/instance_remote.hh"
#include "jetstream/logger.hh"
#include "jetstream/types.hh"

#include <algorithm>
#include <cctype>

namespace Jetstream {

bool IsRemoteBrokerSchemeSupported(const std::string& broker) {
    return broker.starts_with("https://") || broker.starts_with("http://");
}

std::string GetRemoteCodecName(const Instance::Remote::CodecType& codec) {
    switch (codec) {
        case Instance::Remote::CodecType::AV1:
            return "av1";
        case Instance::Remote::CodecType::VP8:
            return "vp8";
        case Instance::Remote::CodecType::VP9:
            return "vp9";
        case Instance::Remote::CodecType::H264:
            return "h264";
        default:
            JST_ERROR("Unknown remote codec.");
            throw Result::ERROR;
    }
}

Instance::Remote::CodecType StringToRemoteCodec(const std::string& codec) {
    std::string sanitizedCodec = codec;
    std::transform(sanitizedCodec.begin(), sanitizedCodec.end(), sanitizedCodec.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    if (sanitizedCodec == "av1") {
        return Instance::Remote::CodecType::AV1;
    } else if (sanitizedCodec == "vp8") {
        return Instance::Remote::CodecType::VP8;
    } else if (sanitizedCodec == "vp9") {
        return Instance::Remote::CodecType::VP9;
    } else if (sanitizedCodec == "h264") {
        return Instance::Remote::CodecType::H264;
    }

    JST_ERROR("Unknown remote codec: {}", codec);
    throw Result::ERROR;
}

const char* GetRemoteCodecPrettyName(const Instance::Remote::CodecType& codec) {
    switch (codec) {
        case Instance::Remote::CodecType::AV1:
            return "AV1";
        case Instance::Remote::CodecType::VP8:
            return "VP8";
        case Instance::Remote::CodecType::VP9:
            return "VP9";
        case Instance::Remote::CodecType::H264:
            return "H264";
        default:
            return "Unknown";
    }
}

std::string GetRemoteEncoderName(const Instance::Remote::EncoderType& encoder) {
    switch (encoder) {
        case Instance::Remote::EncoderType::Auto:
            return "auto";
        case Instance::Remote::EncoderType::Software:
            return "software";
        case Instance::Remote::EncoderType::NVENC:
            return "nvenc";
        case Instance::Remote::EncoderType::V4L2:
            return "v4l2";
        case Instance::Remote::EncoderType::VideoToolbox:
            return "videotoolbox";
        case Instance::Remote::EncoderType::MediaFoundation:
            return "mediafoundation";
        default:
            JST_ERROR("Unknown remote encoder.");
            throw Result::ERROR;
    }
}

Instance::Remote::EncoderType StringToRemoteEncoder(const std::string& encoder) {
    std::string sanitizedEncoder = encoder;
    std::transform(sanitizedEncoder.begin(), sanitizedEncoder.end(), sanitizedEncoder.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    if (sanitizedEncoder == "auto") {
        return Instance::Remote::EncoderType::Auto;
    } else if (sanitizedEncoder == "software") {
        return Instance::Remote::EncoderType::Software;
    } else if (sanitizedEncoder == "nvenc") {
        return Instance::Remote::EncoderType::NVENC;
    } else if (sanitizedEncoder == "v4l2") {
        return Instance::Remote::EncoderType::V4L2;
    } else if (sanitizedEncoder == "videotoolbox") {
        return Instance::Remote::EncoderType::VideoToolbox;
    } else if (sanitizedEncoder == "mediafoundation") {
        return Instance::Remote::EncoderType::MediaFoundation;
    }

    JST_ERROR("Unknown remote encoder: {}", encoder);
    throw Result::ERROR;
}

const char* GetRemoteEncoderPrettyName(const Instance::Remote::EncoderType& encoder) {
    switch (encoder) {
        case Instance::Remote::EncoderType::Auto:
            return "Auto";
        case Instance::Remote::EncoderType::Software:
            return "Software";
        case Instance::Remote::EncoderType::NVENC:
            return "NVENC";
        case Instance::Remote::EncoderType::V4L2:
            return "V4L2";
        case Instance::Remote::EncoderType::VideoToolbox:
            return "VideoToolbox";
        case Instance::Remote::EncoderType::MediaFoundation:
            return "MediaFoundation";
        default:
            return "Unknown";
    }
}

std::string GetRemoteTransportName(const Instance::Remote::TransportType& transport) {
    switch (transport) {
        case Instance::Remote::TransportType::Auto:
            return "auto";
        case Instance::Remote::TransportType::UDP:
            return "udp";
        case Instance::Remote::TransportType::TCP:
            return "tcp";
        default:
            JST_ERROR("Unknown remote transport.");
            throw Result::ERROR;
    }
}

Instance::Remote::TransportType StringToRemoteTransport(const std::string& transport) {
    std::string sanitizedTransport = transport;
    std::transform(sanitizedTransport.begin(), sanitizedTransport.end(), sanitizedTransport.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    if (sanitizedTransport == "auto") {
        return Instance::Remote::TransportType::Auto;
    } else if (sanitizedTransport == "udp") {
        return Instance::Remote::TransportType::UDP;
    } else if (sanitizedTransport == "tcp") {
        return Instance::Remote::TransportType::TCP;
    }

    JST_ERROR("Unknown remote transport: {}", transport);
    throw Result::ERROR;
}

const char* GetRemoteTransportPrettyName(const Instance::Remote::TransportType& transport) {
    switch (transport) {
        case Instance::Remote::TransportType::Auto:
            return "Auto";
        case Instance::Remote::TransportType::UDP:
            return "UDP";
        case Instance::Remote::TransportType::TCP:
            return "TCP";
        default:
            return "Unknown";
    }
}

bool IsRemotePortRangeValid(U64 minPort, U64 maxPort) {
    if (minPort == 0 && maxPort == 0) {
        return true;
    }

    return minPort > 0 && minPort <= maxPort && maxPort <= 65535;
}

bool ParseRemotePortRange(const std::string& value, U16& minPort, U16& maxPort) {
    if (value.empty()) {
        minPort = 0;
        maxPort = 0;
        return true;
    }

    const auto separator = value.find('-');
    if (separator == std::string::npos) {
        return false;
    }

    const auto parsePort = [](const std::string& text, U64& port) {
        if (text.empty() || text.size() > 5 || !std::all_of(text.begin(), text.end(), [](unsigned char c) {
            return std::isdigit(c);
        })) {
            return false;
        }
        port = std::stoull(text);
        return true;
    };

    U64 min = 0;
    U64 max = 0;
    if (!parsePort(value.substr(0, separator), min) ||
        !parsePort(value.substr(separator + 1), max) ||
        min == 0 ||
        !IsRemotePortRangeValid(min, max)) {
        return false;
    }

    minPort = static_cast<U16>(min);
    maxPort = static_cast<U16>(max);
    return true;
}

std::string GetRemotePortRangeName(U64 minPort, U64 maxPort) {
    if (minPort == 0 && maxPort == 0) {
        return "";
    }

    return jst::fmt::format("{}-{}", minPort, maxPort);
}

}  // namespace Jetstream
