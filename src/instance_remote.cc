#include "jetstream/instance_remote.hh"
#include "jetstream/logger.hh"
#include "jetstream/types.hh"

#include <algorithm>
#include <cctype>

namespace Jetstream {

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
        default:
            return "Unknown";
    }
}

}  // namespace Jetstream
