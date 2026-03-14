#ifndef JETSTREAM_VIEWPORT_TYPES_HH
#define JETSTREAM_VIEWPORT_TYPES_HH

#include "jetstream/types.hh"

namespace Jetstream::Viewport {

enum class VideoCodec : uint32_t {
    H264,
    AV1,
    VP8,
    VP9,
};

inline std::string VideoCodecToString(const VideoCodec& codec) {
    switch (codec) {
        case VideoCodec::AV1:
            return "av1";
        case VideoCodec::VP8:
            return "vp8";
        case VideoCodec::VP9:
            return "vp9";
        case VideoCodec::H264:
            return "h264";
        default:
            JST_ERROR("Unknown video codec.");
            throw Result::ERROR;
    }
}

inline VideoCodec StringToVideoCodec(const std::string& codec) {
    std::string sanitizedCodec = codec;
    std::transform(sanitizedCodec.begin(), sanitizedCodec.end(), sanitizedCodec.begin(), ::tolower);

    if (sanitizedCodec == "av1") {
        return VideoCodec::AV1;
    } else if (sanitizedCodec == "vp8") {
        return VideoCodec::VP8;
    } else if (sanitizedCodec == "vp9") {
        return VideoCodec::VP9;
    } else if (sanitizedCodec == "h264") {
        return VideoCodec::H264;
    }

    JST_ERROR("Unknown video codec: {}", codec);
    throw Result::ERROR;
}

}  // namespace Jetstream::Render

#endif
