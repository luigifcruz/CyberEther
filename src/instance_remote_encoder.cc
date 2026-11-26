#include "instance_remote_encoder.hh"

#include "jetstream/logger.hh"

#include <algorithm>
#include <cstdarg>
#include <mutex>
#include <string_view>

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
#include <cuda.h>
#endif

#ifdef JST_OS_MAC
#include <CoreVideo/CoreVideo.h>
#endif

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
#include <libavutil/hwcontext_cuda.h>
#endif
}

namespace Jetstream {

namespace {

constexpr std::chrono::seconds kReleaseTimeout{1};

std::string ErrorString(int error) {
    char buffer[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(error, buffer, sizeof(buffer));
    return buffer;
}

void SetOption(AVCodecContext* context, const char* key, const char* value) {
    if (av_opt_set(context->priv_data, key, value, 0) < 0) {
        JST_DEBUG("[REMOTE] Encoder option '{}' is not supported by '{}'.", key, context->codec->name);
    }
}

void RouteLogs() {
    static std::once_flag once;
    std::call_once(once, [] {
        av_log_set_callback([](void* object, int level, const char* format, va_list arguments) {
            if (level > AV_LOG_WARNING) {
                return;
            }

            char line[1024] = {};
            int prefix = 1;
            av_log_format_line2(object, level, format, arguments, line, sizeof(line), &prefix);

            std::string_view text(line);
            while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
                text.remove_suffix(1);
            }
            JST_DEBUG("[REMOTE] FFmpeg: {}", text);
        });
    });
}

AVPixelFormat InputFormat(const AVCodec* codec) {
    const void* formats = nullptr;
    int count = 0;
    if (avcodec_get_supported_config(nullptr, codec, AV_CODEC_CONFIG_PIX_FORMAT, 0, &formats, &count) < 0) {
        return AV_PIX_FMT_NONE;
    }
    if (!formats) {
        return AV_PIX_FMT_NV12;
    }

    const auto* list = static_cast<const AVPixelFormat*>(formats);
    for (const AVPixelFormat format : {AV_PIX_FMT_BGRA, AV_PIX_FMT_NV12, AV_PIX_FMT_YUV420P}) {
        if (std::find(list, list + count, format) != list + count) {
            return format;
        }
    }
    return AV_PIX_FMT_NONE;
}

U8 H264Level(const Extent2D<U64>& size, U32 framerate, U64 bitrate) {
    struct Level {
        U8 id;
        U64 frame;
        U64 rate;
        U64 kilobits;
    };

    static constexpr Level levels[] = {
        {0x1F, 3600, 108000, 14000},
        {0x20, 5120, 216000, 20000},
        {0x28, 8192, 245760, 20000},
        {0x29, 8192, 245760, 50000},
        {0x2A, 8704, 522240, 50000},
        {0x32, 22080, 589824, 135000},
        {0x33, 36864, 983040, 240000},
        {0x34, 36864, 2073600, 240000},
        {0x3C, 139264, 4177920, 240000},
        {0x3D, 139264, 8355840, 480000},
        {0x3E, 139264, 16711680, 800000},
    };

    const U64 frame = ((size.x + 15) / 16) * ((size.y + 15) / 16);
    for (const auto& level : levels) {
        if (frame <= level.frame && frame * framerate <= level.rate && bitrate <= level.kilobits * 1000) {
            return level.id;
        }
    }

    return levels[std::size(levels) - 1].id;
}

bool HasParameterSets(Instance::Remote::CodecType codec, const U8* data, int size) {
    for (int index = 0; index + 3 < size; ++index) {
        if (data[index] != 0x00 || data[index + 1] != 0x00 || data[index + 2] != 0x01) {
            continue;
        }

        const U8 header = data[index + 3];
        if (codec == Instance::Remote::CodecType::H264 && (header & 0x1F) == 7) {
            return true;
        }
        if (codec == Instance::Remote::CodecType::H265 && ((header >> 1) & 0x3F) == 32) {
            return true;
        }
    }

    return false;
}

}  // namespace

Instance::Remote::Encoder::~Encoder() {
    (void)destroy();
}

const char* Instance::Remote::Encoder::CodecName(CodecType codec, EncoderType encoder) {
    switch (encoder) {
        case EncoderType::Software:
            switch (codec) {
                case CodecType::H264: return "libopenh264";
                default:              return nullptr;
            }
        case EncoderType::NVENC:
            switch (codec) {
                case CodecType::H264: return "h264_nvenc";
                case CodecType::H265: return "hevc_nvenc";
                case CodecType::AV1:  return "av1_nvenc";
                default:              return nullptr;
            }
        case EncoderType::VideoToolbox:
            switch (codec) {
                case CodecType::H264: return "h264_videotoolbox";
                case CodecType::H265: return "hevc_videotoolbox";
                default:              return nullptr;
            }
        case EncoderType::V4L2:
            switch (codec) {
                case CodecType::H264: return "h264_v4l2m2m";
                case CodecType::H265: return "hevc_v4l2m2m";
                default:              return nullptr;
            }
        default:
            return nullptr;
    }
}

std::string Instance::Remote::Encoder::Parameters(const Config& config) {
    switch (config.codec) {
        case CodecType::H264:
            return jst::fmt::format("level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e0{:02x}",
                                    H264Level(config.size, config.framerate, config.bitrate));
        case CodecType::H265:
            return "level-id=153;profile-id=1;tier-flag=0;tx-mode=SRST";
        default:
            return {};
    }
}

bool Instance::Remote::Encoder::Available(const Config& config) {
    const char* name = CodecName(config.codec, config.encoder);
    if (!name) {
        return false;
    }

    RouteLogs();

    const AVCodec* avcodec = avcodec_find_encoder_by_name(name);
    if (!avcodec) {
        return false;
    }

    Encoder probe;
    probe.config = config;
    probe.config.size = {640, 480};
    probe.context = avcodec_alloc_context3(avcodec);

    return probe.context &&
           Configure(probe.context, probe.config) == Result::SUCCESS &&
           probe.createDevice() == Result::SUCCESS &&
           avcodec_open2(probe.context, avcodec, nullptr) == 0;
}

Result Instance::Remote::Encoder::Configure(AVCodecContext* context, const Config& config) {
    const AVCodec* codec = context->codec;
    if (!codec) {
        return Result::ERROR;
    }

    context->width = static_cast<int>(config.size.x);
    context->height = static_cast<int>(config.size.y);
    context->time_base = {1, 1'000'000};
    context->framerate = {static_cast<int>(config.framerate), 1};
    context->bit_rate = static_cast<int64_t>(config.bitrate);
    context->rc_max_rate = static_cast<int64_t>(config.bitrate);
    context->gop_size = static_cast<int>(config.framerate);
    context->max_b_frames = 0;
    context->color_range = AVCOL_RANGE_MPEG;
    context->colorspace = AVCOL_SPC_BT709;
    context->color_primaries = AVCOL_PRI_BT709;
    context->color_trc = AVCOL_TRC_IEC61966_2_1;

    if (config.device == DeviceType::CUDA) {
        context->pix_fmt = AV_PIX_FMT_CUDA;
    } else if (config.encoder == EncoderType::VideoToolbox) {
        context->pix_fmt = AV_PIX_FMT_VIDEOTOOLBOX;
        context->sw_pix_fmt = AV_PIX_FMT_BGRA;
    } else {
        context->pix_fmt = InputFormat(codec);
    }
    if (context->pix_fmt == AV_PIX_FMT_NONE) {
        JST_ERROR("[REMOTE] Encoder '{}' accepts no supported pixel format.", codec->name);
        return Result::ERROR;
    }

    switch (config.encoder) {
        case EncoderType::Software:
            context->rc_buffer_size = static_cast<int>(config.bitrate);
            SetOption(context, "rc_mode", "bitrate");
            SetOption(context, "allow_skip_frames", "0");
            break;
        case EncoderType::NVENC:
            if (config.codec == CodecType::H264) {
                SetOption(context, "profile", "baseline");
            }
            SetOption(context, "preset", "p2");
            SetOption(context, "tune", "ull");
            SetOption(context, "rc", "cbr");
            SetOption(context, "zerolatency", "1");
            SetOption(context, "delay", "0");
            SetOption(context, "forced-idr", "1");
            break;
        case EncoderType::VideoToolbox:
            context->profile = config.codec == CodecType::H264 ? AV_PROFILE_H264_CONSTRAINED_BASELINE : context->profile;
            SetOption(context, "realtime", "1");
            SetOption(context, "prio_speed", "1");
            SetOption(context, "allow_sw", "0");
            break;
        case EncoderType::V4L2:
            if (config.codec == CodecType::H264) {
                context->profile = AV_PROFILE_H264_CONSTRAINED_BASELINE;
            }
            break;
        default:
            JST_ERROR("[REMOTE] Unsupported encoder '{}'.", config.encoder);
            return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result Instance::Remote::Encoder::create(const Config& config) {
    JST_CHECK(destroy());

    RouteLogs();
    this->config = config;

    const char* name = CodecName(config.codec, config.encoder);
    const AVCodec* codec = name ? avcodec_find_encoder_by_name(name) : nullptr;
    if (!codec) {
        JST_ERROR("[REMOTE] Encoder '{}' does not support codec '{}'.", config.encoder, config.codec);
        return Result::ERROR;
    }

    context = avcodec_alloc_context3(codec);
    packet = av_packet_alloc();
    input = av_frame_alloc();
    if (!context || !packet || !input) {
        JST_ERROR("[REMOTE] Failed to allocate encoder state.");
        (void)destroy();
        return Result::ERROR;
    }

    if (Configure(context, config) != Result::SUCCESS ||
        createDevice() != Result::SUCCESS ||
        createConverter() != Result::SUCCESS) {
        (void)destroy();
        return Result::ERROR;
    }

    if (const int error = avcodec_open2(context, codec, nullptr); error < 0) {
        JST_ERROR("[REMOTE] Failed to open encoder '{}': {}", codec->name, ErrorString(error));
        (void)destroy();
        return Result::ERROR;
    }

    headers.assign(context->extradata, context->extradata + context->extradata_size);

    JST_DEBUG("[REMOTE] Opened encoder '{}' ({}x{}, {}).", codec->name,
              config.size.x, config.size.y, av_get_pix_fmt_name(context->pix_fmt));
    return Result::SUCCESS;
}

Result Instance::Remote::Encoder::createDevice() {
    if (config.device != DeviceType::CUDA) {
        return Result::SUCCESS;
    }

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
    device = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_CUDA);
    if (!device) {
        JST_ERROR("[REMOTE] Failed to allocate the CUDA encoder device.");
        return Result::ERROR;
    }

    auto* deviceContext = reinterpret_cast<AVHWDeviceContext*>(device->data);
    auto* cudaContext = static_cast<AVCUDADeviceContext*>(deviceContext->hwctx);
    cudaContext->cuda_ctx = static_cast<CUcontext>(config.context);

    if (const int error = av_hwdevice_ctx_init(device); error < 0) {
        JST_ERROR("[REMOTE] Failed to initialize the CUDA encoder device: {}", ErrorString(error));
        return Result::ERROR;
    }

    frames = av_hwframe_ctx_alloc(device);
    if (!frames) {
        JST_ERROR("[REMOTE] Failed to allocate the CUDA encoder frames.");
        return Result::ERROR;
    }

    auto* framesContext = reinterpret_cast<AVHWFramesContext*>(frames->data);
    framesContext->format = AV_PIX_FMT_CUDA;
    framesContext->sw_format = AV_PIX_FMT_BGRA;
    framesContext->width = static_cast<int>(config.size.x);
    framesContext->height = static_cast<int>(config.size.y);

    if (const int error = av_hwframe_ctx_init(frames); error < 0) {
        JST_ERROR("[REMOTE] Failed to initialize the CUDA encoder frames: {}", ErrorString(error));
        return Result::ERROR;
    }

    context->hw_frames_ctx = av_buffer_ref(frames);
    return Result::SUCCESS;
#else
    JST_ERROR("[REMOTE] CUDA encoder input is not available in this build.");
    return Result::ERROR;
#endif
}

Result Instance::Remote::Encoder::createConverter() {
    if (context->pix_fmt == AV_PIX_FMT_BGRA ||
        context->pix_fmt == AV_PIX_FMT_CUDA ||
        context->pix_fmt == AV_PIX_FMT_VIDEOTOOLBOX) {
        return Result::SUCCESS;
    }

    converter = sws_getContext(context->width, context->height, AV_PIX_FMT_BGRA,
                               context->width, context->height, context->pix_fmt,
                               SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
    if (!converter) {
        JST_ERROR("[REMOTE] Failed to create the encoder color converter.");
        return Result::ERROR;
    }

    const int* coefficients = sws_getCoefficients(SWS_CS_ITU709);
    sws_setColorspaceDetails(converter, coefficients, 1, coefficients, 0, 0, 1 << 16, 1 << 16);

    converted = av_frame_alloc();
    if (!converted) {
        JST_ERROR("[REMOTE] Failed to allocate the converted encoder frame.");
        return Result::ERROR;
    }

    return allocateConverted();
}

Result Instance::Remote::Encoder::allocateConverted() {
    av_frame_unref(converted);
    converted->format = context->pix_fmt;
    converted->width = context->width;
    converted->height = context->height;

    if (const int error = av_image_fill_linesizes(converted->linesize, context->pix_fmt, context->width); error < 0) {
        JST_ERROR("[REMOTE] Failed to size the converted encoder frame: {}", ErrorString(error));
        return Result::ERROR;
    }

    const AVPixFmtDescriptor* descriptor = av_pix_fmt_desc_get(context->pix_fmt);
    for (int plane = 0; plane < av_pix_fmt_count_planes(context->pix_fmt); ++plane) {
        const int height = plane == 0 ? context->height : AV_CEIL_RSHIFT(context->height, descriptor->log2_chroma_h);
        converted->buf[plane] = av_buffer_alloc(static_cast<size_t>(converted->linesize[plane]) * height);
        if (!converted->buf[plane]) {
            av_frame_unref(converted);
            JST_ERROR("[REMOTE] Failed to allocate the converted encoder frame.");
            return Result::ERROR;
        }
        converted->data[plane] = converted->buf[plane]->data;
    }

    return Result::SUCCESS;
}

Result Instance::Remote::Encoder::destroy() {
    (void)waitForRelease();
    headers.clear();

    if (converter) {
        sws_freeContext(converter);
        converter = nullptr;
    }
    av_frame_free(&converted);
    av_frame_free(&input);
    av_packet_free(&packet);
    avcodec_free_context(&context);
    av_buffer_unref(&frames);
    av_buffer_unref(&device);

    return Result::SUCCESS;
}

Result Instance::Remote::Encoder::encode(const void* data,
                                         std::chrono::microseconds timestamp,
                                         bool keyframe,
                                         std::vector<Packet>& packets) {
    if (!context) {
        JST_ERROR("[REMOTE] Encoder is not open.");
        return Result::ERROR;
    }

    av_frame_unref(input);
    input->width = context->width;
    input->height = context->height;
    input->pts = timestamp.count();
    input->pict_type = keyframe ? AV_PICTURE_TYPE_I : AV_PICTURE_TYPE_NONE;
    input->color_range = context->color_range;
    input->colorspace = context->colorspace;
    input->color_primaries = context->color_primaries;
    input->color_trc = context->color_trc;

    const int stride = context->width * 4;

    if (context->pix_fmt == AV_PIX_FMT_CUDA) {
        {
            std::lock_guard<std::mutex> lock(releaseMutex);
            released = false;
        }

        input->format = AV_PIX_FMT_CUDA;
        input->data[0] = static_cast<U8*>(const_cast<void*>(data));
        input->linesize[0] = stride;
        input->hw_frames_ctx = av_buffer_ref(frames);
        input->buf[0] = av_buffer_create(input->data[0],
                                         static_cast<size_t>(stride) * context->height,
                                         [](void* opaque, uint8_t*) {
                                             auto* encoder = static_cast<Encoder*>(opaque);
                                             std::lock_guard<std::mutex> lock(encoder->releaseMutex);
                                             encoder->released = true;
                                             encoder->releaseCondition.notify_all();
                                         },
                                         this,
                                         AV_BUFFER_FLAG_READONLY);
        if (!input->buf[0] || !input->hw_frames_ctx) {
            av_frame_unref(input);
            std::lock_guard<std::mutex> lock(releaseMutex);
            released = true;
            JST_ERROR("[REMOTE] Failed to wrap the CUDA encoder frame.");
            return Result::ERROR;
        }

        return submitBorrowed(packets);
    }

#ifdef JST_OS_MAC
    if (context->pix_fmt == AV_PIX_FMT_VIDEOTOOLBOX) {
        {
            std::lock_guard<std::mutex> lock(releaseMutex);
            released = false;
        }

        CVPixelBufferRef pixels = nullptr;
        const CVReturn status = CVPixelBufferCreateWithBytes(kCFAllocatorDefault,
                                                             static_cast<size_t>(context->width),
                                                             static_cast<size_t>(context->height),
                                                             kCVPixelFormatType_32BGRA,
                                                             const_cast<void*>(data),
                                                             static_cast<size_t>(stride),
                                                             [](void* opaque, const void*) {
                                                                 auto* encoder = static_cast<Encoder*>(opaque);
                                                                 std::lock_guard<std::mutex> lock(encoder->releaseMutex);
                                                                 encoder->released = true;
                                                                 encoder->releaseCondition.notify_all();
                                                             },
                                                             this,
                                                             nullptr,
                                                             &pixels);
        if (status != kCVReturnSuccess || !pixels) {
            std::lock_guard<std::mutex> lock(releaseMutex);
            released = true;
            JST_ERROR("[REMOTE] Failed to wrap the VideoToolbox encoder frame ({}).", static_cast<int>(status));
            return Result::ERROR;
        }

        input->format = AV_PIX_FMT_VIDEOTOOLBOX;
        input->data[3] = reinterpret_cast<U8*>(pixels);
        input->buf[0] = av_buffer_create(reinterpret_cast<U8*>(pixels), 0, [](void*, uint8_t* buffer) {
            CFRelease(reinterpret_cast<CVPixelBufferRef>(buffer));
        }, nullptr, 0);
        if (!input->buf[0]) {
            av_frame_unref(input);
            CFRelease(pixels);
            JST_ERROR("[REMOTE] Failed to wrap the VideoToolbox encoder frame.");
            (void)waitForRelease();
            return Result::ERROR;
        }

        return submitBorrowed(packets);
    }
#endif

    input->format = AV_PIX_FMT_BGRA;
    input->data[0] = static_cast<U8*>(const_cast<void*>(data));
    input->linesize[0] = stride;

    if (!converter) {
        const Result result = submit(input);
        input->data[0] = nullptr;
        JST_CHECK(result);
        return receive(packets);
    }

    bool writable = converted->buf[0] != nullptr;
    for (int plane = 0; plane < AV_NUM_DATA_POINTERS && converted->buf[plane]; ++plane) {
        writable = writable && av_buffer_is_writable(converted->buf[plane]);
    }
    if (!writable) {
        JST_CHECK(allocateConverted());
    }

    const U8* sources[4] = {input->data[0]};
    const int strides[4] = {stride};
    const int scaled = sws_scale(converter, sources, strides, 0, context->height, converted->data, converted->linesize);
    input->data[0] = nullptr;
    if (scaled <= 0) {
        JST_ERROR("[REMOTE] Failed to convert the encoder frame.");
        return Result::ERROR;
    }

    converted->pts = input->pts;
    converted->pict_type = input->pict_type;
    converted->color_range = input->color_range;
    converted->colorspace = input->colorspace;
    converted->color_primaries = input->color_primaries;
    converted->color_trc = input->color_trc;

    JST_CHECK(submit(converted));
    return receive(packets);
}

std::string Instance::Remote::Encoder::format() const {
    if (!context) {
        return {};
    }
    const char* name = av_get_pix_fmt_name(context->pix_fmt);
    return name ? name : "";
}

Result Instance::Remote::Encoder::submitBorrowed(std::vector<Packet>& packets) {
    const Result submitted = submit(input);
    av_frame_unref(input);
    const Result received = submitted == Result::SUCCESS ? receive(packets) : Result::ERROR;

    if (!waitForRelease()) {
        JST_ERROR("[REMOTE] Encoder did not release the input frame. Reopening it.");
        avcodec_free_context(&context);
        (void)waitForRelease();
        (void)create(Config(config));
        return Result::ERROR;
    }

    return received;
}

Result Instance::Remote::Encoder::submit(AVFrame* frame) {
    if (const int error = avcodec_send_frame(context, frame); error < 0) {
        JST_ERROR("[REMOTE] Failed to submit frame to encoder: {}", ErrorString(error));
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result Instance::Remote::Encoder::receive(std::vector<Packet>& packets) {
    while (true) {
        const int error = avcodec_receive_packet(context, packet);
        if (error == AVERROR(EAGAIN) || error == AVERROR_EOF) {
            break;
        }
        if (error < 0) {
            JST_ERROR("[REMOTE] Failed to receive packet from encoder: {}", ErrorString(error));
            return Result::ERROR;
        }

        Packet& output = packets.emplace_back();
        output.timestamp = std::chrono::microseconds(packet->pts);
        output.keyframe = (packet->flags & AV_PKT_FLAG_KEY) != 0;
        if (config.codec == CodecType::H264 || config.codec == CodecType::H265) {
            const bool parameterSets = HasParameterSets(config.codec, packet->data, packet->size);
            if (output.keyframe && !parameterSets) {
                output.data.assign(headers.begin(), headers.end());
            } else if (!output.keyframe && parameterSets) {
                headers.assign(packet->data, packet->data + packet->size);
            }
        }
        output.data.insert(output.data.end(), packet->data, packet->data + packet->size);
        av_packet_unref(packet);
    }

    return Result::SUCCESS;
}

bool Instance::Remote::Encoder::waitForRelease() {
    std::unique_lock<std::mutex> lock(releaseMutex);
    return releaseCondition.wait_for(lock, kReleaseTimeout, [this] { return released; });
}

}  // namespace Jetstream
