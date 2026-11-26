#include "instance_remote_encoder.hh"

#include "jetstream/logger.hh"

#include <algorithm>
#include <cstdarg>
#include <mutex>
#include <string_view>

#ifdef JST_OS_MAC
#include <CoreVideo/CoreVideo.h>
#endif

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
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

Instance::Remote::Encoder::Export Instance::Remote::Encoder::ExportFor(EncoderType encoder) {
#if defined(JST_OS_LINUX)
    switch (encoder) {
        case EncoderType::NVENC: return Export::OpaqueFd;
        case EncoderType::VAAPI: return Export::DmaBuf;
        default:                 return Export::None;
    }
#elif defined(JST_OS_WINDOWS)
    switch (encoder) {
        case EncoderType::NVENC:
        case EncoderType::AMF:
        case EncoderType::QSV:   return Export::D3D11;
        default:                 return Export::None;
    }
#else
    (void)encoder;
    return Export::None;
#endif
}

#if !defined(JETSTREAM_LOADER_FFMPEG_CUDA_AVAILABLE)
std::unique_ptr<Instance::Remote::Encoder::Input> Instance::Remote::Encoder::CreateCudaInput() {
    return nullptr;
}
#endif

#if !defined(JST_OS_LINUX)
std::unique_ptr<Instance::Remote::Encoder::Input> Instance::Remote::Encoder::CreateVaapiInput() {
    return nullptr;
}
#endif

#if !defined(JST_OS_WINDOWS)
std::unique_ptr<Instance::Remote::Encoder::Input> Instance::Remote::Encoder::CreateD3D11Input() {
    return nullptr;
}
#endif

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
        case EncoderType::AMF:
            switch (codec) {
                case CodecType::H264: return "h264_amf";
                case CodecType::H265: return "hevc_amf";
                case CodecType::AV1:  return "av1_amf";
                default:              return nullptr;
            }
        case EncoderType::QSV:
            switch (codec) {
                case CodecType::H264: return "h264_qsv";
                case CodecType::H265: return "hevc_qsv";
                case CodecType::AV1:  return "av1_qsv";
                default:              return nullptr;
            }
        case EncoderType::VAAPI:
            switch (codec) {
                case CodecType::H264: return "h264_vaapi";
                case CodecType::H265: return "hevc_vaapi";
                case CodecType::AV1:  return "av1_vaapi";
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
    if (probe.config.size.x == 0 || probe.config.size.y == 0) {
        probe.config.size = {640, 480};
    }
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

    if (config.device == DeviceType::Vulkan) {
        context->pix_fmt = AV_PIX_FMT_NONE;
    } else if (config.encoder == EncoderType::VideoToolbox) {
        context->pix_fmt = AV_PIX_FMT_VIDEOTOOLBOX;
        context->sw_pix_fmt = AV_PIX_FMT_BGRA;
    } else if (config.encoder == EncoderType::QSV) {
        context->pix_fmt = AV_PIX_FMT_NV12;
    } else {
        context->pix_fmt = InputFormat(codec);
    }
    if (config.device != DeviceType::Vulkan && context->pix_fmt == AV_PIX_FMT_NONE) {
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
        case EncoderType::AMF:
            if (config.codec == CodecType::H264) {
                context->profile = AV_PROFILE_H264_CONSTRAINED_BASELINE;
            }
            context->flags |= AV_CODEC_FLAG_LOW_DELAY;
            SetOption(context, "usage", "ultralowlatency");
            SetOption(context, "quality", "speed");
            SetOption(context, "rc", "cbr");
            SetOption(context, "async_depth", "1");
            SetOption(context, "forced_idr", "1");
            if (config.codec == CodecType::AV1) {
                SetOption(context, "latency", "lowest_latency");
                SetOption(context, "header_insertion_mode", "frame");
            } else {
                SetOption(context, "latency", "1");
            }
            if (config.codec == CodecType::H265) {
                SetOption(context, "header_insertion_mode", "idr");
            }
            break;
        case EncoderType::QSV:
            if (config.codec == CodecType::H264) {
                SetOption(context, "profile", std::to_string(AV_PROFILE_H264_CONSTRAINED_BASELINE).c_str());
            }
            if (config.codec != CodecType::AV1) {
                SetOption(context, "scenario", "remotegaming");
            }
            SetOption(context, "preset", "veryfast");
            SetOption(context, "async_depth", "1");
            SetOption(context, "low_delay_brc", "1");
            SetOption(context, "forced_idr", "1");
            break;
        case EncoderType::VAAPI:
            if (config.codec == CodecType::H264) {
                context->profile = AV_PROFILE_H264_CONSTRAINED_BASELINE;
            }
            SetOption(context, "rc_mode", "CBR");
            SetOption(context, "async_depth", "1");
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
    if (config.device != DeviceType::Vulkan) {
        return Result::SUCCESS;
    }

    switch (ExportFor(config.encoder)) {
        case Export::OpaqueFd: sharing = CreateCudaInput();  break;
        case Export::DmaBuf:   sharing = CreateVaapiInput(); break;
        case Export::D3D11:    sharing = CreateD3D11Input(); break;
        default:               sharing = nullptr;            break;
    }

    if (!sharing) {
        JST_ERROR("[REMOTE] Encoder '{}' can't take shared frames on this system.", config.encoder);
        return Result::ERROR;
    }

    return sharing->create(*this);
}

Result Instance::Remote::Encoder::createFilter(AVBufferRef* frames, const char* description) {
    const auto* framesContext = reinterpret_cast<const AVHWFramesContext*>(frames->data);

    graph = avfilter_graph_alloc();
    filtered = av_frame_alloc();
    if (!graph || !filtered) {
        JST_ERROR("[REMOTE] Failed to allocate the encoder filter.");
        return Result::ERROR;
    }

    source = avfilter_graph_alloc_filter(graph, avfilter_get_by_name("buffer"), "source");
    AVBufferSrcParameters* parameters = av_buffersrc_parameters_alloc();
    if (!source || !parameters) {
        av_free(parameters);
        JST_ERROR("[REMOTE] Failed to allocate the encoder filter source.");
        return Result::ERROR;
    }

    parameters->format = framesContext->format;
    parameters->width = context->width;
    parameters->height = context->height;
    parameters->time_base = {1, 1'000'000};
    parameters->hw_frames_ctx = frames;
    parameters->color_space = AVCOL_SPC_RGB;
    parameters->color_range = AVCOL_RANGE_JPEG;
    const int set = av_buffersrc_parameters_set(source, parameters);
    av_free(parameters);

    if (set < 0 || avfilter_init_str(source, nullptr) < 0) {
        JST_ERROR("[REMOTE] Failed to configure the encoder filter source.");
        return Result::ERROR;
    }

    if (const int error = avfilter_graph_create_filter(&sink, avfilter_get_by_name("buffersink"), "sink",
                                                       nullptr, nullptr, graph); error < 0) {
        JST_ERROR("[REMOTE] Failed to create the encoder filter sink: {}", ErrorString(error));
        return Result::ERROR;
    }

    AVFilterInOut* outputs = avfilter_inout_alloc();
    AVFilterInOut* inputs = avfilter_inout_alloc();
    if (!outputs || !inputs) {
        avfilter_inout_free(&outputs);
        avfilter_inout_free(&inputs);
        JST_ERROR("[REMOTE] Failed to allocate the encoder filter links.");
        return Result::ERROR;
    }

    outputs->name = av_strdup("in");
    outputs->filter_ctx = source;
    inputs->name = av_strdup("out");
    inputs->filter_ctx = sink;

    const int parsed = avfilter_graph_parse_ptr(graph, description, &inputs, &outputs, nullptr);
    avfilter_inout_free(&outputs);
    avfilter_inout_free(&inputs);
    if (parsed < 0) {
        JST_ERROR("[REMOTE] Failed to parse the encoder filter '{}': {}", description, ErrorString(parsed));
        return Result::ERROR;
    }

    if (const int error = avfilter_graph_config(graph, nullptr); error < 0) {
        JST_ERROR("[REMOTE] Failed to configure the encoder filter '{}': {}", description, ErrorString(error));
        return Result::ERROR;
    }

    AVBufferRef* output = av_buffersink_get_hw_frames_ctx(sink);
    if (!output) {
        JST_ERROR("[REMOTE] Encoder filter '{}' produced no hardware frames.", description);
        return Result::ERROR;
    }

    context->pix_fmt = static_cast<AVPixelFormat>(av_buffersink_get_format(sink));
    context->hw_frames_ctx = av_buffer_ref(output);
    if (!context->hw_frames_ctx) {
        JST_ERROR("[REMOTE] Failed to reference the encoder filter frames.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result Instance::Remote::Encoder::createConverter() {
    if (config.device == DeviceType::Vulkan ||
        context->pix_fmt == AV_PIX_FMT_BGRA ||
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
    avfilter_graph_free(&graph);
    source = nullptr;
    sink = nullptr;
    av_frame_free(&filtered);
    av_frame_free(&converted);
    av_frame_free(&input);
    av_packet_free(&packet);
    avcodec_free_context(&context);
    sharing.reset();

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

    if (config.device == DeviceType::Vulkan) {
        return encodeShared(data, timestamp, keyframe, packets);
    }

    av_frame_unref(input);
    input->width = context->width;
    input->height = context->height;
    stamp(input, timestamp, keyframe);

    const int stride = context->width * 4;

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

AVBufferRef* Instance::Remote::Encoder::borrow(U8* data, size_t size) {
    return av_buffer_create(data, size, [](void* opaque, uint8_t*) {
        auto* encoder = static_cast<Encoder*>(opaque);
        std::lock_guard<std::mutex> lock(encoder->releaseMutex);
        encoder->released = true;
        encoder->releaseCondition.notify_all();
    }, this, AV_BUFFER_FLAG_READONLY);
}

void Instance::Remote::Encoder::stamp(AVFrame* frame, std::chrono::microseconds timestamp, bool keyframe) const {
    frame->pts = timestamp.count();
    frame->pict_type = keyframe ? AV_PICTURE_TYPE_I : AV_PICTURE_TYPE_NONE;
    frame->color_range = context->color_range;
    frame->colorspace = context->colorspace;
    frame->color_primaries = context->color_primaries;
    frame->color_trc = context->color_trc;
}

Result Instance::Remote::Encoder::encodeShared(const void* data,
                                               std::chrono::microseconds timestamp,
                                               bool keyframe,
                                               std::vector<Packet>& packets) {
    const bool borrowed = sharing->borrowed();
    if (borrowed) {
        std::lock_guard<std::mutex> lock(releaseMutex);
        released = false;
    }

    av_frame_unref(input);
    if (sharing->wrap(*this, *static_cast<const SharedFrame*>(data), input) != Result::SUCCESS) {
        av_frame_unref(input);
        std::lock_guard<std::mutex> lock(releaseMutex);
        released = true;
        return Result::ERROR;
    }

    stamp(input, timestamp, keyframe);
    if (graph) {
        input->colorspace = AVCOL_SPC_RGB;
        input->color_range = AVCOL_RANGE_JPEG;
    }

    if (borrowed) {
        return submitBorrowed(packets);
    }

    const Result delivered = deliver(input, packets);
    av_frame_unref(input);
    return delivered;
}

Result Instance::Remote::Encoder::deliver(AVFrame* frame, std::vector<Packet>& packets) {
    if (!graph) {
        JST_CHECK(submit(frame));
        av_frame_unref(frame);
        return receive(packets);
    }

    const int64_t pts = frame->pts;
    const AVPictureType type = frame->pict_type;

    if (const int error = av_buffersrc_add_frame_flags(source, frame, 0); error < 0) {
        JST_ERROR("[REMOTE] Failed to filter the encoder frame: {}", ErrorString(error));
        return Result::ERROR;
    }

    while (true) {
        const int error = av_buffersink_get_frame(sink, filtered);
        if (error == AVERROR(EAGAIN) || error == AVERROR_EOF) {
            break;
        }
        if (error < 0) {
            JST_ERROR("[REMOTE] Failed to receive the filtered encoder frame: {}", ErrorString(error));
            return Result::ERROR;
        }

        if (sharing && sharing->synchronize(filtered) != Result::SUCCESS) {
            av_frame_unref(filtered);
            return Result::ERROR;
        }

        filtered->pts = pts;
        filtered->pict_type = type;
        filtered->color_range = context->color_range;
        filtered->colorspace = context->colorspace;
        filtered->color_primaries = context->color_primaries;
        filtered->color_trc = context->color_trc;
        const Result submitted = submit(filtered);
        av_frame_unref(filtered);
        JST_CHECK(submitted);
    }

    return receive(packets);
}

Result Instance::Remote::Encoder::submitBorrowed(std::vector<Packet>& packets) {
    const Result delivered = deliver(input, packets);
    av_frame_unref(input);

    if (!waitForRelease()) {
        JST_ERROR("[REMOTE] Encoder did not release the input frame. Reopening it.");
        avcodec_free_context(&context);
        avfilter_graph_free(&graph);
        (void)waitForRelease();
        (void)create(Config(config));
        return Result::ERROR;
    }

    return delivered;
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
