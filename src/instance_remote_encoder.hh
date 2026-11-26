#pragma once

#include "jetstream/instance_remote.hh"
#include "jetstream/viewport/capture.hh"

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct AVBufferRef;
struct AVCodecContext;
struct AVFilterContext;
struct AVFilterGraph;
struct AVFrame;
struct AVPacket;
struct SwsContext;

namespace Jetstream {

struct Instance::Remote::Encoder {
 public:
    using Export = Viewport::FrameCapture::Export;
    using SharedDevice = Viewport::FrameCapture::SharedDevice;
    using SharedFrame = Viewport::FrameCapture::SharedFrame;

    struct Config {
        CodecType codec = CodecType::H264;
        EncoderType encoder = EncoderType::Software;
        DeviceType device = DeviceType::CPU;
        Extent2D<U64> size;
        U32 framerate = 30;
        U64 bitrate = 25'000'000;
        SharedDevice adapter;
    };

    struct Packet {
        std::vector<U8> data;
        std::chrono::microseconds timestamp{0};
        bool keyframe = false;
    };

    ~Encoder();

    static Export ExportFor(EncoderType encoder);
    static bool Available(const Config& config);
    static std::string Parameters(const Config& config);

    Result create(const Config& config);
    Result destroy();
    std::string format() const;

    Result encode(const void* data,
                  std::chrono::microseconds timestamp,
                  bool keyframe,
                  std::vector<Packet>& packets);

 private:
    struct Input {
        virtual ~Input() = default;

        virtual Result create(Encoder& encoder) = 0;
        virtual Result wrap(Encoder& encoder, const SharedFrame& frame, AVFrame* output) = 0;
        virtual bool borrowed() const = 0;
        virtual Result synchronize(AVFrame*) { return Result::SUCCESS; }
    };

    struct CudaInput;
    struct VaapiInput;
    struct D3D11Input;

    static std::unique_ptr<Input> CreateCudaInput();
    static std::unique_ptr<Input> CreateVaapiInput();
    static std::unique_ptr<Input> CreateD3D11Input();

    static const char* CodecName(CodecType codec, EncoderType encoder);
    static Result Configure(AVCodecContext* context, const Config& config);

    Result createDevice();
    Result createFilter(AVBufferRef* frames, const char* description);
    Result createConverter();
    Result allocateConverted();
    AVBufferRef* borrow(U8* data, size_t size);
    void stamp(AVFrame* frame, std::chrono::microseconds timestamp, bool keyframe) const;
    Result encodeShared(const void* data,
                        std::chrono::microseconds timestamp,
                        bool keyframe,
                        std::vector<Packet>& packets);
    Result submit(AVFrame* frame);
    Result deliver(AVFrame* frame, std::vector<Packet>& packets);
    Result submitBorrowed(std::vector<Packet>& packets);
    Result receive(std::vector<Packet>& packets);
    bool waitForRelease();

    Config config;

    AVCodecContext* context = nullptr;
    AVFilterGraph* graph = nullptr;
    AVFilterContext* source = nullptr;
    AVFilterContext* sink = nullptr;
    AVFrame* input = nullptr;
    AVFrame* filtered = nullptr;
    AVFrame* converted = nullptr;
    AVPacket* packet = nullptr;
    SwsContext* converter = nullptr;
    std::unique_ptr<Input> sharing;
    std::vector<U8> headers;

    std::mutex releaseMutex;
    std::condition_variable releaseCondition;
    bool released = true;
};

}  // namespace Jetstream
