#pragma once

#include "jetstream/instance_remote.hh"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <vector>

struct AVBufferRef;
struct AVCodecContext;
struct AVFrame;
struct AVPacket;
struct SwsContext;

namespace Jetstream {

struct Instance::Remote::Encoder {
 public:
    struct Config {
        CodecType codec = CodecType::H264;
        EncoderType encoder = EncoderType::Software;
        DeviceType device = DeviceType::CPU;
        Extent2D<U64> size;
        U32 framerate = 30;
        U64 bitrate = 25'000'000;
        void* context = nullptr;
    };

    struct Packet {
        std::vector<U8> data;
        std::chrono::microseconds timestamp{0};
        bool keyframe = false;
    };

    ~Encoder();

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
    static const char* CodecName(CodecType codec, EncoderType encoder);
    static Result Configure(AVCodecContext* context, const Config& config);

    Result createDevice();
    Result createConverter();
    Result allocateConverted();
    Result submit(AVFrame* frame);
    Result submitBorrowed(std::vector<Packet>& packets);
    Result receive(std::vector<Packet>& packets);
    bool waitForRelease();

    Config config;

    AVCodecContext* context = nullptr;
    AVBufferRef* device = nullptr;
    AVBufferRef* frames = nullptr;
    AVFrame* input = nullptr;
    AVFrame* converted = nullptr;
    AVPacket* packet = nullptr;
    SwsContext* converter = nullptr;
    std::vector<U8> headers;

    std::mutex releaseMutex;
    std::condition_variable releaseCondition;
    bool released = true;
};

}  // namespace Jetstream
