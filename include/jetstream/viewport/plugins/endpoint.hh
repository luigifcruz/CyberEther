#ifndef JETSTREAM_VIEWPORT_PLUGINS_ENDPOINT_HH
#define JETSTREAM_VIEWPORT_PLUGINS_ENDPOINT_HH

#include "jetstream/viewport/adapters/generic.hh"

#ifdef JETSTREAM_LOADER_GSTREAMER_AVAILABLE
#include <gst/gst.h>
#endif

#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

namespace Jetstream::Viewport {

class Endpoint {
 public:
    Result create(const Viewport::Config& config, const Device& viewport_device);
    Result destroy();

    const Device& inputMemoryDevice() const {
        return _inputMemoryDevice;
    }

    Result pushNewFrame(const void* data);

 private:
    enum class Type {
        File,
        Pipe,
        Socket,
        Unknown
    };

    enum class Strategy {
        None,
        Software,
        HardwareNvidia,
        HardwareRpi
    };

    std::string StrategyToString(const Strategy& strategy) {
        switch (strategy) {
            case Strategy::None:
                return "None";
            case Strategy::Software:
                return "Software";
            case Strategy::HardwareNvidia:
                return "Hardware NVIDIA (NVENC)";
            case Strategy::HardwareRpi:
                return "Hardware Raspberry Pi";
            default:
                return "Unknown";
        }
    }

    Config config;
    Endpoint::Type type;

    Device _inputMemoryDevice = Device::None;
    Strategy _encodingStrategy = Strategy::None;

    Device viewportDevice = Device::None;

    Result createFileEndpoint();
    Result createPipeEndpoint();
    Result createSocketEndpoint();

    Result destroyFileEndpoint();
    Result destroyPipeEndpoint();
    Result destroySocketEndpoint();

    static Endpoint::Type DetermineEndpointType(const std::string& endpoint);

    // Pipe endpoint.
    int pipeFileDescriptor;
    bool pipeCreated = false;

#ifdef JETSTREAM_LOADER_GSTREAMER_AVAILABLE
    GstElement* pipeline;
    GstElement* source;
    GstElement* encoder;

    std::mutex bufferMutex;
    std::condition_variable bufferCond;
    bool bufferProcessed = false;

    bool forceKeyframe = false;
    std::chrono::time_point<std::chrono::steady_clock> initialFrameTime;
    std::chrono::time_point<std::chrono::steady_clock> lastKeyframeTime;

    static void OnBufferReleaseCallback(gpointer user_data);

    // Broker endpoint.
    std::string brokerAddress;
    int brokerPort;
    int brokerEndpointFileDescriptor;
    int brokerClientFileDescriptor;
    bool brokerEndpointRunning;
    std::thread brokerEndpointThread;

    // Socket endpoint.
    std::string socketAddress;
    int socketPort;
    bool socketConnected = false;
    bool socketStreaming = false;

    // File endpoint. 
    std::string fileExtension;

    Result createGstreamerEndpoint();
    Result startGstreamerEndpoint();
    Result stopGstreamerEndpoint();
    Result destroyGstreamerEndpoint();

    Result checkGstreamerPlugins(const std::vector<std::string>& plugins);
#endif
};

}  // namespace Jetstream::Viewport

#endif