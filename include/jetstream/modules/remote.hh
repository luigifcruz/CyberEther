#ifndef JETSTREAM_MODULES_REMOTE_HH
#define JETSTREAM_MODULES_REMOTE_HH

#include <gst/gst.h>

#include <future>
#include <thread>
#include <chrono>
#include <mutex>

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory/base.hh"
#include "jetstream/render/base.hh"
#include "jetstream/render/extras.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

template<Device D, typename T = void>
class Remote : public Module, public Present {
 public:
    // Configuration 

    struct Config {
        std::string endpoint = "tcp://0.0.0.0:5000";
        Size2D<U64> viewSize = {1280, 720};

        JST_SERDES(
            JST_SERDES_VAL("endpoint", endpoint);
            JST_SERDES_VAL("viewSize", viewSize);
        );
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        JST_SERDES();
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        JST_SERDES();
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    // Taint & Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string_view name() const {
        return "remote";
    }

    std::string_view prettyName() const {
        return "Remote";
    }

    void summary() const final;

    // Constructor

    Result create();
    Result destroy();

    // Miscellaneous

    struct Statistics {
        F32 latency;
        U64 frames;
    };

    constexpr const Statistics& statistics() const {
        return _statistics;
    }

    constexpr const F32& getRemoteFramerate() const {
        return remoteFramerate;
    }

    constexpr const Size2D<U64>& getRemoteFramebufferSize() const {
        return remoteFramebufferSize;
    }

    constexpr const Render::VideoCodec& getRemoteFramebufferCodec() const {
        return remoteFramebufferCodec;
    }

    constexpr const bool& isBrokerConnected() const {
        return brokerConnected;
    }

    constexpr const bool& isSocketStreaming() const {
        return socketStreaming;
    }

    constexpr const Size2D<U64>& viewSize() const {
        return config.viewSize;
    }
    const Size2D<U64>& viewSize(const Size2D<U64>& viewSize);

    void registerMousePos(const F32& x, const F32& y);
    void registerMouseButton(const I32& button, const bool& pressed);
    void registerMouseScroll(const F32& deltaX, const F32& deltaY);
    void registerKey(const I32& key, const bool& pressed);
    void registerChar(const char& key);

    Render::Texture& getTexture();

 protected:
    Result createPresent() final;
    Result present() final;
    Result destroyPresent() final;

 private:
    // Gstreamer.
    GstElement* pipeline;
    GstElement* demuxer;

    // Broker endpoint.
    bool brokerConnected;
    int brokerPort;
    std::string brokerAddress;
    int brokerFileDescriptor;
    std::thread brokerThread;

    // Socket endpoint.
    bool socketConnected;
    bool socketStreaming;
    int socketPort;
    std::string socketAddress;

    // Remote framebuffer.
    Size2D<U64> remoteFramebufferSize;
    Render::VideoCodec remoteFramebufferCodec;
    std::vector<U8> remoteFramebufferMemory;
    F32 remoteFramerate;

    // Local framebuffer.
    bool localFramebufferAvailable;
    std::mutex localFramebufferMutex;
    std::shared_ptr<Render::Buffer> fillScreenVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillScreenTextureVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillScreenIndicesBuffer;

    std::shared_ptr<Render::Texture> texture;
    std::shared_ptr<Render::Texture> remoteFramebufferTexture;
    std::shared_ptr<Render::Program> program;
    std::shared_ptr<Render::Surface> surface;
    std::shared_ptr<Render::Vertex> vertex;
    std::shared_ptr<Render::Draw> drawVertex;

    Statistics _statistics;
    std::chrono::high_resolution_clock::time_point lastPingTime;

    Result createGstreamerEndpoint();
    Result destroyGstreamerEndpoint();

    static GstFlowReturn OnSampleCallback(GstElement* sink, gpointer data);

    JST_DEFINE_IO();
};

}  // namespace Jetstream

#endif
