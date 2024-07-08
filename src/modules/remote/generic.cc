#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>

#include <regex>

#include "jetstream/modules/remote.hh"
#include "jetstream/instance.hh"
#include "jetstream/backend/devices/cpu/helpers.hh"

#include "shaders/remote_shaders.hh"
#include "assets/constants.hh"

namespace Jetstream {

template<Device D, typename T>
Result Remote<D, T>::create() {
    JST_DEBUG("Initializing Remote module.");

    // Set variables.

    _statistics = {};

    // Check if endpoint is valid.

    if (config.endpoint.empty()) {
        JST_ERROR("Invalid endpoint.");
        return Result::ERROR;
    }

    // Get socket address and port with regex.

    std::smatch matches;
    std::regex pattern(R"(([\w.-]+|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d{1,5}))");
    if (std::regex_match(config.endpoint, matches, pattern)) {
        if (matches.size() == 3) {
            brokerAddress = matches[1].str();
            brokerPort = std::stoi(matches[2].str());
        }
    } else {
        JST_ERROR("Invalid endpoint format. Expected `address:port`. Example: `10.10.1.1:5000`.");
        return Result::ERROR;
    }

    // Create TCP client.

    brokerFileDescriptor = socket(AF_INET, SOCK_STREAM, 0);
    if (brokerFileDescriptor < 0) {
        JST_ERROR("Failed to open socket.");
        return Result::ERROR;
    }

    // Set socket options.

    int delay = 1;
    if (setsockopt(brokerFileDescriptor, IPPROTO_TCP, TCP_NODELAY, &delay, sizeof(delay)) < 0) {
        JST_ERROR("Failed to set no delay socket option.");
        return Result::ERROR;
    }

    int keepalive = 1;
    if (setsockopt(brokerFileDescriptor, SOL_SOCKET, SO_KEEPALIVE, &keepalive, sizeof(keepalive)) < 0) {
        JST_ERROR("Failed to set keep alive socket option.");
        return Result::ERROR;
    }

    // Connect to server.

    sockaddr_in serverAddress;
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_addr.s_addr = inet_addr(brokerAddress.c_str());
    serverAddress.sin_port = htons(brokerPort);

    if (connect(brokerFileDescriptor, (sockaddr*)&serverAddress, sizeof(serverAddress)) < 0) {
        JST_ERROR("Failed to connect to broker.");
        return Result::ERROR;
    }
    brokerConnected = true;

    // Send framebuffer size request command. 

    {
        auto response = jst::fmt::format("cmd:fbsize\n");
        send(brokerFileDescriptor, response.c_str(), response.size(), 0);
    }

    // Create thread.

    std::promise<Result> brokerEndpointSignal;
    std::future<Result> brokerEndpointFuture = brokerEndpointSignal.get_future();

    brokerThread = std::thread([&, b = std::move(brokerEndpointSignal)]() mutable {
        while (brokerConnected) {
            // Read header.

            char buffer[1024] = {0};

            if (read(brokerFileDescriptor, buffer, sizeof(buffer)) < 0) {
                break;
            }
            const auto packetTime = std::chrono::high_resolution_clock::now();

            std::string line;
            std::istringstream lineStream(buffer);
            while (std::getline(lineStream, line)) {
                // Parse command `pong`.

                if (line.compare(0, 4, "pong") == 0) {
                    _statistics.latency = std::chrono::duration_cast<std::chrono::milliseconds>(packetTime - lastPingTime).count();
                    continue;
                }

                JST_TRACE("[ENDPOINT] Received message from server: '{}'.", line);

                // Parse command `err`.

                if (line.compare(0, 3, "err") == 0) {
                    JST_ERROR("Received `err` from server: '{}'.", line);
                    close(brokerFileDescriptor);
                    break;
                }

                // Parse command `ok`.

                if (line.compare(0, 2, "ok") == 0) {
                    // Parse command `ok:fbsize:width,height`.

                    if (line.compare(0, 9, "ok:fbsize") == 0) {
                        remoteFramebufferSize.x = std::stoi(line.substr(10, 15));
                        remoteFramebufferSize.y = std::stoi(line.substr(16, 21));

                        JST_DEBUG("Received `ok:fbsize` from server: `width={}, height={}`.", remoteFramebufferSize.x, 
                                                                                              remoteFramebufferSize.y);
                        
                        auto response = jst::fmt::format("cmd:framerate\n");
                        send(brokerFileDescriptor, response.c_str(), response.size(), 0);

                        continue;
                    }

                    // Parse command `ok:framerate:fps`.

                    if (line.compare(0, 12, "ok:framerate") == 0) {
                        remoteFramerate = std::stoi(line.substr(13, line.length() - 1));

                        JST_DEBUG("Received `ok:framerate` from server: `framerate={}`.", remoteFramerate);

                        auto response = jst::fmt::format("cmd:codec\n");
                        send(brokerFileDescriptor, response.c_str(), response.size(), 0);

                        continue;
                    }

                    // Parse command `ok:codec:id`.

                    if (line.compare(0, 8, "ok:codec") == 0) {
                        const auto codec = line.substr(9, line.length() - 1);
                        remoteFramebufferCodec = Viewport::StringToVideoCodec(codec);

                        JST_DEBUG("Received `ok:codec` from server: `id={}`.", codec);
                        
                        b.set_value(Result::SUCCESS);

                        continue;
                    }

                    // Parse command `ok:connect`.

                    if (line.compare(0, 10, "ok:connect") == 0) {
                        socketConnected = true;
                        socketAddress = "0.0.0.0";
                        socketPort = brokerPort + 1;

                        if (createGstreamerEndpoint() != Result::SUCCESS) {
                            close(brokerFileDescriptor);
                            break;
                        }

                        continue;
                    }

                    JST_DEBUG("Received `ok` from server: '{}'.", line);
                    continue;
                }

                // Reply for unrecognized message.

                {
                    JST_ERROR("Received unrecognized message from server: '{}'.", line);
                    auto response = jst::fmt::format("err:Unrecognized message.\n");
                    send(brokerFileDescriptor, response.c_str(), response.size(), 0);
                    continue;
                }
            }
        }

        JST_INFO("[ENDPOINT] Server disconnected. Closing socket.");

        destroyGstreamerEndpoint();
        brokerConnected = false;
    });

    // Wait for broker signal.

    if (brokerEndpointFuture.wait_for(std::chrono::seconds(15)) != std::future_status::ready) {
        JST_ERROR("Failed to get a response from headless server.");
        return Result::ERROR;
    }
    JST_CHECK(brokerEndpointFuture.get());

    // Allocating framebuffer memory.

    remoteFramebufferMemory.resize(remoteFramebufferSize.x * 
                                   remoteFramebufferSize.y * 4);

    // Send streaming start command. 

    {
        auto response = jst::fmt::format("cmd:connect\n");
        send(brokerFileDescriptor, response.c_str(), response.size(), 0);
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Remote<D, T>::destroy() {
    // Close broker endpoint.

    if (brokerConnected) {
        // Send streaming stop command.

        if (socketStreaming) {
            auto response = jst::fmt::format("cmd:disconnect\n");
            send(brokerFileDescriptor, response.c_str(), response.size(), 0);
        }

        // Close broker endpoint.

        brokerConnected = false;
        close(brokerFileDescriptor);
        shutdown(brokerFileDescriptor, SHUT_RD);
        if (brokerThread.joinable()) {
            brokerThread.join();
        }
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
void Remote<D, T>::info() const {
    JST_DEBUG("  Endpoint:     {}", config.endpoint);
    JST_DEBUG("  Window Size:  [{}, {}]", config.viewSize.x, config.viewSize.y);
}

template<Device D, typename T>
Result Remote<D, T>::createGstreamerEndpoint() {
    JST_DEBUG("Creating gstreamer endpoint.");

    // Initialize gstreamer.

    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
    }

    // List required plugins.

    std::vector<std::string> plugins = {
        "app",
        "rawparse",
        "coreelements",
        "matroska",
        "rtp",
        "udp",
        "rtpmanager",
    };

    // Add codec specific plugins.

    if (remoteFramebufferCodec == Viewport::VideoCodec::H264) {
        plugins.push_back("libav");
    } else if (remoteFramebufferCodec == Viewport::VideoCodec::VP8) {
        plugins.push_back("vpx");
    } else if (remoteFramebufferCodec == Viewport::VideoCodec::VP9) {
        plugins.push_back("vpx");
    } else if (remoteFramebufferCodec == Viewport::VideoCodec::AV1) {
        plugins.push_back("dav1d");
    } else if (remoteFramebufferCodec == Viewport::VideoCodec::FFV1) {
        plugins.push_back("libav");
    } else {
        JST_ERROR("Unsupported remote framebuffer codec.");
        return Result::ERROR;
    }

    // Check if all plugins are available.

    for (const auto& plugin : plugins) {
        if (!gst_registry_find_plugin(gst_registry_get(), plugin.c_str())) {
            JST_ERROR("Gstreamer plugin '{}' is not available.", plugin);
            return Result::ERROR;
        }
    }

    // Create pipeline.

    pipeline = gst_pipeline_new("headless-sink-pipeline");

    if (!pipeline) {
        JST_ERROR("Failed to create gstreamer pipeline.");
        return Result::ERROR;
    }

    // Create elements.

    std::map<std::string, GstElement*> elements;
    std::vector<std::string> elementOrder = {
        "source",
        "srcfilter",
        "jitter",
        "demuxer",
        "decoder",
        "convert",
        "sinkfilter",
        "sink"
    };

    elements["source"] = gst_element_factory_make("udpsrc", "source");
    elements["srcfilter"] = gst_element_factory_make("capsfilter", "srcfilter");
    elements["jitter"] = gst_element_factory_make("rtpjitterbuffer", "jitter");
    elements["demuxer"] = gst_element_factory_make("rtpgstdepay", "demuxer");

    if (remoteFramebufferCodec == Viewport::VideoCodec::FFV1) {
        elements["decoder"] = gst_element_factory_make("avdec_ffv1", "decoder");
    } else if (remoteFramebufferCodec == Viewport::VideoCodec::VP8) {
        elements["decoder"] = gst_element_factory_make("vp8dec", "decoder");
    } else if (remoteFramebufferCodec == Viewport::VideoCodec::VP9) {
        elements["decoder"] = gst_element_factory_make("vp9dec", "decoder");
    } else if (remoteFramebufferCodec == Viewport::VideoCodec::AV1) {
        elements["decoder"] = gst_element_factory_make("dav1ddec", "decoder");
    } else if (remoteFramebufferCodec == Viewport::VideoCodec::H264) {
        elements["decoder"] = gst_element_factory_make("avdec_h264", "decoder");
    }

    elements["convert"] = gst_element_factory_make("videoconvert", "convert");
    elements["sinkfilter"] = gst_element_factory_make("capsfilter", "sinkfilter");
    elements["sink"] = gst_element_factory_make("appsink", "sink");

    for (const auto& [name, element] : elements) {
        if (!element) {
            JST_ERROR("Failed to create gstreamer element '{}'.", name);
            gst_object_unref(pipeline);
            return Result::ERROR;
        }
    }

    // Configure elements.

    GstCaps* caps = gst_caps_new_simple("application/x-rtp",
                                        "media", G_TYPE_STRING, "application",
                                        nullptr);
    g_object_set(elements["srcfilter"], "caps", caps, nullptr);
    gst_caps_unref(caps);

    g_object_set(elements["source"], "address", socketAddress.c_str(), "port", socketPort, nullptr);
    g_object_set(elements["source"], "buffer-size", Backend::GetSocketBufferSize(), nullptr);

    GstCaps* convertCaps = gst_caps_new_simple("video/x-raw",
                                               "format", G_TYPE_STRING, "RGBA",
                                               nullptr);
    g_object_set(elements["sinkfilter"], "caps", convertCaps, nullptr);
    gst_caps_unref(convertCaps);

    g_object_set(elements["sink"], "emit-signals", true, nullptr);
    g_object_set(elements["sink"], "sync", false, nullptr);

    // Link callbacks.

    g_signal_connect(elements["sink"], "new-sample", G_CALLBACK(OnSampleCallback), this);
        
    // Add elements to pipeline.

    for (const auto& [name, element] : elements) {
        if (!gst_bin_add(GST_BIN(pipeline), element)) {
            JST_ERROR("Failed to add gstreamer element '{}' to pipeline.", name);
            gst_object_unref(pipeline);
            return Result::ERROR;
        }
    }

    // Link elements.

    std::string lastElement = "source";
    for (const auto& name : elementOrder) {
        if (name == "source") {
            continue;
        }

        if (!gst_element_link(elements[lastElement], elements[name])) {
            JST_ERROR("Failed to link gstreamer element '{}' -> '{}'.", lastElement, name);
            gst_object_unref(pipeline);
            return Result::ERROR;
        }

        lastElement = name;
    }

    // Start pipeline.

    if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        JST_ERROR("Failed to start gstreamer pipeline.");
        gst_object_unref(pipeline);
        return Result::ERROR;
    }

    // Signal success.
    socketStreaming = true;

    return Result::SUCCESS;
}

template<Device D, typename T>
GstFlowReturn Remote<D, T>::OnSampleCallback(GstElement* sink, gpointer data) {
    auto* that = reinterpret_cast<Remote*>(data);

    GstSample* sample;
    g_signal_emit_by_name(sink, "pull-sample", &sample);
    if (sample) {
        GstBuffer* buffer = gst_sample_get_buffer(sample);
        GstMapInfo map;
        gst_buffer_map(buffer, &map, GST_MAP_READ);

        {
            std::lock_guard<std::mutex> lock(that->localFramebufferMutex);
            memcpy(that->remoteFramebufferMemory.data(), map.data, map.size);
            that->localFramebufferAvailable = true;
        }

        that->_statistics.frames += 1;
        
        gst_buffer_unmap(buffer, &map);

        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }

    return GST_FLOW_ERROR;
}

template<Device D, typename T>
Result Remote<D, T>::destroyGstreamerEndpoint() {
    JST_DEBUG("Destroying gstreamer endpoint.");

    // Stop pipeline.

    if (socketStreaming) {
        // Stop piping frames.
        socketStreaming = false;

        // Send EOS.
        gst_element_send_event(pipeline, gst_event_new_eos());

        // Wait pipeline to process EOS.

        GstBus* bus = gst_element_get_bus(pipeline);
        GstMessage* msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE, GST_MESSAGE_EOS);
        if (msg) {
            gst_message_unref(msg);
        }
        gst_object_unref(bus);

        // Stop pipeline.
        gst_element_set_state(pipeline, GST_STATE_NULL);

        // Destroy pipeline.
        gst_object_unref(pipeline);        
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Remote<D, T>::createPresent() {
    // Frame rendering.

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &FillScreenVertices;
        cfg.elementByteSize = sizeof(F32);
        cfg.size = 12;
        cfg.target = Render::Buffer::Target::VERTEX;
        JST_CHECK(window->build(fillScreenVerticesBuffer, cfg));
        JST_CHECK(window->bind(fillScreenVerticesBuffer));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &FillScreenTextureVerticesXYFlip;
        cfg.elementByteSize = sizeof(F32);
        cfg.size = 8;
        cfg.target = Render::Buffer::Target::VERTEX;
        JST_CHECK(window->build(fillScreenTextureVerticesBuffer, cfg));
        JST_CHECK(window->bind(fillScreenTextureVerticesBuffer));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &FillScreenIndices;
        cfg.elementByteSize = sizeof(U32);
        cfg.size = 6;
        cfg.target = Render::Buffer::Target::VERTEX_INDICES;
        JST_CHECK(window->build(fillScreenIndicesBuffer, cfg));
        JST_CHECK(window->bind(fillScreenIndicesBuffer));
    }

    {
        Render::Vertex::Config cfg;
        cfg.buffers = {
            {fillScreenVerticesBuffer, 3},
            {fillScreenTextureVerticesBuffer, 2},
        };
        cfg.indices = fillScreenIndicesBuffer;
        JST_CHECK(window->build(vertex, cfg));
    }

    {
        Render::Draw::Config cfg;
        cfg.buffer = vertex;
        cfg.mode = Render::Draw::Mode::TRIANGLES;
        JST_CHECK(window->build(drawVertex, cfg));
    }

    {
        Render::Texture::Config cfg;
        cfg.size = remoteFramebufferSize;
        cfg.buffer = remoteFramebufferMemory.data();
        JST_CHECK(window->build(remoteFramebufferTexture, cfg));
        JST_CHECK(window->bind(remoteFramebufferTexture));
    }

    {
        Render::Program::Config cfg;
        cfg.shaders = ShadersPackage["framebuffer"];
        cfg.draw = drawVertex;
        cfg.textures = {remoteFramebufferTexture};
        JST_CHECK(window->build(program, cfg));
    }

    // Surface.

    {
        Render::Texture::Config cfg;
        cfg.size = config.viewSize;
        JST_CHECK(window->build(framebufferTexture, cfg));
    }

    {
        Render::Surface::Config cfg;
        cfg.framebuffer = framebufferTexture;
        cfg.programs = {program};
        JST_CHECK(window->build(surface, cfg));
        JST_CHECK(window->bind(surface));
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Remote<D, T>::destroyPresent() {
    JST_CHECK(window->unbind(surface));
    JST_CHECK(window->unbind(remoteFramebufferTexture));
    JST_CHECK(window->unbind(fillScreenVerticesBuffer));
    JST_CHECK(window->unbind(fillScreenTextureVerticesBuffer));
    JST_CHECK(window->unbind(fillScreenIndicesBuffer));

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Remote<D, T>::present() {
    if (socketStreaming) {
        if (localFramebufferAvailable) {
            std::lock_guard<std::mutex> lock(localFramebufferMutex);
            JST_CHECK(remoteFramebufferTexture->fill());
            localFramebufferAvailable = false;
        }

        if ((_statistics.frames % static_cast<U64>(remoteFramerate)) == 0) {
            static char buffer[8] = "ping\n";
            lastPingTime = std::chrono::high_resolution_clock::now();
            send(brokerFileDescriptor, buffer, 5, 0);
        }
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
void Remote<D, T>::registerMousePos(const F32& x, const F32& y) {
    if (socketStreaming) {
        const I32 X = static_cast<I32>((x / config.viewSize.x) * remoteFramebufferSize.x);
        const I32 Y = static_cast<I32>((y / config.viewSize.y) * remoteFramebufferSize.y);

        static char buffer[32];
        snprintf(buffer, 32, "hid:mouse:pos:%05d,%05d\n", X, Y);
        send(brokerFileDescriptor, buffer, 26, 0);
    }
}

template<Device D, typename T>
void Remote<D, T>::registerMouseButton(const I32& button, const bool& pressed) {
    if (socketStreaming) {
        static char buffer[32];
        snprintf(buffer, 32, "hid:mouse:btn:%01d,%01d\n", button, ((pressed) ? 1 : 0));
        send(brokerFileDescriptor, buffer, 18, 0);
    }
}

template<Device D, typename T>
void Remote<D, T>::registerMouseScroll(const F32& deltaX, const F32& deltaY) {
    if (socketStreaming) {
        const I32 DeltaX = static_cast<I32>(deltaX * 100.0f);
        const I32 DeltaY = static_cast<I32>(deltaY * 100.0f);

        static char buffer[32];
        snprintf(buffer, 32, "hid:mouse:scroll:%+06d,%+06d\n", DeltaX, DeltaY);
        send(brokerFileDescriptor, buffer, 31, 0);
    }
}

template<Device D, typename T>
void Remote<D, T>::registerKey(const I32& key, const bool& pressed) {
    if (socketStreaming) {
        static char buffer[16];
        snprintf(buffer, 16, "hid:key:%04d,%01d\n", key, ((pressed) ? 1 : 0));
        send(brokerFileDescriptor, buffer, 15, 0);
    }
}

template<Device D, typename T>
void Remote<D, T>::registerChar(const char& key) {
    if (socketStreaming) {
        static char buffer[16];
        snprintf(buffer, 16, "hid:char:%c\n", key);
        send(brokerFileDescriptor, buffer, 11, 0);
    }
}

template<Device D, typename T>
const Extent2D<U64>& Remote<D, T>::viewSize(const Extent2D<U64>& viewSize) {
    Extent2D<U64> correctedViewSize = viewSize;

    // Correct aspect ratio.

    const F32 nativeAspectRatio = static_cast<F32>(remoteFramebufferSize.x) / 
                                  static_cast<F32>(remoteFramebufferSize.y);

    const F32 viewAspectRatio = static_cast<F32>(viewSize.x) / 
                                static_cast<F32>(viewSize.y);

    if (viewAspectRatio > nativeAspectRatio) {
        correctedViewSize.x = correctedViewSize.y * nativeAspectRatio;
    } else {
        correctedViewSize.y = correctedViewSize.x / nativeAspectRatio;
    }

    // Submit new size.

    if (surface->size(correctedViewSize) != config.viewSize) {
        config.viewSize = surface->size();
    }
    return config.viewSize;
}

template<Device D, typename T>
Render::Texture& Remote<D, T>::getTexture() {
    return *framebufferTexture;
};

JST_REMOTE_CPU(JST_INSTANTIATION)

}  // namespace Jetstream
