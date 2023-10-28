#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>

#include "jetstream/modules/remote.hh"
#include "jetstream/instance.hh"
#include "shaders/remote_shaders.hh"

// TODO: Give two-shits about V-Sync. 

namespace Jetstream {

template<Device D, typename T>
Result Remote<D, T>::create() {
    JST_DEBUG("Initializing Remote module.");

    // Set variables.

    brokerEndpointRunning = false;
    socketConnected = false;
    brokerConnected = false;
    frameCounter = 0;

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

    brokerEndpointFileDescriptor = socket(AF_INET, SOCK_STREAM, 0);
    if (brokerEndpointFileDescriptor < 0) {
        JST_ERROR("Failed to open socket.");
        return Result::ERROR;
    }

    // Set socket options.

    int flag = 1;
    if (setsockopt(brokerEndpointFileDescriptor, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, sizeof(flag)) < 0) {
        JST_ERROR("Failed to set socket options.");
        return Result::ERROR;
    }

    // Connect to server.

    sockaddr_in serverAddress;
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_addr.s_addr = inet_addr(brokerAddress.c_str());
    serverAddress.sin_port = htons(brokerPort);

    if (connect(brokerEndpointFileDescriptor, (sockaddr*)&serverAddress, sizeof(serverAddress)) < 0) {
        JST_ERROR("Failed to connect to server.");
        return Result::ERROR;
    }
    brokerConnected = true;

    // Send framebuffer size request command. 

    {
        auto response = fmt::format("cmd:fbsize\ncmd:framerate\n");
        send(brokerEndpointFileDescriptor, response.c_str(), response.size(), 0);
    }

    // Create thread.

    std::promise<Result> brokerEndpointSignal;
    std::future<Result> brokerEndpointFuture = brokerEndpointSignal.get_future();

    brokerEndpointRunning = true;
    brokerEndpointThread = std::thread([&, b = std::move(brokerEndpointSignal)]() mutable {
        while (brokerEndpointRunning) {
            // Read header.

            char buffer[1024] = {0};

            if (read(brokerEndpointFileDescriptor, buffer, sizeof(buffer)) < 0) {
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

                JST_TRACE("[ENDPOINT] Received message from server: `{}`.", line);

                // Parse command `err`.

                if (line.compare(0, 3, "err") == 0) {
                    JST_ERROR("Received `err` from server: `{}`.", line);
                    close(brokerEndpointFileDescriptor);
                    break;
                }

                // Parse command `ok`.

                if (line.compare(0, 2, "ok") == 0) {
                    // Parse command `ok:fbsize:width,height`.

                    if (line.compare(0, 9, "ok:fbsize") == 0) {
                        remoteFramebufferSize.width = std::stoi(line.substr(10, 15));
                        remoteFramebufferSize.height = std::stoi(line.substr(16, 21));

                        JST_DEBUG("Received `ok:fbsize` from server: `width={}, height={}`.", remoteFramebufferSize.width, 
                                                                                              remoteFramebufferSize.height);
                        b.set_value(Result::SUCCESS);
                        continue;
                    }

                    // Parse command `ok:framerate:fps`.

                    if (line.compare(0, 12, "ok:framerate") == 0) {
                        remoteFramerate = std::stoi(line.substr(13, line.length() - 1));
                        JST_DEBUG("Received `ok:framerate` from server: `framerate={}`.", remoteFramerate);
                        continue;
                    }

                    // Parse command `ok:connect`.

                    if (line.compare(0, 10, "ok:connect") == 0) {
                        socketAddress = "0.0.0.0";
                        socketPort = brokerPort + 1;
                        if (createGstreamerEndpoint() != Result::SUCCESS) {
                            close(brokerEndpointFileDescriptor);
                            break;
                        }
                        socketConnected = true;
                        continue;
                    }

                    JST_DEBUG("Received `ok` from server: `{}`.", line);
                    continue;
                }

                // Reply for unrecognized message.

                {
                    JST_ERROR("Received unrecognized message from server: `{}`.", line);
                    auto response = fmt::format("err:Unrecognized message.\n");
                    send(brokerEndpointFileDescriptor, response.c_str(), response.size(), 0);
                    continue;
                }
            }
        }

        if (socketConnected) {
            socketConnected = false;
            destroyGstreamerEndpoint();
        }

        brokerConnected = false;
        close(brokerEndpointFileDescriptor);
    });

    // Wait for broker signal.

    if (brokerEndpointFuture.wait_for(std::chrono::seconds(15)) != std::future_status::ready) {
        JST_ERROR("Failed to get a response from headless server.");
        return Result::ERROR;
    }
    JST_CHECK(brokerEndpointFuture.get());

    // Allocating framebuffer memory.

    remoteFramebufferMemory.resize(remoteFramebufferSize.width * 
                                   remoteFramebufferSize.height * 4);

    // Send streaming start command. 

    {
        auto response = fmt::format("cmd:connect\n");
        send(brokerEndpointFileDescriptor, response.c_str(), response.size(), 0);
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Remote<D, T>::destroy() {
    // Send streaming stop command.

    if (socketConnected) {
        auto response = fmt::format("cmd:disconnect\n");
        send(brokerEndpointFileDescriptor, response.c_str(), response.size(), 0);
    }

    brokerEndpointRunning = false;
    if (brokerEndpointThread.joinable()) {
        brokerEndpointThread.join();
    }

    // Reset variables.

    socketConnected = false;
    brokerConnected = false;
    frameCounter = 0;

    return Result::SUCCESS;
}

template<Device D, typename T>
void Remote<D, T>::summary() const {
    JST_INFO("  Endpoint:     {}", config.endpoint);
    JST_INFO("  Window Size:  [{}, {}]", config.viewSize.width, config.viewSize.height);
}

template<Device D, typename T>
Result Remote<D, T>::createGstreamerEndpoint() {
    JST_DEBUG("Creating gstreamer endpoint.");

    // Initialize gstreamer.

    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
    }

    // Check if all plugins are available.

    std::vector<std::string> plugins = {
        "app",
        "rawparse",
        "videoconvertscale",
        "coreelements",
        "matroska",
        "rtp",
        "udp",
        "rtpmanager",
    };

    for (const auto& plugin : plugins) {
        if (!gst_registry_find_plugin(gst_registry_get(), plugin.c_str())) {
            JST_ERROR("Gstreamer plugin `{}` is not available.", plugin);
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
    };

    elements["source"] = gst_element_factory_make("udpsrc", "source");
    elements["srcfilter"] = gst_element_factory_make("capsfilter", "srcfilter");
    elements["jitter"] = gst_element_factory_make("rtpjitterbuffer", "jitter");
    elements["demuxer"] = demuxer = gst_element_factory_make("rtpgstdepay", "demuxer");
    
    for (const auto& [name, element] : elements) {
        if (!element) {
            JST_ERROR("Failed to create gstreamer element `{}`.", name);
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

    // Link callback.

    std::future<Result> muxerFuture = muxerSignal.get_future();
    auto* srcpad = gst_element_get_static_pad(elements["demuxer"], "src");
    gst_pad_add_probe(srcpad, GST_PAD_PROBE_TYPE_BUFFER, MuxerReadyCallabck, this, nullptr);
    gst_object_unref(srcpad);
        
    // Add elements to pipeline.

    for (const auto& [name, element] : elements) {
        if (!gst_bin_add(GST_BIN(pipeline), element)) {
            JST_ERROR("Failed to add gstreamer element `{}` to pipeline.", name);
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
            JST_ERROR("Failed to link gstreamer element `{}` -> `{}`.", lastElement, name);
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

    // Wait for muxer signal.
    muxerFuture.wait();
    JST_CHECK(muxerFuture.get());

    return Result::SUCCESS;
}

template<Device D, typename T>
GstPadProbeReturn Remote<D, T>::MuxerReadyCallabck(GstPad* pad, GstPadProbeInfo*, gpointer data) {
    auto* that = reinterpret_cast<Remote*>(data);

    // Get remote framebuffer size.

    GstCaps* caps = gst_pad_get_current_caps(pad);
    GstStructure* capsStruct = gst_caps_get_structure(caps, 0);

    int width, height;
    if (!gst_structure_get_int(capsStruct, "width", &width) ||
        !gst_structure_get_int(capsStruct, "height", &height)) {
        JST_ERROR("Failed to get remote framebuffer size.");
        gst_object_unref(that->pipeline);
        that->muxerSignal.set_value(Result::ERROR);
        return GST_PAD_PROBE_REMOVE;
    }

    assert(static_cast<int>(that->remoteFramebufferSize.width) == width);
    assert(static_cast<int>(that->remoteFramebufferSize.height) == height);

    // Get remote framebuffer size.

    const gchar* codec = gst_structure_get_name(capsStruct);
    JST_DEBUG("Remote framebuffer codec: {}.", codec);

    Render::VideoCodec renderCodec;
    std::string plugin;

    if (strcmp(codec, "video/x-h264") == 0) {
        renderCodec = Render::VideoCodec::H264;
        plugin = "libav";
    } else if (strcmp(codec, "video/x-vp8") == 0) {
        renderCodec = Render::VideoCodec::VP8;
        plugin = "vpx";
    } else if (strcmp(codec, "video/x-vp9") == 0) {
        renderCodec = Render::VideoCodec::VP9;
        plugin = "vpx";
    } else if (strcmp(codec, "video/x-av1") == 0) {
        renderCodec = Render::VideoCodec::AV1;
        plugin = "dav1d";
    } else if (strcmp(codec, "video/x-ffv") == 0) {
        renderCodec = Render::VideoCodec::FFV1;
        plugin = "libav";
    } else {
        JST_ERROR("Unsupported remote framebuffer codec.");
        gst_object_unref(that->pipeline);
        that->muxerSignal.set_value(Result::ERROR);
        return GST_PAD_PROBE_REMOVE;
    }

    if (!gst_registry_find_plugin(gst_registry_get(), plugin.c_str())) {
        JST_ERROR("Gstreamer plugin `{}` is not available.", plugin);
        gst_object_unref(that->pipeline);
        that->muxerSignal.set_value(Result::ERROR);
        return GST_PAD_PROBE_REMOVE;
    }

    // Create pipeline.

    std::map<std::string, GstElement*> elements;
    std::vector<std::string> elementOrder = {
        "demuxer",
        "decoder",
        "convert",
        "sinkfilter",
        "sink"
    };

    if (renderCodec == Render::VideoCodec::FFV1) {
        elements["decoder"] = gst_element_factory_make("avdec_ffv1", "decoder");
    } else if (renderCodec == Render::VideoCodec::VP8) {
        elements["decoder"] = gst_element_factory_make("vp8dec", "decoder");
    } else if (renderCodec == Render::VideoCodec::VP9) {
        elements["decoder"] = gst_element_factory_make("vp9dec", "decoder");
    } else if (renderCodec == Render::VideoCodec::AV1) {
        elements["decoder"] = gst_element_factory_make("dav1ddec", "decoder");
    } else if (renderCodec == Render::VideoCodec::H264) {
        elements["decoder"] = gst_element_factory_make("avdec_h264", "decoder");
    }

    elements["convert"] = gst_element_factory_make("videoconvert", "convert");
    elements["sinkfilter"] = gst_element_factory_make("capsfilter", "sinkfilter");
    elements["sink"] = that->sink = gst_element_factory_make("appsink", "sink");

    for (const auto& [name, element] : elements) {
        if (!element) {
            JST_ERROR("Failed to create gstreamer element `{}`.", name);
            gst_object_unref(that->pipeline);
            that->muxerSignal.set_value(Result::ERROR);
            return GST_PAD_PROBE_REMOVE;
        }
    }

    // Configure elements.

    g_object_set(elements["sink"], "emit-signals", true, nullptr);

    // Link callbacks.

    g_signal_connect(elements["sink"], "new-sample", G_CALLBACK(OnSampleCallback), data);

    // Add elements to pipeline.

    for (const auto& [name, element] : elements) {
        if (!gst_bin_add(GST_BIN(that->pipeline), element)) {
            JST_ERROR("Failed to add gstreamer element `{}` to pipeline.", name);
            gst_object_unref(that->pipeline);
            that->muxerSignal.set_value(Result::ERROR);
            return GST_PAD_PROBE_REMOVE;
        }
    }

    // Link elements.

    elements["demuxer"] = that->demuxer;

    std::string lastElement = "demuxer";
    for (const auto& name : elementOrder) {
        if (name == "demuxer") {
            continue;
        }

        if (!gst_element_link(elements[lastElement], elements[name])) {
            JST_ERROR("Failed to link gstreamer element `{}` -> `{}`.", lastElement, name);
            gst_object_unref(that->pipeline);
            that->muxerSignal.set_value(Result::ERROR);
            return GST_PAD_PROBE_REMOVE;
        }

        lastElement = name;
    }

    // Set converter pixel format. 

    GstCaps* convertCaps = gst_caps_new_simple("video/x-raw",
                                               "format", G_TYPE_STRING, "RGBA",
                                               nullptr);
    g_object_set(elements["sinkfilter"], "caps", convertCaps, nullptr);
    gst_caps_unref(convertCaps);

    // Signal success.
    that->muxerSignal.set_value(Result::SUCCESS);

    // Start pipeline.
    gst_element_set_state(that->pipeline, GST_STATE_PLAYING);

    return GST_PAD_PROBE_REMOVE;
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

        memcpy(that->remoteFramebufferMemory.data(), map.data, map.size);

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

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Remote<D, T>::createPresent() {
    Render::Buffer::Config fillScreenVerticesConf;
    fillScreenVerticesConf.buffer = &Render::Extras::FillScreenVertices;
    fillScreenVerticesConf.elementByteSize = sizeof(float);
    fillScreenVerticesConf.size = 12;
    fillScreenVerticesConf.target = Render::Buffer::Target::VERTEX;
    JST_CHECK(window->build(fillScreenVerticesBuffer, fillScreenVerticesConf));

    Render::Buffer::Config fillScreenTextureVerticesConf;
    fillScreenTextureVerticesConf.buffer = &Render::Extras::FillScreenTextureVerticesXYFlip;
    fillScreenTextureVerticesConf.elementByteSize = sizeof(float);
    fillScreenTextureVerticesConf.size = 8;
    fillScreenTextureVerticesConf.target = Render::Buffer::Target::VERTEX;
    JST_CHECK(window->build(fillScreenTextureVerticesBuffer, fillScreenTextureVerticesConf));

    Render::Buffer::Config fillScreenIndicesConf;
    fillScreenIndicesConf.buffer = &Render::Extras::FillScreenIndices;
    fillScreenIndicesConf.elementByteSize = sizeof(uint32_t);
    fillScreenIndicesConf.size = 6;
    fillScreenIndicesConf.target = Render::Buffer::Target::VERTEX_INDICES;
    JST_CHECK(window->build(fillScreenIndicesBuffer, fillScreenIndicesConf));

    Render::Vertex::Config vertexCfg;
    vertexCfg.buffers = {
        {fillScreenVerticesBuffer, 3},
        {fillScreenTextureVerticesBuffer, 2},
    };
    vertexCfg.indices = fillScreenIndicesBuffer;
    JST_CHECK(window->build(vertex, vertexCfg));

    Render::Draw::Config drawVertexCfg;
    drawVertexCfg.buffer = vertex;
    drawVertexCfg.mode = Render::Draw::Mode::TRIANGLES;
    JST_CHECK(window->build(drawVertex, drawVertexCfg));

    Render::Texture::Config remoteFramebufferTextureCfg;
    remoteFramebufferTextureCfg.size = remoteFramebufferSize;
    remoteFramebufferTextureCfg.buffer = remoteFramebufferMemory.data();
    JST_CHECK(window->build(remoteFramebufferTexture, remoteFramebufferTextureCfg));

    Render::Program::Config programCfg;
    programCfg.shaders = ShadersPackage["framebuffer"];
    programCfg.draw = drawVertex;
    programCfg.textures = {remoteFramebufferTexture};
    JST_CHECK(window->build(program, programCfg));

    Render::Texture::Config textureCfg;
    textureCfg.size = config.viewSize;
    JST_CHECK(window->build(texture, textureCfg));

    Render::Surface::Config surfaceCfg;
    surfaceCfg.framebuffer = texture;
    surfaceCfg.programs = {program};
    JST_CHECK(window->build(surface, surfaceCfg));
    JST_CHECK(window->bind(surface));

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Remote<D, T>::destroyPresent() {
    JST_CHECK(window->unbind(surface));

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Remote<D, T>::present() {
    if (socketConnected) {
        JST_CHECK(remoteFramebufferTexture->fill());
    }

    if (brokerEndpointRunning) {
        if ((frameCounter % static_cast<U64>(remoteFramerate)) == 0) {
            static char buffer[8] = "ping\n";
            lastPingTime = std::chrono::high_resolution_clock::now();
            send(brokerEndpointFileDescriptor, buffer, 5, 0);
        }
    }

    frameCounter += 1;

    return Result::SUCCESS;
}

template<Device D, typename T>
void Remote<D, T>::registerMousePos(const F32& x, const F32& y) {
    if (socketConnected) {
        const I32 X = static_cast<I32>((x / config.viewSize.width) * remoteFramebufferSize.width);
        const I32 Y = static_cast<I32>((y / config.viewSize.height) * remoteFramebufferSize.height);

        static char buffer[32];
        snprintf(buffer, 32, "hid:mouse:pos:%05d,%05d\n", X, Y);
        send(brokerEndpointFileDescriptor, buffer, 26, 0);
    }
}

template<Device D, typename T>
void Remote<D, T>::registerMouseButton(const I32& button, const bool& pressed) {
    if (socketConnected) {
        static char buffer[32];
        snprintf(buffer, 32, "hid:mouse:btn:%01d,%01d\n", button, ((pressed) ? 1 : 0));
        send(brokerEndpointFileDescriptor, buffer, 18, 0);
    }
}

template<Device D, typename T>
void Remote<D, T>::registerMouseScroll(const F32& deltaX, const F32& deltaY) {
    if (socketConnected) {
        const I32 DeltaX = static_cast<I32>(deltaX * 100.0f);
        const I32 DeltaY = static_cast<I32>(deltaY * 100.0f);

        static char buffer[32];
        snprintf(buffer, 32, "hid:mouse:scroll:%+06d,%+06d\n", DeltaX, DeltaY);
        send(brokerEndpointFileDescriptor, buffer, 31, 0);
    }
}

template<Device D, typename T>
void Remote<D, T>::registerKey(const I32& key, const bool& pressed) {
    if (socketConnected) {
        static char buffer[16];
        snprintf(buffer, 16, "hid:key:%04d,%01d\n", key, ((pressed) ? 1 : 0));
        send(brokerEndpointFileDescriptor, buffer, 15, 0);
    }
}

template<Device D, typename T>
void Remote<D, T>::registerChar(const char& key) {
    if (socketConnected) {
        static char buffer[16];
        snprintf(buffer, 16, "hid:char:%c\n", key);
        send(brokerEndpointFileDescriptor, buffer, 11, 0);
    }
}

template<Device D, typename T>
const Size2D<U64>& Remote<D, T>::viewSize(const Size2D<U64>& viewSize) {
    Size2D<U64> correctedViewSize = viewSize;

    // Correct aspect ratio.

    const F32 nativeAspectRatio = static_cast<F32>(remoteFramebufferSize.width) / 
                                  static_cast<F32>(remoteFramebufferSize.height);

    const F32 viewAspectRatio = static_cast<F32>(viewSize.width) / 
                                static_cast<F32>(viewSize.height);

    if (viewAspectRatio > nativeAspectRatio) {
        correctedViewSize.width = correctedViewSize.height * nativeAspectRatio;
    } else {
        correctedViewSize.height = correctedViewSize.width / nativeAspectRatio;
    }

    // Submit new size.

    if (surface->size(correctedViewSize) != config.viewSize) {
        config.viewSize = surface->size();
    }
    return config.viewSize;
}

template<Device D, typename T>
Render::Texture& Remote<D, T>::getTexture() {
    return *texture;
};

template class Remote<Device::CPU, void>;

}  // namespace Jetstream
