#include "jetstream/viewport/plugins/endpoint.hh"
#include "jetstream/backend/devices/cpu/helpers.hh"

#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#ifdef JETSTREAM_LOADER_GSTREAMER_AVAILABLE
#include <gst/app/gstappsrc.h>
#include <gst/video/video-event.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#endif

#include <map>
#include <regex>
#include <string>

namespace Jetstream::Viewport {

Endpoint::Type Endpoint::DetermineEndpointType(const std::string& endpoint) {
    // Check for socket.
    std::regex ip_pattern(R"((?:\w+://)?(?:[a-zA-Z0-9.-]+|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?::\d{1,5})?)");
    if (std::regex_match(endpoint, ip_pattern)) {
        return Endpoint::Type::Socket;
    }

    // Check for file.
    size_t last_dot = endpoint.find_last_of(".");
    size_t last_sep = endpoint.find_last_of("/\\");
    if (last_dot != std::string::npos && (last_sep == std::string::npos || last_dot > last_sep)) {
        return Endpoint::Type::File;
    }

    // Check for pipe.
    if (last_dot == std::string::npos) {
        return Endpoint::Type::Pipe;
    }

    return Endpoint::Type::Unknown;
}

Result Endpoint::create(const Viewport::Config& _config, const Device& _viewport_device) {
    JST_DEBUG("[ENDPOINT] Initializing plugin.");

    // Set variables.

    config = _config;
    viewportDevice = _viewport_device;

    // Check if endpoint is valid.

    if (config.endpoint.empty()) {
        JST_ERROR("[ENDPOINT] Invalid endpoint.");
        return Result::ERROR;
    }

    // Check if endpoint is a file, pipe, or socket.

    type = DetermineEndpointType(config.endpoint);

    if (type == Endpoint::Type::Unknown) {
        JST_ERROR("[ENDPOINT] Unknown endpoint type.");
        return Result::ERROR;
    }

#ifndef JST_OS_WINDOWS
    if (type == Endpoint::Type::Pipe) {
        JST_DEBUG("[ENDPOINT] Endpoint is a pipe.");
        JST_CHECK(createPipeEndpoint());
        return Result::SUCCESS;
    }
#endif

#ifdef JETSTREAM_LOADER_GSTREAMER_AVAILABLE
    JST_CHECK(createGstreamerEndpoint());

    if (type == Endpoint::Type::File) {
        JST_DEBUG("[ENDPOINT] Endpoint is a file.");
        JST_CHECK(createFileEndpoint());
        return Result::SUCCESS;
    }

    if (type == Endpoint::Type::Socket) {
        JST_DEBUG("[ENDPOINT] Endpoint is a socket.");
        JST_CHECK(createSocketEndpoint());
        return Result::SUCCESS;
    }
#endif

    JST_ERROR("[ENDPOINT] Endpoint type is not available.");
    return Result::ERROR;
}

Result Endpoint::destroy() {
    JST_DEBUG("[ENDPOINT] Destroying plugin.");

    // Destroy endpoints.

#ifndef JST_OS_WINDOWS
    if (type == Endpoint::Type::Pipe) {
        JST_CHECK(destroyPipeEndpoint());
        return Result::SUCCESS;
    }
#endif

#ifdef JETSTREAM_LOADER_GSTREAMER_AVAILABLE
    if (type == Endpoint::Type::File) {
        JST_CHECK(destroyFileEndpoint());
        return Result::SUCCESS;
    }

    if (type == Endpoint::Type::Socket) {
        JST_CHECK(destroySocketEndpoint());
        return Result::SUCCESS;
    }

    JST_CHECK(destroyGstreamerEndpoint());
#endif

    return Result::SUCCESS;
}

#ifndef JST_OS_WINDOWS

Result Endpoint::createPipeEndpoint() {
    JST_INFO("[ENDPOINT] Creating pipe endpoint ({}).", config.endpoint);

    // Check if pipe exists, if not, create it.

    if (access(config.endpoint.c_str(), F_OK) == -1) {
        JST_DEBUG("[ENDPOINT] Pipe does not exist, creating it.");
        pipeCreated = true;

        if (mkfifo(config.endpoint.c_str(), 0666) < 0) {
            JST_ERROR("[ENDPOINT] Failed to create pipe.");
            return Result::ERROR;
        }
    }

    // Open pipe.

    pipeFileDescriptor = open(config.endpoint.c_str(), O_WRONLY);
    if (pipeFileDescriptor < 0) {
        JST_ERROR("[ENDPOINT] Failed to open pipe.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result Endpoint::destroyPipeEndpoint() {
    JST_DEBUG("[ENDPOINT] Destroying pipe endpoint.");

    // Close pipe, and if it was created, delete it.

    if (pipeFileDescriptor > 0) {
        close(pipeFileDescriptor);
    }

    if (pipeCreated) {
        unlink(config.endpoint.c_str());
    }

    pipeCreated = false;

    return Result::SUCCESS;
}

#endif

#ifdef JETSTREAM_LOADER_GSTREAMER_AVAILABLE

Result Endpoint::createSocketEndpoint() {
    JST_INFO("[ENDPOINT] Creating socket endpoint ({}).", config.endpoint);

    // Get broker address and port with regex.

    std::smatch matches;
    std::regex pattern(R"(([\w.-]+|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d{1,5}))");
    if (std::regex_match(config.endpoint, matches, pattern)) {
        if (matches.size() == 3) {
            brokerAddress = matches[1].str();
            brokerPort = std::stoi(matches[2].str());
        }
    } else {
        JST_ERROR("[ENDPOINT] Invalid endpoint format. Expected `address:port`. Example: `10.10.1.1:5000`.");
        return Result::ERROR;
    }

    // Create TCP socket server.

    brokerEndpointFileDescriptor = socket(AF_INET, SOCK_STREAM, 0);
    if (brokerEndpointFileDescriptor < 0) {
        JST_ERROR("[ENDPOINT] Failed to create broker endpoint.");
        return Result::ERROR;
    }

    // Set socket options.

    int keepalive = 1;
    if (setsockopt(brokerEndpointFileDescriptor, SOL_SOCKET, SO_KEEPALIVE, &keepalive, sizeof(keepalive)) < 0) {
        JST_ERROR("[ENDPOINT] Failed to set socket keep-alive option on broker endpoint.");
        return Result::ERROR;
    }

    int reuse = 1;
    if (setsockopt(brokerEndpointFileDescriptor, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        JST_ERROR("[ENDPOINT] Failed to set socket reuse option on broker endpoint.");
        return Result::ERROR;
    }

    int delay = 1;
    if (setsockopt(brokerEndpointFileDescriptor, IPPROTO_TCP, TCP_NODELAY, &delay, sizeof(delay)) < 0) {
        JST_ERROR("[ENDPOINT] Failed to set socket no delay option on broker endpoint.");
        return Result::ERROR;
    }

    // Bind socket to port.

    struct sockaddr_in brokerEndpointAddress = {};
    brokerEndpointAddress.sin_family = AF_INET;
    brokerEndpointAddress.sin_addr.s_addr = inet_addr(brokerAddress.c_str());
    brokerEndpointAddress.sin_port = htons(brokerPort);

    if (bind(brokerEndpointFileDescriptor, (struct sockaddr*)&brokerEndpointAddress, sizeof(brokerEndpointAddress)) < 0) {
        JST_ERROR("[ENDPOINT] Failed to bind broker endpoint.");
        return Result::ERROR;
    }

    // Listen for connections.

    if (listen(brokerEndpointFileDescriptor, 5) < 0) {
        JST_ERROR("[ENDPOINT] Failed to listen for connections on broker endpoint.");
        return Result::ERROR;
    }

    brokerEndpointRunning = true;
    brokerEndpointThread = std::thread([&]() {
        while (brokerEndpointRunning) {
            // Accept connection.

            brokerClientFileDescriptor = accept(brokerEndpointFileDescriptor, nullptr, nullptr);
            if (brokerClientFileDescriptor < 0) {
                continue;
            }

            while (true) {
                // Read data.

                char buffer[1024] = {0};
                
                if (read(brokerClientFileDescriptor, buffer, sizeof(buffer)) < 0) {
                    break;
                }

                std::string line;
                std::istringstream lineStream(buffer);
                while (std::getline(lineStream, line)) {
                    // Parse command `ping`.

                    if (line.compare(0, 4, "ping") == 0) {
                        auto response = fmt::format("pong\n");
                        send(brokerClientFileDescriptor, response.c_str(), response.size(), 0);
                        continue;
                    }

                    JST_TRACE("[ENDPOINT] Received message from client: '{}'.", line);

                    // Parse command `hid:mouse:pos:{x},{y}`.

                    if (line.compare(0, 13, "hid:mouse:pos") == 0) {
                        const auto x = std::stoi(line.substr(14, 19));
                        const auto y = std::stoi(line.substr(20, 25));
                        JST_TRACE("[ENDPOINT] Received `hid:mouse:pos` from client: `x={}, y={}`.", x, y);

                        // TODO: This should be submitted before the frame begin.
                        ImGui::GetIO().AddMousePosEvent(x, y);

                        continue;
                    }

                    // Parse command `hid:mouse:btn:{btn},{down}`.

                    if (line.compare(0, 13, "hid:mouse:btn") == 0) {
                        const auto button = std::stoi(line.substr(14, 15));
                        const auto down = std::stoi(line.substr(16, 17));
                        JST_TRACE("[ENDPOINT] Received `hid:mouse:btn` from client: `button={}, down={}`.", button, down);

                        // TODO: This should be submitted before the frame begin.
                        ImGui::GetIO().AddMouseButtonEvent(button, down);

                        continue;
                    }

                    // Parse command `hid:mouse:scroll:{x},{y}`.

                    if (line.compare(0, 16, "hid:mouse:scroll") == 0) {
                        const auto x = std::stoi(line.substr(17, 23)) / 100.0f;
                        const auto y = std::stoi(line.substr(24, 29)) / 100.0f;
                        JST_TRACE("[ENDPOINT] Received `hid:mouse:scroll` from client: `x={}, y={}`.", x, y);

                        // TODO: This should be submitted before the frame begin.
                        ImGui::GetIO().MouseWheel = x;
                        ImGui::GetIO().MouseWheelH = y;

                        continue;
                    }

                    // Parse command `hid:key:{id},{pressed}`.

                    if (line.compare(0, 7, "hid:key") == 0) {
                        const auto id = std::stoi(line.substr(8, 12));
                        const auto pressed = std::stoi(line.substr(13, 14));
                        JST_TRACE("[ENDPOINT] Received `hid:key` from client: `id={}, pressed={}`.", id, pressed);

                        // TODO: This should be submitted before the frame begin.
                        ImGui::GetIO().AddKeyEvent(static_cast<ImGuiKey>(id), pressed);

                        continue;
                    }

                    // Parse command `hid:char:{key}`.

                    if (line.compare(0, 8, "hid:char") == 0) {
                        const char key = line.substr(9, 10).c_str()[0];
                        JST_TRACE("[ENDPOINT] Received `hid:char` from client: `key={}`.", key);

                        // TODO: This should be submitted before the frame begin.
                        ImGui::GetIO().AddInputCharacter(key);

                        continue;
                    }

                    // Parse command `cmd`.

                    if (line.compare(0, 3, "cmd") == 0) {
                        // Parse command `cmd:connect`.

                        if (line.compare(0, 11, "cmd:connect") == 0) {
                            // Check if socket is already connected.

                            if (socketConnected) {
                                JST_ERROR("[ENDPOINT] Socket is already connected.");
                                auto response = fmt::format("err:Socket is already connected.\n");
                                send(brokerClientFileDescriptor, response.c_str(), response.size(), 0);
                                continue;
                            }

                            // Get client address.

                            struct sockaddr_in brokerEndpointClientAddress = {};
                            socklen_t brokerEndpointClientAddressLength = sizeof(brokerEndpointClientAddress);
                            if (getpeername(brokerClientFileDescriptor, (struct sockaddr*)&brokerEndpointClientAddress, &brokerEndpointClientAddressLength) < 0) {
                                JST_ERROR("[ENDPOINT] Failed to get client address from broker endpoint.");
                                auto response = fmt::format("err:Failed to get client address.\n");
                                send(brokerClientFileDescriptor, response.c_str(), response.size(), 0);
                                continue;
                            }

                            // Get client address string.

                            char brokerEndpointClientAddressString[INET_ADDRSTRLEN];
                            inet_ntop(AF_INET, &brokerEndpointClientAddress.sin_addr, brokerEndpointClientAddressString, INET_ADDRSTRLEN);
                            socketAddress = brokerEndpointClientAddressString;
                            socketPort = brokerPort + 1;
                            socketConnected = true;

                            // Create Gstreamer endpoint.

                            if (startGstreamerEndpoint() != Result::SUCCESS) {
                                auto response = fmt::format("err:{}\n", JST_LOG_LAST_ERROR());
                                send(brokerClientFileDescriptor, response.c_str(), response.size(), 0);
                                continue;
                            }

                            // Send response.

                            {
                                auto response = fmt::format("ok:connect\n");
                                send(brokerClientFileDescriptor, response.c_str(), response.size(), 0);
                            }

                            continue;
                        }

                        // Parse command `cmd:disconnect`.

                        if (line.compare(0, 14, "cmd:disconnect") == 0) {
                            // Check if socket is already disconnected.

                            if (!socketConnected) {
                                JST_ERROR("[ENDPOINT] Socket is already disconnected.");
                                auto response = fmt::format("err:Socket is already disconnected.\n");
                                send(brokerClientFileDescriptor, response.c_str(), response.size(), 0);
                                continue;
                            }

                            // Send response.

                            {
                                auto response = fmt::format("ok:disconnect\n", socketAddress, socketPort);
                                send(brokerClientFileDescriptor, response.c_str(), response.size(), 0);
                            }

                            close(brokerClientFileDescriptor);
                            break;
                        }

                        // Parse command `cmd:fbsize`.

                        if (line.compare(0, 10, "cmd:fbsize") == 0) {
                            auto response = fmt::format("ok:fbsize:{:05},{:05}\n", config.size.width, config.size.height);
                            send(brokerClientFileDescriptor, response.c_str(), response.size(), 0);

                            continue;
                        }

                        // Parse command `cmd:framerate`.

                        if (line.compare(0, 13, "cmd:framerate") == 0) {
                            auto response = fmt::format("ok:framerate:{}\n", config.framerate);
                            send(brokerClientFileDescriptor, response.c_str(), response.size(), 0);

                            continue;
                        }

                        // Parse command `cmd:codec`.

                        if (line.compare(0, 9, "cmd:codec") == 0) {
                            auto response = fmt::format("ok:codec:{}\n", Render::VideoCodecToString(config.codec));
                            send(brokerClientFileDescriptor, response.c_str(), response.size(), 0);

                            continue;
                        }
                    }

                    // Parse command `err`.

                    if (line.compare(0, 3, "err") == 0) {
                        JST_ERROR("[ENDPOINT] Received `err` from client: '{}'.", line);
                        continue;
                    }

                    // Parse command `ok`.

                    if (line.compare(0, 2, "ok") == 0) {
                        JST_DEBUG("[ENDPOINT] Received `ok` from client: '{}'.", line);
                        continue;
                    }

                    // Reply for unrecognized message.

                    {
                        JST_ERROR("[ENDPOINT] Received unrecognized message from client: '{}'.", line);
                        auto response = fmt::format("err:Unrecognized message.\n");
                        send(brokerClientFileDescriptor, response.c_str(), response.size(), 0);
                        continue;
                    }
                }
            }

            // Close client socket.
            
            JST_INFO("[ENDPOINT] Client disconnected. Closing socket.");
            close(brokerClientFileDescriptor);
            stopGstreamerEndpoint();
            socketConnected = false;
        }
    });

    return Result::SUCCESS;
}

Result Endpoint::destroySocketEndpoint() {
    JST_DEBUG("[ENDPOINT] Destroying socket endpoint.");

    if (socketConnected) {
        socketConnected = false;
        auto response = fmt::format("err:Server is closing down.\n");
        send(brokerClientFileDescriptor, response.c_str(), response.size(), 0);
        close(brokerClientFileDescriptor);
    }

    if (brokerEndpointRunning) {
        brokerEndpointRunning = false;
        shutdown(brokerEndpointFileDescriptor, SHUT_RD);
        close(brokerEndpointFileDescriptor);
        if (brokerEndpointThread.joinable()) {
            brokerEndpointThread.join();
        }
    }

    JST_CHECK(stopGstreamerEndpoint());
    
    return Result::SUCCESS;
}

Result Endpoint::createGstreamerEndpoint() {
    JST_DEBUG("[ENDPOINT] Creating gstreamer endpoint.");

    // Initialize gstreamer.

    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
    }

    // Check if all plugins are available.

    std::vector<std::string> plugins = {
        "app",
        "rawparse",
        "coreelements",
    };

    // Inject socket endpoint plugins.

    if (type == Endpoint::Type::Socket) {
        plugins.push_back("rtp");
        plugins.push_back("udp");
    }

    // Inject file endpoint plugins.

    if (type == Endpoint::Type::File) {
        plugins.push_back("matroska");
    }

    // Check required core plugins.

    JST_CHECK(checkGstreamerPlugins(plugins));

    // Inject codec plugins.

    std::vector<std::tuple<Device, Strategy, std::vector<std::string>>> combinations;

    if (config.codec == Render::VideoCodec::H264) {
#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
        if (viewportDevice == Device::Vulkan && Backend::State<Device::CUDA>()->isAvailable()) {
            combinations.push_back({Device::CUDA, Strategy::HardwareNVENC, {"nvcodec"}});
            GST_DEBUG("[ENDPOINT] Checking for NVENC strategy support for h264.");
        }
#endif

        if (checkGstreamerPlugins({"video4linux2"}, true) == Result::SUCCESS) {
            GstElementFactory* factory = gst_element_factory_find("v4l2h264enc");
            if (factory) {
                combinations.push_back({Device::CPU, Strategy::HardwareV4L2, {"video4linux2"}});
                gst_object_unref(GST_OBJECT(factory));
                GST_DEBUG("[ENDPOINT] Checking for V4L2 strategy support for h264.");
            }
        }

        combinations.push_back({Device::CPU, Strategy::Software, {"x264"}});
    }

    if (config.codec == Render::VideoCodec::FFV1) {
        combinations.push_back({Device::CPU, Strategy::Software, {"libav"}});
    }

    if (config.codec == Render::VideoCodec::VP8) {
        combinations.push_back({Device::CPU, Strategy::Software, {"vpx"}});
    }

    if (config.codec == Render::VideoCodec::VP9) {
        combinations.push_back({Device::CPU, Strategy::Software, {"vpx"}});
    }

    if (config.codec == Render::VideoCodec::AV1) {
        combinations.push_back({Device::CPU, Strategy::Software, {"rav1e"}});
    }

    for (const auto& [device, strategy, plugins] : combinations) {
        if ((strategy != Strategy::Software) && !config.hardwareAcceleration) {
            continue;
        }
        if (checkGstreamerPlugins(plugins, true) == Result::SUCCESS) {
            _inputMemoryDevice = device;
            _encodingStrategy = strategy;

            JST_INFO("[ENDPOINT] Using {} encoding with {} memory.", StrategyToString(strategy),
                                                                     GetDevicePrettyName(device));

            return Result::SUCCESS;
        }
        JST_DEBUG("[ENDPOINT] Failed to find plugins: {}", plugins);
    }

    JST_ERROR("[ENDPOINT] No encoding combination is available.");
    JST_ERROR("[ENDPOINT] This is tipically caused by missing plugins.");
    return Result::ERROR;
}

Result Endpoint::destroyGstreamerEndpoint() {
    JST_DEBUG("[ENDPOINT] Destroying gstreamer endpoint.");

    _encodingStrategy = Strategy::None;
    _inputMemoryDevice = Device::None;

    return Result::SUCCESS;
}

Result Endpoint::checkGstreamerPlugins(const std::vector<std::string>& plugins, 
                                       const bool& silent) {
    for (const auto& plugin : plugins) {
        if (!gst_registry_find_plugin(gst_registry_get(), plugin.c_str())) {
            if (!silent) {
                JST_ERROR("[ENDPOINT] Gstreamer plugin '{}' is not available.", plugin);
            }
            return Result::ERROR;
        }
    }

    return Result::SUCCESS;
}

Result Endpoint::startGstreamerEndpoint() {
    JST_DEBUG("[ENDPOINT] Starting gstreamer endpoint.");

    // Create pipeline.

    pipeline = gst_pipeline_new("headless-src-pipeline");

    if (!pipeline) {
        JST_ERROR("[ENDPOINT] Failed to create gstreamer pipeline.");
        return Result::ERROR;
    }

    // Create elements.

    std::map<std::string, GstElement*> elements;
    std::vector<std::string> elementOrder;

    elements["source"] = source = gst_element_factory_make("appsrc", "source");
    elementOrder.push_back("source");

    elements["caps"] = gst_element_factory_make("capsfilter", "caps");
    elementOrder.push_back("caps");

    if (_encodingStrategy == Strategy::Software || _encodingStrategy == Strategy::HardwareV4L2) {
        elements["rawparser"] = gst_element_factory_make("rawvideoparse", "rawparser");
        elementOrder.push_back("rawparser");

        elements["convert"] = gst_element_factory_make("videoconvert", "convert");
        elementOrder.push_back("convert");
    }

    if (config.codec == Render::VideoCodec::H264) {
        if (_encodingStrategy == Strategy::HardwareNVENC) {
            elements["encoder"] = encoder = gst_element_factory_make("nvh264enc", "encoder");
            elementOrder.push_back("encoder");

            elements["parser"] = gst_element_factory_make("h264parse", "parser");
            elementOrder.push_back("parser");
        }

        if (_encodingStrategy == Strategy::HardwareV4L2) {
            elements["encoder"] = encoder = gst_element_factory_make("v4l2h264enc", "encoder");
            elementOrder.push_back("encoder");

            elements["parser"] = gst_element_factory_make("h264parse", "parser");
            elementOrder.push_back("parser");

            elements["hwcaps"] = gst_element_factory_make("capsfilter", "hwcaps");
            elementOrder.push_back("hwcaps");
        }

        if (_encodingStrategy == Strategy::Software) {
            elements["encoder"] = encoder = gst_element_factory_make("x264enc", "encoder");
            elementOrder.push_back("encoder");
        }
    }

    if (config.codec == Render::VideoCodec::FFV1) {
        elements["encoder"] = encoder = gst_element_factory_make("avenc_ffv1", "encoder");
        elementOrder.push_back("encoder");
    }

    if (config.codec == Render::VideoCodec::VP8) {
        elements["encoder"] = encoder = gst_element_factory_make("vp8enc", "encoder");
        elementOrder.push_back("encoder");
    }

    if (config.codec == Render::VideoCodec::VP9) {
        elements["encoder"] = encoder = gst_element_factory_make("vp9enc", "encoder");
        elementOrder.push_back("encoder");
    }

    if (config.codec == Render::VideoCodec::AV1) {
        elements["encoder"] = encoder = gst_element_factory_make("rav1enc", "encoder");
        elementOrder.push_back("encoder");
    }

    if (type == Endpoint::Type::Socket) {
        elements["muxer"] = gst_element_factory_make("rtpgstpay", "muxer");
        elementOrder.push_back("muxer");

        elements["sink"] = gst_element_factory_make("udpsink", "sink");
        elementOrder.push_back("sink");
    }

    if (type == Endpoint::Type::File) {
        if (fileExtension == "mkv") {
            elements["muxer"] = gst_element_factory_make("matroskamux", "muxer");
            elementOrder.push_back("muxer");
        }

        elements["sink"] = gst_element_factory_make("filesink", "sink");
        elementOrder.push_back("sink");
    }
    
    for (const auto& [name, element] : elements) {
        if (!element) {
            JST_ERROR("[ENDPOINT] Failed to create gstreamer element '{}'.", name);
            gst_object_unref(pipeline);
            return Result::ERROR;
        }
    }

    // Configure elements.

    g_object_set(elements["source"], "block", true, nullptr);
    g_object_set(elements["source"], "format", 3, nullptr);
    g_object_set(elements["source"], "leaky-type", 2, nullptr);
    g_object_set(elements["source"], "is-live", true, nullptr);
    g_object_set(elements["source"], "max-bytes", 2*config.size.width*config.size.height*4, nullptr);

    GstCaps* caps = gst_caps_new_simple("video/x-raw",
                                        "format", G_TYPE_STRING, "BGRA",
                                        "width", G_TYPE_INT, config.size.width,
                                        "height", G_TYPE_INT, config.size.height,
                                        "framerate", GST_TYPE_FRACTION, config.framerate, 1,
                                        "interlace-mode", G_TYPE_STRING, "progressive",
                                        nullptr);
    
    if (_encodingStrategy == Strategy::HardwareNVENC && _inputMemoryDevice == Device::CUDA) {
        GstCapsFeatures *features = gst_caps_features_new("memory:CUDAMemory", NULL);
        gst_caps_set_features(caps, 0, features);
    }

    g_object_set(elements["caps"], "caps", caps, nullptr);
    gst_caps_unref(caps);

    if (_encodingStrategy == Strategy::Software || _encodingStrategy == Strategy::HardwareV4L2) {
        g_object_set(elements["rawparser"], "use-sink-caps", 0, nullptr);
        g_object_set(elements["rawparser"], "format", 12, nullptr);
        g_object_set(elements["rawparser"], "width", config.size.width, nullptr);
        g_object_set(elements["rawparser"], "height", config.size.height, nullptr);
        g_object_set(elements["rawparser"], "framerate", 1.0f/config.framerate, nullptr);
    }

    if (config.codec == Render::VideoCodec::H264) {
        if (_encodingStrategy == Strategy::HardwareNVENC) {
            g_object_set(elements["encoder"], "zerolatency", true, nullptr);
            g_object_set(elements["encoder"], "preset", 5, nullptr);
        }

        if (_encodingStrategy == Strategy::HardwareV4L2) {
            GstCaps* hwcaps = gst_caps_new_simple("video/x-h264",
                                                  "level", G_TYPE_STRING, "5",
                                                  "profile", G_TYPE_STRING, "main",
                                                  nullptr);
            g_object_set(elements["hwcaps"], "caps", hwcaps, nullptr);
            gst_caps_unref(hwcaps);
        }

        if (_encodingStrategy == Strategy::Software) {
            g_object_set(elements["encoder"], "speed-preset", 1, nullptr);
            g_object_set(elements["encoder"], "tune", 4, nullptr);
            g_object_set(elements["encoder"], "bitrate", 25*1024, nullptr);
        }
    }

    if (config.codec == Render::VideoCodec::AV1) {
        g_object_set(elements["encoder"], "low-latency", true, nullptr);
        g_object_set(elements["encoder"], "speed-preset", 10, nullptr);
        g_object_set(elements["encoder"], "bitrate", 25*1024*1024, nullptr);
    }

    if (config.codec == Render::VideoCodec::VP8 ||
        config.codec == Render::VideoCodec::VP9) {
        g_object_set(elements["encoder"], "target-bitrate", 25*1024*1024, nullptr);
    }

    if (type == Endpoint::Type::Socket) {
        g_object_set(elements["muxer"], "config-interval", 1, nullptr);

        g_object_set(elements["sink"], "sync", false, nullptr);
        g_object_set(elements["sink"], "buffer-size", Backend::GetSocketBufferSize(), nullptr);
        g_object_set(elements["sink"], "host", socketAddress.c_str(), "port", socketPort, nullptr);
    } 

    if (type == Endpoint::Type::File) {
        g_object_set(elements["sink"], "location", config.endpoint.c_str(), nullptr);
    }

    // Add elements to pipeline.

    for (const auto& [name, element] : elements) {
        if (!gst_bin_add(GST_BIN(pipeline), element)) {
            JST_ERROR("[ENDPOINT] Failed to add gstreamer element '{}' to pipeline.", name);
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
            JST_ERROR("[ENDPOINT] Failed to link gstreamer element '{}' -> '{}'.", lastElement, name);
            gst_object_unref(pipeline);
            return Result::ERROR;
        }

        lastElement = name;
    }

    // Start pipeline.

    if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        JST_ERROR("[ENDPOINT] Failed to start gstreamer pipeline.");
        gst_object_unref(pipeline);
        return Result::ERROR;
    }

    initialFrameTime = std::chrono::steady_clock::now();
    forceKeyframe = true;
    socketStreaming = true;

    return Result::SUCCESS;
}

Result Endpoint::stopGstreamerEndpoint() {
    JST_DEBUG("[ENDPOINT] Stopping gstreamer endpoint.");

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

        // Cleanup.
        gst_object_unref(pipeline);
    }

    return Result::SUCCESS;
}

Result Endpoint::createFileEndpoint() {
    JST_DEBUG("[ENDPOINT] Creating file endpoint ({}).", config.endpoint);

    // Get file extension with regex.

    std::smatch matches;
    std::regex pattern(R"(^\.?\/?[^\/\\]*\.([\w]+)$)");
    if (std::regex_match(config.endpoint, matches, pattern)) {
        if (matches.size() == 2) {
            fileExtension = matches[1].str();
        }
    } else {
        JST_ERROR("[ENDPOINT] Invalid endpoint format. Expected `filename.extension`. Example: `./video.mkv`.");
        return Result::ERROR;
    }

    if (fileExtension != "mkv") {
        JST_ERROR("[ENDPOINT] Invalid file extension. Expected `mkv`.");
        return Result::ERROR;
    }

    JST_CHECK(startGstreamerEndpoint());

    return Result::SUCCESS;
}

Result Endpoint::destroyFileEndpoint() {
    JST_DEBUG("[ENDPOINT] Destroying file endpoint.");

    JST_CHECK(stopGstreamerEndpoint());

    return Result::SUCCESS;
}

void Endpoint::OnBufferReleaseCallback(gpointer user_data) {
    auto* that = reinterpret_cast<Endpoint*>(user_data);
    std::unique_lock<std::mutex> lock(that->bufferMutex);
    that->bufferProcessed = true;
    that->bufferCond.notify_one();
}

#endif

Result Endpoint::pushNewFrame(const void* data) {
#ifndef JST_OS_WINDOWS
    if (type == Endpoint::Type::Pipe) {
        write(pipeFileDescriptor, data, config.size.width * config.size.height * 4);
    }
#endif

#ifdef JETSTREAM_LOADER_GSTREAMER_AVAILABLE
    if ((type == Endpoint::Type::File || type == Endpoint::Type::Socket) && socketStreaming) {
        // Create buffer.

        GstBuffer* buffer = gst_buffer_new_wrapped_full(GST_MEMORY_FLAG_READONLY,
                                                        const_cast<void*>(data),
                                                        config.size.width * config.size.height * 4,
                                                        0,
                                                        config.size.width * config.size.height * 4,
                                                        this,
                                                        &OnBufferReleaseCallback);

        // Calculate timings.

        const auto currentFrameTime = std::chrono::steady_clock::now();
        const auto elapsedSinceLastFrame = std::chrono::duration_cast<std::chrono::nanoseconds>(currentFrameTime - 
                                                                                                initialFrameTime);
        const auto elapsedSinceLastKeyframe = std::chrono::duration_cast<std::chrono::seconds>(currentFrameTime - 
                                                                                               lastKeyframeTime);

        // Set buffer timings (PTS and DTS).

        GST_BUFFER_PTS(buffer) = static_cast<U64>(elapsedSinceLastFrame.count());
        GST_BUFFER_DTS(buffer) = GST_CLOCK_TIME_NONE;

        // Force keyframe every 1 seconds.

        if ((elapsedSinceLastKeyframe.count() > 1) || forceKeyframe) {
            GstEvent* force_key_unit_event = gst_video_event_new_downstream_force_key_unit(
                GST_CLOCK_TIME_NONE,
                GST_CLOCK_TIME_NONE,
                GST_CLOCK_TIME_NONE,
                TRUE,
                0
            );

            gst_element_send_event(encoder, force_key_unit_event);

            lastKeyframeTime = currentFrameTime;
            forceKeyframe = false;
        }

        // Push frame to pipeline.

        if (gst_app_src_push_buffer(GST_APP_SRC(source), buffer) != GST_FLOW_OK) {
            JST_ERROR("[ENDPOINT] Failed to push buffer to gstreamer pipeline.");
            return Result::ERROR;
        }

        // Wait for buffer to be processed.

        {
            std::unique_lock<std::mutex> lock(bufferMutex);
            bufferCond.wait(lock, [&]{ return bufferProcessed; });
            bufferProcessed = false;
        }
    }
#endif

    return Result::SUCCESS;
}

}  // namespace Jetstream::Viewport