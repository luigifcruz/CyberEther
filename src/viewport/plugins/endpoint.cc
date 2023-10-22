#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <map>
#include <regex>
#include <string>

#include <gst/app/gstappsrc.h>

#include "jetstream/viewport/plugins/endpoint.hh"

namespace Jetstream::Viewport {

Endpoint::Type Endpoint::DetermineEndpointType(const std::string& endpoint) {
    // Check for socket.
    std::regex ip_pattern(R"((?:\w+://)?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d{1,5})?)");
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

Result Endpoint::create(const Viewport::Config& _config) {
    JST_DEBUG("[ENDPOINT] Initializing plugin.");

    config = _config;

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

    if (type == Endpoint::Type::Pipe) {
        JST_DEBUG("[ENDPOINT] Endpoint is a pipe.");
        JST_CHECK(createPipeEndpoint());
        return Result::SUCCESS;
    }

#ifdef JETSTREAM_LOADER_GSTREAMER_AVAILABLE
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

    if (type == Endpoint::Type::Pipe) {
        JST_CHECK(destroyPipeEndpoint());
    }

#ifdef JETSTREAM_LOADER_GSTREAMER_AVAILABLE
    if (type == Endpoint::Type::File) {
        JST_CHECK(destroyFileEndpoint());
    }

    if (type == Endpoint::Type::Socket) {
        JST_CHECK(destroySocketEndpoint());
    }
#endif

    return Result::SUCCESS;
}

Result Endpoint::createPipeEndpoint() {
    JST_DEBUG("[ENDPOINT] Creating pipe endpoint ({}).", config.endpoint);

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

#ifdef JETSTREAM_LOADER_GSTREAMER_AVAILABLE

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
        "videoconvertscale",
        "coreelements",
        "matroska",
    };

    // Inject TCP or UDP plugin.

    if (type == Endpoint::Type::Socket) {
        if (socketType == "tcp") {
            plugins.push_back("tcp");
        }

        if (socketType == "udp") {
            plugins.push_back("udp");
        }
    }

    // Inject codec plugins.

    if (config.codec == Render::VideoCodec::FFV1) {
        plugins.push_back("libav");
    }

    if (config.codec == Render::VideoCodec::H264) {
        plugins.push_back("x264");
    }

    if (config.codec == Render::VideoCodec::HEVC) {
        plugins.push_back("x265");
    }

    if (config.codec == Render::VideoCodec::VP9) {
        plugins.push_back("vpx");
    }

    if (config.codec == Render::VideoCodec::AV1) {
        plugins.push_back("aom");
    }

    for (const auto& plugin : plugins) {
        if (!gst_registry_find_plugin(gst_registry_get(), plugin.c_str())) {
            JST_ERROR("[ENDPOINT] Gstreamer plugin `{}` is not available.", plugin);
            return Result::ERROR;
        }
    }

    // Create pipeline.

    pipeline = gst_pipeline_new("headless-src-pipeline");

    if (!pipeline) {
        JST_ERROR("[ENDPOINT] Failed to create gstreamer pipeline.");
        return Result::ERROR;
    }

    // Create elements.

    std::map<std::string, GstElement*> elements;
    std::vector<std::string> elementOrder = {
        "source",
        "parser",
        "convert",
        "encoder",
        "muxer",
        "sink"
    };

    elements["source"] = source = gst_element_factory_make("appsrc", "source");
    elements["parser"] = gst_element_factory_make("rawvideoparse", "parser");
    elements["convert"] = gst_element_factory_make("videoconvert", "convert");

    if (config.codec == Render::VideoCodec::FFV1) {
        elements["encoder"] = gst_element_factory_make("avenc_ffv1", "encoder");
    }

    if (config.codec == Render::VideoCodec::H264) {
        elements["encoder"] = gst_element_factory_make("x264enc", "encoder");
    }

    if (config.codec == Render::VideoCodec::HEVC) {
        elements["encoder"] = gst_element_factory_make("x265enc", "encoder");
    }

    if (config.codec == Render::VideoCodec::VP9) {
        elements["encoder"] = gst_element_factory_make("vp9enc", "encoder");
    }

    if (config.codec == Render::VideoCodec::AV1) {
        elements["encoder"] = gst_element_factory_make("av1enc", "encoder");
    }

    elements["muxer"] = gst_element_factory_make("matroskamux", "muxer");

    if (type == Endpoint::Type::Socket) {
        if (socketType == "tcp") {
            elements["sink"] = gst_element_factory_make("tcpserversink", "sink");
        }

        if (socketType == "udp") {
            elements["sink"] = gst_element_factory_make("udpsink", "sink");
        }
    }

    if (type == Endpoint::Type::File) {
        elements["sink"] = gst_element_factory_make("filesink", "sink");
    }
    
    for (const auto& [name, element] : elements) {
        if (!element) {
            JST_ERROR("[ENDPOINT] Failed to create gstreamer element `{}`.", name);
            gst_object_unref(pipeline);
            return Result::ERROR;
        }
    }

    // Configure elements.

    g_object_set(elements["source"], "block", true, nullptr);
    g_object_set(elements["source"], "max-bytes", 32*1024*1024, nullptr);
    g_object_set(elements["parser"], "use-sink-caps", 0, nullptr);
    g_object_set(elements["parser"], "width", config.size.width, nullptr);
    g_object_set(elements["parser"], "height", config.size.height, nullptr);
    g_object_set(elements["parser"], "format", 12, nullptr);
    g_object_set(elements["parser"], "framerate", 0, 0, nullptr);

    if (type == Endpoint::Type::Socket) {
        g_object_set(elements["sink"], "host", socketAddress.c_str(), "port", socketPort, nullptr);
    } 

    if (type == Endpoint::Type::File) {
        g_object_set(elements["sink"], "location", config.endpoint.c_str(), nullptr);
    }

    // Add elements to pipeline.

    for (const auto& [name, element] : elements) {
        if (!gst_bin_add(GST_BIN(pipeline), element)) {
            JST_ERROR("[ENDPOINT] Failed to add gstreamer element `{}` to pipeline.", name);
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
            JST_ERROR("[ENDPOINT] Failed to link gstreamer element `{}`.", name);
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

    return Result::SUCCESS;
}

Result Endpoint::destroyGstreamerEndpoint() {
    JST_DEBUG("[ENDPOINT] Destroying gstreamer endpoint.");

    // Stop pipeline.

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);

    // De-initialize gstreamer.

    if (gst_is_initialized()) {
        gst_deinit();
    }

    return Result::SUCCESS;
}

Result Endpoint::createFileEndpoint() {
    JST_DEBUG("[ENDPOINT] Creating file endpoint ({}).", config.endpoint);

    JST_CHECK(createGstreamerEndpoint());

    return Result::SUCCESS;
}

Result Endpoint::destroyFileEndpoint() {
    JST_DEBUG("[ENDPOINT] Destroying file endpoint.");

    JST_CHECK(destroyGstreamerEndpoint());

    return Result::SUCCESS;
}

Result Endpoint::createSocketEndpoint() {
    JST_DEBUG("[ENDPOINT] Creating socket endpoint ({}).", config.endpoint);

    // Get socket address and port with regex.

    std::smatch matches;
    std::regex pattern(R"((\w+)://([\d\.]+):(\d+))");
    if (std::regex_match(config.endpoint, matches, pattern)) {
        if (matches.size() == 4) {
            socketType = matches[1].str();
            socketAddress = matches[2].str();
            socketPort = std::stoi(matches[3].str());
        }
    } else {
        JST_ERROR("[ENDPOINT] Invalid endpoint format. Expected `protocol://address:port`. Example: `tcp://10.10.1.1:5000`.");
        return Result::ERROR;
    }

    if (socketType != "tcp" && socketType != "udp") {
        JST_ERROR("[ENDPOINT] Invalid socket type. Expected `tcp` or `udp`.");
        return Result::ERROR;
    }

    JST_CHECK(createGstreamerEndpoint());

    return Result::SUCCESS;
}

Result Endpoint::destroySocketEndpoint() {
    JST_DEBUG("[ENDPOINT] Destroying socket endpoint.");

    JST_CHECK(destroyGstreamerEndpoint());

    return Result::SUCCESS;
}

#endif

Result Endpoint::newFrameHost(const uint8_t* data) {
    if (type == Endpoint::Type::Pipe) {
        write(pipeFileDescriptor, data, config.size.width * config.size.height * 4);
    }

#ifdef JETSTREAM_LOADER_GSTREAMER_AVAILABLE
    if (type == Endpoint::Type::File || type == Endpoint::Type::Socket) {
        GstBuffer* buffer = gst_buffer_new_wrapped_full(GST_MEMORY_FLAG_READONLY,
                                                        const_cast<uint8_t*>(data),
                                                        config.size.width * config.size.height * 4,
                                                        0,
                                                        config.size.width * config.size.height * 4,
                                                        nullptr,
                                                        nullptr);
        if (gst_app_src_push_buffer(GST_APP_SRC(source), buffer) != GST_FLOW_OK) {
            JST_ERROR("[ENDPOINT] Failed to push buffer to gstreamer pipeline.");
            return Result::ERROR;
        }
    }
#endif

    return Result::SUCCESS;
}

}  // namespace Jetstream::Viewport