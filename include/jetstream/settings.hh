#ifndef JETSTREAM_SETTINGS_HH
#define JETSTREAM_SETTINGS_HH

#include <optional>
#include <string>
#include <vector>

#include "jetstream/parser.hh"

namespace Jetstream {

struct JETSTREAM_API Settings {
 public:
    struct Size {
        U64 width = 1920;
        U64 height = 1080;

        JST_SERDES(width, height);
    };

    struct Graphics {
        std::optional<DeviceType> device{};
        U64 deviceId = 0;
        bool headless = false;
        Size size;
        F32 scale = 1.0f;
        U64 framerate = 60;

        JST_SERDES(device, size, scale, framerate);
    };

    struct Remote {
        struct Access {
            bool autoApprove = false;

            JST_SERDES(autoApprove);
        };

        struct Video {
            U64 framerate = 30;
            std::string codec = "h264";
            std::string encoder = "auto";

            JST_SERDES(framerate, codec, encoder);
        };

        struct Broker {
            std::string address = "0.0.0.0";
            U64 port = 8686;
            std::string publicUrl;
            std::string certificate;
            std::string key;

            JST_SERDES(address, port, publicUrl, certificate, key);
        };

        struct Media {
            std::string transport = "auto";
            U64 minPort = 0;
            U64 maxPort = 0;
            std::string address;

            JST_SERDES(transport, minPort, maxPort, address);
        };

        Access access;
        Video video;
        Broker broker;
        Media media;

        JST_SERDES(access, video, broker, media);
    };

    struct Interface {
        std::string themeKey = "Dark";
        bool infoPanelEnabled = true;
        bool backgroundParticles = true;

        JST_SERDES(themeKey, infoPanelEnabled, backgroundParticles);
    };

    struct Developer {
        I32 logLevel = JST_LOG_DEBUG_DEFAULT_LEVEL;
        bool latencyEnabled = false;
        bool timingEnabled = false;

        JST_SERDES(logLevel, latencyEnabled, timingEnabled);
    };

    struct Benchmark {
        std::string format = "markdown";

        JST_SERDES(format);
    };

    struct Registry {
        std::vector<std::string> plugins;

        JST_SERDES(plugins);
    };

    struct Runtime {
        struct Python {
            std::string path;

            JST_SERDES(path);
        };

        Python python;
        std::string dependencyPolicy = "prompt";

        JST_SERDES(python, dependencyPolicy);
    };

    Graphics graphics;
    Remote remote;
    Interface interface;
    Developer developer;
    Benchmark benchmark;
    Registry registry;
    Runtime runtime;

    JST_SERDES(graphics, remote, interface, developer, registry, runtime);

    static Result Get(Settings& settings);
    static Result Set(const Settings& settings, bool persist = true);
    static Result SetAsync(const Settings& settings);
    static std::optional<std::string> TakePersistenceError();
    static Result Flush();

 private:
    struct Impl;
};

}  // namespace Jetstream

#endif  // JETSTREAM_SETTINGS_HH
