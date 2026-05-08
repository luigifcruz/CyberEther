#ifndef JETSTREAM_SETTINGS_HH
#define JETSTREAM_SETTINGS_HH

#include <optional>
#include <string>

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
        bool headless = false;
        Size size;
        F32 scale = 1.0f;
        U64 framerate = 60;

        JST_SERDES(device, headless, size, scale, framerate);
    };

    struct Remote {
        std::string brokerUrl = "https://cyberether.org";
        std::string codec = "h264";
        std::string encoder = "auto";
        bool autoJoinSessions = false;
        U64 framerate = 30;

        JST_SERDES(brokerUrl, codec, encoder, autoJoinSessions, framerate);
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
        bool runtimeMetricsEnabled = false;

        JST_SERDES(logLevel, latencyEnabled, runtimeMetricsEnabled);
    };

    struct Benchmark {
        std::string format = "markdown";

        JST_SERDES(format);
    };

    Graphics graphics;
    Remote remote;
    Interface interface;
    Developer developer;
    Benchmark benchmark;

    JST_SERDES(graphics, remote, interface, developer, benchmark);

    static Result Get(Settings& settings);
    static Result Set(const Settings& settings, bool persist = true);

 private:
    struct Impl;
};

}  // namespace Jetstream

#endif  // JETSTREAM_SETTINGS_HH
