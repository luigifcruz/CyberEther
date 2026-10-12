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

    struct Intelligence {
        struct Provider {
            std::string name;
            std::string host;
            std::string api;
            std::string chatPath;
            std::string modelsPath;
            std::string apiKeyId;
            bool requestReasoning = false;

            JST_SERDES(name, host, api, chatPath, modelsPath, apiKeyId, requestReasoning);
        };

        struct Model {
            std::string provider;
            std::string id;
            std::string name;
            U64 contextLength = 0;
            U64 maxOutputTokens = 0;
            std::string reasoningEffort;
            std::string reasoningContext;
            U64 outputLength = 0;
            std::vector<std::string> reasoningEfforts;

            JST_SERDES(provider, id, name, contextLength, maxOutputTokens, reasoningEffort, reasoningContext, outputLength,
                       reasoningEfforts);
        };

        struct Voice {
            std::string model;
            std::string voice;
            Model backend;

            JST_SERDES(model, voice, backend);
        };

        bool enabled = true;
        std::string provider;
        std::string model;
        std::vector<Model> models;
        std::vector<Provider> customProviders;
        Voice voice;

        JST_SERDES(enabled, provider, model, models, customProviders, voice);
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
    Intelligence intelligence;
    Developer developer;
    Benchmark benchmark;
    Registry registry;
    Runtime runtime;

    JST_SERDES(graphics, remote, interface, intelligence, developer, registry, runtime);

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
